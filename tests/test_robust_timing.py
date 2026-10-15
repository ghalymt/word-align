"""Stage 5 of the robust flow: anchors, forced-alignment windows, fallbacks."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.timing import (AlignWindow, CONF_FA_CONFIRMED, CONF_INTERPOLATED,
                              build_anchors, check_window, enforce_order,
                              interpolate, match_source, plan_windows,
                              run_anchor_alignment, windows_from_chunks)
from wordalign.utils import normalize_word

TEXT = ("good morning everyone doctor smith will open the session today we have "
        "a lot to cover so please keep your questions short")


def timed(text, t0=0.0, step=0.4, shift=0.0, dur=0.3):
    return [{"word": w, "start": round(t0 + i * step + shift, 3),
             "end": round(t0 + i * step + shift + dur, 3)}
            for i, w in enumerate(text.split())]


def known(text):
    return [normalize_word(w) for w in text.split()]


class TestAnchors(unittest.TestCase):

    def test_runs_and_short_matches(self):
        src = timed("good morning everyone XX smith will open")
        m = match_source(known("good morning everyone doctor smith"), src, "Vosk")
        self.assertEqual(sorted(m.runs), [0, 1, 2])
        self.assertEqual(sorted(m.short), [4])        # "smith" alone

    def test_backwards_match_is_dropped(self):
        words = known("the cat sat on the mat")
        src = timed("the cat sat on the mat")
        src[4] = dict(src[4], start=-1.0 + 0.0, end=0.1)   # "the" times run backwards
        src[4]["start"] = 0.0
        m = match_source(words, src, "Vosk")
        self.assertNotIn(4, m.runs)

    def test_strong_when_two_sources_agree(self):
        words = known(TEXT)
        vosk = match_source(words, timed(TEXT), "Vosk")
        qwen = match_source(words, timed(TEXT, shift=0.08), "Qwen3-ASR")
        far = match_source(words, timed(TEXT, shift=0.6), "WhisperX")
        anchors = build_anchors(len(words), [vosk, qwen])
        self.assertTrue(all(a.strong for a in anchors.values()))
        anchors = build_anchors(len(words), [vosk, far])
        self.assertFalse(any(a.strong for a in anchors.values()))

    def test_windows_end_on_anchors_and_stay_short(self):
        n = 200
        words = " ".join(f"w{i}" for i in range(n))
        src = timed(words, step=0.5)
        m1 = match_source(known(words), src, "Vosk")
        m2 = match_source(known(words), timed(words, step=0.5, shift=0.05), "Qwen3-ASR")
        anchors = build_anchors(n, [m1, m2])
        wins = plan_windows(n, anchors, duration=100.0)
        self.assertGreater(len(wins), 2)
        self.assertEqual(wins[0].first, 0)
        self.assertEqual(wins[-1].last, n - 1)
        for a, b in zip(wins, wins[1:]):
            self.assertEqual(b.first, a.last + 1)
        self.assertTrue(all(w.end - w.start <= 61.0 for w in wins))


class TestWindowChecks(unittest.TestCase):

    def test_collapse_and_disagreement(self):
        win = AlignWindow(0, 0, 5, 0.0, 10.0)
        ok = {"times": [[i * 0.4, i * 0.4 + 0.3] for i in range(6)]}
        self.assertTrue(check_window(ok, win, {})[0])
        collapsed = {"times": [[1.0, 1.0]] * 6}
        self.assertFalse(check_window(collapsed, win, {})[0])
        partial = {"times": [None] * 4 + [[1.0, 1.2], [1.3, 1.5]]}
        self.assertFalse(check_window(partial, win, {})[0])
        from wordalign.timing import Anchor
        anchors = {i: Anchor(i, (i * 0.4 + 2.0, i * 0.4 + 2.3), ["Vosk"], False) for i in range(6)}
        self.assertFalse(check_window(ok, win, anchors)[0])


class TestFallbacks(unittest.TestCase):

    def test_sentence_start_leans_to_the_next_word(self):
        words = [{"word": "begin?", "start": 13.8, "end": 14.1},
                 {"word": "Great,"},
                 {"word": "let's", "start": 15.2, "end": 15.6}]
        self.assertEqual(interpolate(words), 1)
        self.assertGreater(words[1]["start"], 14.5)
        self.assertAlmostEqual(words[1]["end"], 15.2, delta=0.01)
        self.assertEqual(words[1]["timing_conf"], CONF_INTERPOLATED)
        self.assertTrue(words[1]["interpolated"])

    def test_vad_speech_pulls_words_into_speech(self):
        words = [{"word": "one", "start": 0.0, "end": 0.3}, {"word": "two"},
                 {"word": "three", "start": 9.0, "end": 9.3}]
        interpolate(words, speech=[(0.0, 0.35), (5.0, 5.6), (9.0, 9.3)])
        self.assertGreaterEqual(words[1]["start"], 4.9)
        self.assertLessEqual(words[1]["end"], 5.7)

    def test_enforce_order(self):
        words = [{"word": "a", "start": 1.0, "end": 1.5, "timing_conf": 0.9},
                 {"word": "b", "start": 1.2, "end": 1.6, "timing_conf": 0.9}]
        self.assertEqual(enforce_order(words), 1)
        self.assertGreaterEqual(words[1]["start"], 1.5)
        self.assertLess(words[1]["timing_conf"], 0.9)


class TestAlignerChecks(unittest.TestCase):

    def _setup(self, fa_shift):
        from wordalign.timing import assign_times
        words = [{"word": w} for w in TEXT.split()]
        kn = known(TEXT)
        qwen = match_source(kn, timed(TEXT), "Qwen3-ASR")
        park = match_source(kn, timed(TEXT, shift=0.05), "parakeet-ultra")
        anchors = build_anchors(len(words), [qwen, park])
        aligned = {i: ((sp[0] + fa_shift, sp[1] + fa_shift), "Qwen3-FA")
                   for i, sp in qwen.runs.items()}
        assign_times(words, [qwen, park], anchors, aligned)
        return words

    def test_far_aligner_loses_to_two_engines(self):
        words = self._setup(fa_shift=-2.0)
        self.assertNotEqual(words[3]["source"], "Qwen3-FA")
        self.assertAlmostEqual(words[3]["start"], timed(TEXT)[3]["start"], delta=0.06)

    def test_qwen_asr_does_not_confirm_its_own_aligner(self):
        from wordalign.timing import assign_times
        words = [{"word": w} for w in TEXT.split()]
        qwen = match_source(known(TEXT), timed(TEXT), "Qwen3-ASR")
        anchors = build_anchors(len(words), [qwen])
        aligned = {i: (sp, "Qwen3-FA") for i, sp in qwen.runs.items()}
        assign_times(words, [qwen], anchors, aligned)
        self.assertTrue(all(w["timing_conf"] < CONF_FA_CONFIRMED for w in words))
        words = self._setup(fa_shift=0.02)            # parakeet is independent
        self.assertTrue(all(w["timing_conf"] == CONF_FA_CONFIRMED for w in words))

    def test_crosstalk_line_does_not_push_the_main_speaker(self):
        """Reference transcript: A's line, then B's line spoken over A's next
        line, then A's next line. The aligner lays B over A's speech."""
        from wordalign.timing import (assign_times, enforce_order, interpolate,
                                      repair_degenerate)
        a1 = "here she would stay comforted and soothed"
        b = "but the more forgetfulness had then prevailed"
        a2 = "every chance she could steal after practice hours"
        text = f"{a1} {b} {a2}"
        words = [{"word": w} for w in text.split()]
        kn = known(text)
        # Engines hear only speaker A: a1 at 0 s, a2 at 5 s.
        heard = timed(a1, t0=0.0) + timed(a2, t0=5.0)
        vosk = match_source(kn, heard, "Vosk")
        wx = match_source(kn, [dict(w, start=w["start"] + 0.03, end=w["end"] + 0.03)
                               for w in heard], "WhisperX")
        anchors = build_anchors(len(words), [vosk, wx])
        n1, nb = len(a1.split()), len(b.split())
        # The aligner put B's words over A's second line (5.0-7.8 s) and
        # crushed a2 after it.
        aligned = {}
        for k in range(n1):
            aligned[k] = ((k * 0.4, k * 0.4 + 0.3), "Qwen3-FA")
        for k in range(nb):
            aligned[n1 + k] = ((5.0 + k * 0.4, 5.0 + k * 0.4 + 0.3), "Qwen3-FA")
        for k in range(len(a2.split())):
            aligned[n1 + nb + k] = ((8.0 + k * 0.01, 8.0 + k * 0.01), "Qwen3-FA")
        assign_times(words, [vosk, wx], anchors, aligned)
        interpolate(words, 12.0)
        enforce_order(words)
        repair_degenerate(words, 12.0)
        enforce_order(words)
        a2_words = words[n1 + nb:]
        for k, w in enumerate(a2_words):
            self.assertAlmostEqual(w["start"], 5.0 + k * 0.4, delta=0.1, msg=w)
        b_words = words[n1:n1 + nb]
        self.assertTrue(all(w.get("fa_dropped") for w in b_words))
        self.assertTrue(all(w["timing_conf"] < 0.5 for w in b_words))

    def test_zero_length_words_get_room(self):
        from wordalign.timing import repair_degenerate
        words = [{"word": "firs", "start": 108.99, "end": 109.41},
                 {"word": "did", "start": 109.57, "end": 109.57, "timing_conf": 0.95},
                 {"word": "not", "start": 109.57, "end": 109.57, "timing_conf": 0.95},
                 {"word": "think", "start": 109.81, "end": 110.05}]
        self.assertEqual(repair_degenerate(words), 2)
        for a, b in zip(words, words[1:]):
            self.assertLessEqual(a["end"], b["start"] + 1e-6)
        self.assertTrue(all(w["end"] - w["start"] >= 0.04 for w in words))
        self.assertTrue(words[1]["squeezed"] and words[1]["timing_conf"] < 0.5)
        # No room at all: still a minimum length, order kept by enforce_order.
        tight = [{"word": "a", "start": 1.0, "end": 1.0}, {"word": "b", "start": 1.0, "end": 1.0},
                 {"word": "c", "start": 1.0, "end": 1.2}]
        repair_degenerate(tight)
        enforce_order(tight)
        self.assertTrue(all(w["end"] > w["start"] for w in tight))
        self.assertTrue(all(a["end"] <= b["start"] + 1e-6 for a, b in zip(tight, tight[1:])))


class TestChunkWindows(unittest.TestCase):

    def test_split_chunks_get_no_window(self):
        words = [{"word": f"w{i}", "chunk": c}
                 for i, c in enumerate([0, 0, 1, 1, None, 1, 2, 2])]
        wins = windows_from_chunks(words, [(0, 10), (10, 20), (20, 30)], target=15)
        self.assertEqual([(w.first, w.last) for w in wins], [(0, 1), (6, 7)])


class TestRunAnchorAlignment(unittest.TestCase):

    def _fake_aligner(self, shift=0.0, fail_ids=()):
        def align(jobs):
            out = []
            for j in jobs:
                if j["id"] in fail_ids:
                    out.append({"id": j["id"], "times": [[1.0, 1.0]] * len(j["words"])})
                    continue
                n = len(j["words"])
                step = (j["end"] - j["start"]) / max(1, n)
                out.append({"id": j["id"], "status": "ok", "times": [
                    [round(j["start"] + k * step + shift, 3),
                     round(j["start"] + k * step + shift + 0.25, 3)] for k in range(n)]})
            return out
        return align

    def test_fa_confirmed_by_sources(self):
        words = [{"word": w} for w in TEXT.split()]
        src = timed(TEXT)
        index = {w["word"]: i for i, w in enumerate(src)}

        def align(jobs):
            return [{"id": j["id"], "times": [[src[index[x]]["start"], src[index[x]]["end"]]
                                              for x in j["words"]]} for j in jobs]

        stats = run_anchor_alignment(
            words, {"Vosk": src, "Qwen3-ASR": timed(TEXT, shift=0.05)}, duration=12.0,
            aligner=align, log=lambda m: None)
        self.assertGreaterEqual(stats["windows_ok"], 1)
        self.assertTrue(all(w["source"] == "Qwen3-FA" for w in words))
        self.assertTrue(all(w["timing_conf"] == CONF_FA_CONFIRMED for w in words))

    def test_failed_window_falls_back_to_anchors(self):
        words = [{"word": w} for w in TEXT.split()]
        stats = run_anchor_alignment(words, {"Vosk": timed(TEXT)}, duration=12.0,
                                     aligner=self._fake_aligner(fail_ids={0}),
                                     log=lambda m: None)
        self.assertEqual(stats["windows_ok"], 0)
        self.assertTrue(stats["window_failures"])
        self.assertTrue(all(w["source"] == "Vosk" for w in words))

    def test_retry_on_cleaned_track(self):
        words = [{"word": w} for w in TEXT.split()]
        stats = run_anchor_alignment(words, {}, duration=12.0,
                                     aligner=self._fake_aligner(fail_ids={0}),
                                     retry_aligner=self._fake_aligner(),
                                     log=lambda m: None)
        self.assertEqual(stats["windows_retried"], 1)
        self.assertTrue(all(w["source"].endswith("(cleaned)") for w in words))

    def test_unavailable_aligner(self):
        words = [{"word": w} for w in TEXT.split()]
        stats = run_anchor_alignment(words, {"Vosk": timed(TEXT)}, duration=12.0,
                                     aligner=lambda jobs: None, log=lambda m: None)
        self.assertEqual(stats.get("aligner"), "unavailable")
        self.assertTrue(all(w.get("start") is not None for w in words))


class TestPipelineAnchorMode(unittest.TestCase):
    """End to end with recorded engines: anchors + a fake Qwen aligner."""

    def _run(self, transcript, extra_patches, **cfg_kwargs):
        from wordalign.config import PipelineConfig
        from wordalign.core.pipeline import PipelineRunner
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, True))
        audio = Path(d) / "a.wav"
        audio.write_bytes(b"x")
        tpath = None
        if transcript:
            tpath = Path(d) / "t.txt"
            tpath.write_text(transcript, encoding="utf-8")
        flags = dict(use_vosk=True, use_qwen=False, use_whisperx=False, use_mfa=False,
                     language="en")
        flags.update(cfg_kwargs)
        cfg = PipelineConfig(audio_path=str(audio), transcript_path=str(tpath) if tpath else None,
                             output_dir=d, **flags)
        patches = [
            patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}),
            patch("wordalign.engines.vosk_engine.VOSK_AVAILABLE", True),
            patch("wordalign.config.PipelineConfig.vosk_model_path",
                  lambda self, language: "/models/vosk"),
            patch("wordalign.core.pipeline.get_audio_duration", lambda p: 12.0),
            patch("wordalign.utils.get_audio_duration", lambda p: 12.0),
            *extra_patches,
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return PipelineRunner(cfg).run(), d

    def test_reference_mode_uses_forced_alignment(self):
        vosk = timed(TEXT)

        def fake_align(audio, jobs, language, **kw):
            return [{"id": j["id"], "status": "ok", "times": [
                [v["start"] + 0.01, v["end"] + 0.01]
                for v in vosk[:len(j["words"])]]} for j in jobs]

        result, d = self._run(TEXT, [
            patch("wordalign.engines.vosk_engine.run_vosk_parallel", lambda *a, **k: vosk),
            patch("wordalign.engines.qwen_engine.qwen_align", fake_align),
            patch("wordalign.engines.qwen_engine._resolve_python", lambda p: "python"),
        ], use_qwen=True)
        self.assertIsNone(result.error)
        sources = {w.timing_source for w in result.aligned_words}
        self.assertEqual(sources, {"Qwen3-FA"})
        self.assertTrue(result.review_json_path and os.path.exists(result.review_json_path))

    def test_reference_mode_without_aligner_keeps_engine_times(self):
        vosk = timed(TEXT)
        result, _ = self._run(TEXT, [
            patch("wordalign.engines.vosk_engine.run_vosk_parallel", lambda *a, **k: vosk)],
            forced_align=False)
        self.assertEqual({w.timing_source for w in result.aligned_words}, {"Vosk"})
        self.assertAlmostEqual(result.aligned_words[0].start, 0.0)

    def test_fill_only_keeps_the_v21_waterfall(self):
        vosk = timed(TEXT)
        result, _ = self._run(TEXT, [
            patch("wordalign.engines.vosk_engine.run_vosk_parallel", lambda *a, **k: vosk)],
            timing_strategy="fill_only")
        self.assertEqual({w.timing_source for w in result.aligned_words}, {"Vosk"})
        self.assertIsNone(result.review_json_path)


if __name__ == "__main__":
    unittest.main()
