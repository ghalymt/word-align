"""Stage 3 of the robust flow: ROVER voting, voter plumbing, hallucination filter."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.engines.voters import bucket_timed_words, chunks_from_qwen_payload
from wordalign.qa.hallucination import (filter_hallucinations, filter_text_chunk,
                                        find_loops)
from wordalign.rover import build_network, vote


def W(text, t0=None, step=0.4):
    out = []
    for i, w in enumerate(text.split()):
        d = {"word": w}
        if t0 is not None:
            d.update(start=round(t0 + i * step, 3), end=round(t0 + i * step + 0.3, 3))
        out.append(d)
    return out


class TestRover(unittest.TestCase):

    def test_unanimous(self):
        out = vote({"cohere": W("The quick brown fox."), "qwen": W("the quick brown fox"),
                    "parakeet": W("The quick brown fox")})
        self.assertEqual([w["word"] for w in out], ["The", "quick", "brown", "fox."])
        self.assertTrue(all(w["agreement"] == 1.0 for w in out))

    def test_text_only_voter_can_correct(self):
        # Cohere has no timestamps but still outvotes a single wrong engine.
        out = vote({"cohere": W("we sailed to the island"),
                    "parakeet": W("we sailed to the island", 0.0),
                    "qwen": W("we sale to the island", 0.0)})
        self.assertEqual([w["word"] for w in out], "we sailed to the island".split())
        sailed = out[1]
        self.assertLess(sailed["agreement"], 1.0)
        self.assertIn("sale", sailed["alternatives"])
        self.assertIsNotNone(sailed.get("start"))      # timed by a timed voter

    def test_majority_drops_an_inserted_word(self):
        out = vote({"cohere": W("thanks for coming"), "qwen": W("thanks for coming"),
                    "whisperx": W("thanks for coming thank you")})
        self.assertEqual([w["word"] for w in out], ["thanks", "for", "coming"])

    def test_empty_hypothesis_votes_for_silence(self):
        out = vote({"cohere": [], "qwen": [], "whisperx": W("thank you")})
        self.assertEqual(out, [])

    def test_failed_engine_is_left_out(self):
        out = vote({"cohere": W("hello world"), "qwen": None})
        self.assertEqual([w["word"] for w in out], ["hello", "world"])

    def test_network_aligns_insertions(self):
        slots, placed = build_network({"cohere": W("a b c"), "qwen": W("a x b c")},
                                      {"cohere": 1.0, "qwen": 0.9})
        self.assertEqual(placed, ["cohere", "qwen"])
        self.assertEqual(len(slots), 4)


class TestVoterPlumbing(unittest.TestCase):

    def test_bucket_by_midpoint(self):
        words = W("a b c d", 0.0, step=5.0)            # 0, 5, 10, 15 s
        per, orphans = bucket_timed_words(words, [(0.0, 6.0), (9.0, 12.0)])
        self.assertEqual([w["word"] for w in per[0]], ["a", "b"])
        self.assertEqual([w["word"] for w in per[1]], ["c"])
        self.assertEqual([w["word"] for w in orphans], ["d"])

    def test_qwen_payload_chunks(self):
        payload = {"chunks": [{"status": "ok", "words": W("a b", 0.0)},
                              {"status": "failed", "words": []}]}
        out = chunks_from_qwen_payload(payload, 2)
        self.assertEqual(sorted(out), [0])
        self.assertIsNone(chunks_from_qwen_payload(None, 2))


def speech(level, spans=()):
    def fn(s, e):
        for a, b in spans:
            if s < b and e > a:
                return 0.9
        return level
    return fn


class TestHallucinationFilter(unittest.TestCase):

    def test_long_loops_go_whatever_the_audio(self):
        words = W("you " * 9, 0.0)
        kept, removed = filter_hallucinations(words, speech(0.9), speech(0.9))
        self.assertEqual(len(kept), 1)
        self.assertTrue(all("loop" in r["reason"] for r in removed))

    def test_short_repeats_in_real_speech_stay(self):
        words = W("no no no I said", 0.0)
        kept, _ = filter_hallucinations(words, speech(0.9), speech(0.9))
        self.assertEqual(len(kept), 5)

    def test_short_repeats_in_low_speech_go(self):
        words = W("the end the end the end", 0.0)
        kept, removed = filter_hallucinations(words, speech(0.1), speech(0.2))
        self.assertLess(len(kept), 6)
        self.assertTrue(removed)

    def test_stock_phrase_island_in_non_speech(self):
        words = W("we will start now", 0.0) + W("thank you for watching", 10.0)
        kept, removed = filter_hallucinations(
            words, speech(0.05, [(0.0, 2.0)]), speech(0.05, [(0.0, 2.0)]))
        self.assertEqual([w["word"] for w in kept], "we will start now".split())
        self.assertTrue(any("stock" in r["reason"] or "no speech" in r["reason"]
                            for r in removed))

    def test_real_thank_you_is_kept(self):
        words = W("thank you", 3.0)
        kept, removed = filter_hallucinations(words, speech(0.95), speech(0.95))
        self.assertEqual(len(kept), 2)
        self.assertEqual(removed, [])

    def test_no_vad_means_only_sure_loops(self):
        words = W("thank you", 3.0)
        kept, _ = filter_hallucinations(words, None, None)
        self.assertEqual(len(kept), 2)

    def test_untimed_chunk_text(self):
        self.assertEqual(filter_text_chunk("Thank you.", 0.1), ("", "stock phrase in non-speech"))
        self.assertEqual(filter_text_chunk("Thank you.", 0.9), ("Thank you.", None))
        text, reason = filter_text_chunk("go " * 10, 0.9)
        self.assertEqual(text.split(), ["go"])
        self.assertIn("loop", reason)

    def test_find_loops(self):
        self.assertEqual(find_loops(["a", "b", "a", "b", "a", "b", "c"]), [(0, 2, 3)])


class TestTranscriptFreePipeline(unittest.TestCase):
    """End to end without a transcript: VAD chunks, three fake voters, ROVER,
    forced alignment on the voted chunks, review file."""

    CHUNKS = [(0.5, 4.5), (6.0, 10.0)]
    SAID = ["we sailed to the island", "and stayed there for a week"]

    def _qmap(self):
        import numpy as np
        from wordalign.audio.analysis import ChunkInfo, QualityMap
        q = QualityMap(duration=12.0, vad_probs=np.full(int(12 / 0.032), 0.9, np.float32))
        q.speech = list(self.CHUNKS)
        q.chunks = [ChunkInfo(s, e, snr_db=20.0) for s, e in self.CHUNKS]
        return q

    def _timed(self, text, t0):
        return [{"word": w, "start": round(t0 + i * 0.5, 3), "end": round(t0 + i * 0.5 + 0.4, 3)}
                for i, w in enumerate(text.split())]

    def test_vote_align_and_review(self):
        import contextlib
        import io
        import json
        import os
        import tempfile
        from unittest.mock import patch
        import numpy as np
        from wordalign.config import PipelineConfig
        from wordalign.core.pipeline import PipelineRunner

        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, True))
        audio = os.path.join(d, "a.wav")
        Path(audio).write_bytes(b"x")
        cfg = PipelineConfig(audio_path=audio, output_dir=d, language="en",
                             use_vosk=False, use_qwen=True, use_whisperx=False, use_mfa=False,
                             voters=("cohere", "qwen", "parakeet"), sound_events=False)
        qwen_said = ["we sale to the island", "and stayed there for a week"]
        calls = {"align": []}

        def fake_qwen(audio_path, language=None, chunks=None, return_payload=False, **kw):
            pieces = [{"status": "ok", "words": self._timed(t, s)}
                      for t, (s, _) in zip(qwen_said, chunks)]
            return {"words": [w for p in pieces for w in p["words"]], "chunks": pieces}

        def fake_cohere(audio_path, chunks, language, models_dir=None, **kw):
            return {i: [{"word": w} for w in t.split()] for i, t in enumerate(self.SAID)}

        def fake_parakeet(audio_path, chunks, models_dir=None, **kw):
            return {i: self._timed(t, s) for i, (t, (s, _)) in enumerate(zip(self.SAID, chunks))}

        truth = [w for t, (s, _) in zip(self.SAID, self.CHUNKS) for w in self._timed(t, s)]

        def fake_align(audio_path, jobs, language, **kw):
            # Windows cover consecutive words, so a running pointer finds them.
            calls["align"].append(jobs)
            out, k = [], 0
            for j in jobs:
                n = len(j["words"])
                out.append({"id": j["id"], "status": "ok", "times": [
                    [w["start"] + 0.02, w["end"] + 0.02] for w in truth[k:k + n]]})
                k += n
            return out

        patches = [
            patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}),
            patch("wordalign.audio.io.load_audio", lambda p, **k: np.zeros(12 * 16000, np.float32)),
            patch("wordalign.audio.analysis.analyze_audio", lambda *a, **k: self._qmap()),
            patch("wordalign.engines.voters.voter_available", lambda n, l: (True, "")),
            patch("wordalign.engines.voters.run_cohere", fake_cohere),
            patch("wordalign.engines.voters.run_parakeet", fake_parakeet),
            patch("wordalign.engines.qwen_engine.run_qwen", fake_qwen),
            patch("wordalign.engines.qwen_engine.qwen_align", fake_align),
            patch("wordalign.engines.qwen_engine._resolve_python", lambda p: "python"),
            patch("wordalign.core.pipeline.get_audio_duration", lambda p: 12.0),
            patch("wordalign.utils.get_audio_duration", lambda p: 12.0),
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            result = PipelineRunner(cfg).run()

        self.assertIsNone(result.error)
        text = " ".join(w.text for w in result.aligned_words)
        self.assertEqual(text, " ".join(self.SAID))          # ROVER fixed "sale"
        # Transcript-free windows come from the voted VAD chunks.
        self.assertTrue(calls["align"])
        starts = sorted(j["start"] for jobs in calls["align"] for j in jobs)
        self.assertAlmostEqual(starts[0], 0.5, delta=0.3)
        self.assertEqual({w.timing_source for w in result.aligned_words}, {"Qwen3-FA"})
        review = json.loads(Path(result.review_json_path).read_text(encoding="utf-8"))
        self.assertTrue(review["cues"])
        sailed = next(w for c in review["cues"] for w in c["words"] if w["word"] == "sailed")
        self.assertLess(sailed["text_conf"], 1.0)
        self.assertIn("sale", sailed["alternatives"])


if __name__ == "__main__":
    unittest.main()
