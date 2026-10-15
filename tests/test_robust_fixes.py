"""Regressions from the 2.2 review: cancel during a voter, engine switches,
aligner fallback, overlap speakers, partial results, review edit times."""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHUNKS = [(0.5, 4.5), (6.0, 10.0)]
SAID = ["we sailed to the island", "and stayed there for a week"]


def _qmap():
    from wordalign.audio.analysis import ChunkInfo, QualityMap
    q = QualityMap(duration=12.0, vad_probs=np.full(int(12 / 0.032), 0.9, np.float32))
    q.speech = list(CHUNKS)
    q.chunks = [ChunkInfo(s, e, snr_db=20.0) for s, e in CHUNKS]
    return q


def _timed(text, t0):
    return [{"word": w, "start": round(t0 + i * 0.5, 3), "end": round(t0 + i * 0.5 + 0.4, 3)}
            for i, w in enumerate(text.split())]


def _fake_qwen(audio_path, language=None, chunks=None, return_payload=False, **kw):
    pieces = [{"status": "ok", "words": _timed(t, s)} for t, (s, _) in zip(SAID, chunks)]
    return {"words": [w for p in pieces for w in p["words"]], "chunks": pieces}


def _fake_cohere(audio_path, chunks, language, models_dir=None, **kw):
    return {i: [{"word": w} for w in t.split()] for i, t in enumerate(SAID)}


class _PipelineCase(unittest.TestCase):

    def run_pipeline(self, extra_patches=(), runner_hook=None, **cfg_kwargs):
        from wordalign.config import PipelineConfig
        from wordalign.core.pipeline import PipelineRunner
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(d, True))
        audio = os.path.join(d, "a.wav")
        Path(audio).write_bytes(b"x")
        kwargs = dict(use_vosk=False, use_qwen=True, use_whisperx=False, use_mfa=False,
                      voters=("cohere", "qwen"), sound_events=False)
        kwargs.update(cfg_kwargs)
        cfg = PipelineConfig(audio_path=audio, output_dir=d, language="en", **kwargs)
        runner = PipelineRunner(cfg)
        if runner_hook:
            runner_hook(runner)
        patches = [
            patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}),
            patch("wordalign.audio.io.load_audio", lambda p, **k: np.zeros(12 * 16000, np.float32)),
            patch("wordalign.audio.analysis.analyze_audio", lambda *a, **k: _qmap()),
            patch("wordalign.engines.voters.voter_available", lambda n, l: (True, "")),
            patch("wordalign.engines.qwen_engine.run_qwen", _fake_qwen),
            patch("wordalign.core.pipeline.get_audio_duration", lambda p: 12.0),
            patch("wordalign.utils.get_audio_duration", lambda p: 12.0),
            *extra_patches,
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return runner, runner.run()


class TestCancelInVoter(_PipelineCase):

    def test_worker_cancel_is_a_cancel(self):
        from wordalign.engines._worker import WorkerCancelled
        holder = {}

        def cancelling_cohere(*a, **k):
            holder["runner"].cancel()
            raise WorkerCancelled("Cohere Transcribe: cancelled")

        runner, result = self.run_pipeline(
            [patch("wordalign.engines.voters.run_cohere", cancelling_cohere)],
            runner_hook=lambda r: holder.update(runner=r))
        self.assertTrue(result.cancelled)
        self.assertTrue(runner.cancelled)


class TestEngineSwitches(unittest.TestCase):

    def test_switched_off_engines_do_not_vote(self):
        from wordalign.config import PipelineConfig
        from wordalign.core.pipeline import PipelineRunner
        cfg = PipelineConfig(audio_path="a.wav", profile_name="robust",
                             engine_overrides={"whisperx": False, "qwen": False})
        runner = PipelineRunner(cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner._voter_list(), ["cohere", "parakeet"])


class TestVoterPriors(unittest.TestCase):

    def test_profile_sets_the_weights(self):
        from wordalign.config import PipelineConfig
        from wordalign.core.config import PipelineProfile
        from wordalign.core.pipeline import PipelineRunner
        prof = PipelineProfile.from_dict({"name": "x", "robust": {
            "voters": ["cohere", "whisperx"], "voter_priors": {"whisperx": 0.9}}})
        cfg = PipelineConfig(audio_path="a.wav")
        PipelineRunner(cfg, profile=prof)
        self.assertEqual(cfg.voter_priors, {"whisperx": 0.9})


class TestAlignerFallback(_PipelineCase):

    def test_wav2vec2_when_the_qwen_venv_is_missing(self):
        calls = []

        truth = [w for t, (s, _) in zip(SAID, CHUNKS) for w in _timed(t, s)]

        def fake_wav2vec(audio_path, jobs, language, **kw):
            calls.append(len(jobs))
            out, k = [], 0
            for j in jobs:
                n = len(j["words"])
                out.append({"id": j["id"], "status": "ok", "times": [
                    [w["start"] + 0.01, w["end"] + 0.01] for w in truth[k:k + n]]})
                k += n
            return out

        _, result = self.run_pipeline([
            patch("wordalign.engines.voters.run_cohere", _fake_cohere),
            patch("wordalign.engines.qwen_engine._resolve_python", lambda p: None),
            patch("wordalign.engines.whisperx_engine.wav2vec2_available", lambda: True),
            patch("wordalign.engines.whisperx_engine.wav2vec2_align", fake_wav2vec),
        ], use_whisperx=True)
        self.assertTrue(calls)
        self.assertIn("wav2vec2", {w.timing_source for w in result.aligned_words})


class TestPartialResults(_PipelineCase):

    def test_failed_pieces_are_not_cached(self):
        runs = []

        def flaky_qwen(audio_path, language=None, chunks=None, return_payload=False, **kw):
            runs.append(1)
            out = _fake_qwen(audio_path, language, chunks)
            out["chunks"][1] = {"status": "failed", "words": []}
            out["failed"] = [1]
            return out

        patches = [patch("wordalign.engines.voters.run_cohere", _fake_cohere),
                   patch("wordalign.engines.qwen_engine.run_qwen", flaky_qwen)]
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(d, True))
        cache = patch.dict(os.environ, {"WORDALIGN_CACHE": d})
        with cache:
            self.run_pipeline(patches)
            self.run_pipeline(patches)
        self.assertEqual(len(runs), 2)          # the second run did not replay a gap


class TestOverlapSpeakers(unittest.TestCase):

    def test_inserted_words_keep_their_speaker(self):
        from wordalign.diarize import assign_speakers
        turns = [(0.0, 10.0, "SPEAKER_00"), (4.0, 6.0, "SPEAKER_01")]
        words = [{"word": "hello", "start": 4.5, "end": 5.5},
                 {"word": "yes", "start": 4.6, "end": 5.4, "speaker": "Speaker 2",
                  "overlap_added": True}]
        assign_speakers(words, turns)
        self.assertEqual(words[0]["speaker"], "Speaker 1")
        self.assertEqual(words[1]["speaker"], "Speaker 2")

    def test_names_follow_the_file_turns(self):
        from wordalign.engines.sedicow_engine import merge_overlap_words
        voted = [{"word": w, "start": 1.0 + i * 0.5, "end": 1.4 + i * 0.5}
                 for i, w in enumerate("we sailed to the island".split())]
        windows = [{"start": 0.0, "end": 5.0, "speakers": [
            {"label": "SPEAKER_01", "segments": [{"start": 1.0, "end": 3.4,
                                                    "text": "we sailed to the island"}]},
            {"label": "SPEAKER_00", "segments": [{"start": 2.0, "end": 2.6, "text": "really"}]}]}]
        turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 5.0, "SPEAKER_01")]
        merged, added = merge_overlap_words(voted, windows, [(1.5, 3.0)], turns=turns)
        self.assertEqual(added, 1)
        extra = [w for w in merged if w.get("overlap_added")]
        self.assertEqual(extra[0]["speaker"], "Speaker 1")    # SPEAKER_00 speaks first


class TestReviewTimes(unittest.TestCase):

    def test_nan_inf_and_huge_times_are_refused(self):
        from wordalign.gui import review
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, True))
        (d / "x_review.json").write_text(json.dumps({"cues": [
            {"index": 1, "start": 1.0, "end": 2.0, "text": "hi"}]}), encoding="utf-8")
        for bad in ("NaN", "inf", 1e12, "1e400"):
            with self.assertRaises(ValueError):
                review.save_edits(d, {"1": {"status": "edited", "text": "hi", "start": bad}})
        self.assertFalse((d / "x_review_edits.json").exists())


if __name__ == "__main__":
    unittest.main()
