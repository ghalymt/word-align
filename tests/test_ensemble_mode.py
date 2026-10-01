"""Ensemble (transcript-free) mode: cue segmentation and timing refinement."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.align import match_timestamps
from wordalign.config import PipelineConfig
from wordalign.core.pipeline import PipelineRunner
from wordalign.utils import normalize_word

WHISPERX = "wordalign.engines.whisperx_engine.run_whisperx"
QWEN = "wordalign.engines.qwen_engine.run_qwen"
TEXT = ("Hello there everyone. Today we are going to talk about subtitles and "
        "timing. It is a surprisingly deep topic with many edge cases to "
        "consider carefully.")


def _timed(text, start=0.0, step=0.4, shift=0.0):
    words, t = [], start
    for w in text.split():
        words.append({"word": w, "start": t + shift, "end": t + shift + 0.3,
                      "conf": 0.9})
        t += step
    return words


class _EnsembleCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.audio = Path(tmp.name) / "audio.wav"
        self.audio.write_bytes(b"not-real-audio")
        env = patch.dict(os.environ,
                         {"WORDALIGN_CACHE": str(Path(tmp.name) / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def _run(self, cfg, patches):
        runner = PipelineRunner(cfg)
        with contextlib.ExitStack() as stack:
            for target, value in patches.items():
                stack.enter_context(patch(target, value))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return runner.run()

    def _cfg(self, **engines):
        flags = dict(use_vosk=False, use_qwen=False, use_whisperx=False,
                     use_mfa=False)
        flags.update(engines)
        return PipelineConfig(audio_path=str(self.audio), output_dir=self.dir,
                              **flags)


class TestPrimaryTranscriptSegmentation(_EnsembleCase):

    def test_primary_transcript_is_split_into_cues(self):
        # Regression: the primary text was one line, so the whole file
        # became a single subtitle cue.
        words = _timed(TEXT)
        result = self._run(
            self._cfg(use_whisperx=True),
            {WHISPERX: lambda *a, **k: (words, [])})
        self.assertGreaterEqual(len(result.segments), 3)
        for seg in result.segments:
            lines = seg.text.split("\n")
            self.assertLessEqual(len(lines), 2, seg.text)
            for line in lines:
                self.assertLessEqual(len(line), 42, seg.text)
        joined = " ".join(s.text.replace("\n", " ") for s in result.segments)
        self.assertEqual(joined.split(), TEXT.split())


class TestPrimaryTimingRefinement(_EnsembleCase):

    def _vosk_patches(self, vosk_words):
        return {
            "wordalign.engines.vosk_engine.VOSK_AVAILABLE": True,
            "wordalign.engines.vosk_engine.run_vosk_parallel":
                lambda *a, **k: vosk_words,
            "wordalign.config.PipelineConfig.vosk_model_path":
                lambda self, language: "/models/vosk-en",
        }

    def test_vosk_refines_primary_timings(self):
        # Regression: the primary run marked every word matched, so the Vosk
        # and Qwen passes in ensemble mode never changed anything.
        whisper = _timed(TEXT)
        vosk = _timed(TEXT, shift=0.12)
        result = self._run(
            self._cfg(use_whisperx=True, use_vosk=True),
            {WHISPERX: lambda *a, **k: (whisper, []),
             **self._vosk_patches(vosk)})
        self.assertEqual({w.timing_source for w in result.aligned_words},
                         {"Vosk"})
        self.assertAlmostEqual(result.aligned_words[0].start, 0.12)

    def test_refinement_never_relocates_a_word(self):
        whisper = _timed(TEXT)
        vosk = _timed(TEXT, shift=0.12)
        vosk[3] = dict(vosk[3], start=vosk[3]["start"] + 5.0,
                       end=vosk[3]["end"] + 5.0)
        result = self._run(
            self._cfg(use_whisperx=True, use_vosk=True),
            {WHISPERX: lambda *a, **k: (whisper, []),
             **self._vosk_patches(vosk)})
        self.assertEqual(result.aligned_words[3].timing_source, "WhisperX")
        self.assertAlmostEqual(result.aligned_words[3].start, whisper[3]["start"])
        self.assertEqual(result.aligned_words[4].timing_source, "Vosk")

    def test_failed_qwen_is_not_rerun_for_refinement(self):
        calls = []

        def fake_qwen(*args, **kwargs):
            calls.append(1)
            return []

        result = self._run(
            self._cfg(use_whisperx=True, use_qwen=True, use_vosk=True),
            {WHISPERX: lambda *a, **k: ([], []),
             QWEN: fake_qwen,
             **self._vosk_patches(_timed(TEXT))})
        self.assertEqual(len(calls), 1)
        self.assertTrue(result.segments)


class TestMatchTimestampsReplaceSources(unittest.TestCase):

    def _words(self):
        return [{"word": "hello", "start": 1.0, "end": 1.2, "matched": True,
                 "source": "WhisperX"},
                {"word": "world", "start": 1.3, "end": 1.6, "matched": True,
                 "source": "SRT"}]

    def test_default_never_overwrites_matched_words(self):
        aligned = self._words()
        src = [{"word": "hello", "start": 1.1, "end": 1.3},
               {"word": "world", "start": 1.4, "end": 1.7}]
        stats = match_timestamps(aligned, src, "Vosk",
                                 [normalize_word(w["word"]) for w in aligned])
        self.assertEqual(stats["matched"], 0)
        self.assertEqual(aligned[0]["source"], "WhisperX")

    def test_only_listed_sources_are_replaced(self):
        aligned = self._words()
        src = [{"word": "hello", "start": 1.1, "end": 1.3},
               {"word": "world", "start": 1.4, "end": 1.7}]
        match_timestamps(aligned, src, "Vosk",
                         [normalize_word(w["word"]) for w in aligned],
                         replace_sources={"WhisperX"})
        self.assertEqual(aligned[0]["source"], "Vosk")
        self.assertEqual(aligned[0]["start"], 1.1)
        self.assertEqual(aligned[1]["source"], "SRT")


if __name__ == "__main__":
    unittest.main()
