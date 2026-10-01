"""Timing accuracy on real audio with a real engine (Vosk).

tests/golden/audio/espeak_en.flac is synthesised word by word, so every
word's true start/end is known exactly (see make_clip.py). The pipeline
aligns it with a real Vosk model, and the measured timing errors must stay
within bounds -- a regression that shifts or scrambles timings fails here
even when every unit test passes.

Needs a Vosk model: set WORDALIGN_TEST_VOSK_MODEL to an unpacked model
(CI uses vosk-model-small-en-us-0.15). Skipped otherwise, unless
WORDALIGN_REQUIRE_ACCURACY=1, which makes a missing model a failure.
"""
import contextlib
import io
import json
import os
import shutil
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIO_DIR = Path(__file__).resolve().parent / "golden" / "audio"
MODEL = os.environ.get("WORDALIGN_TEST_VOSK_MODEL")
REQUIRED = os.environ.get("WORDALIGN_REQUIRE_ACCURACY") == "1"

# Bounds on |engine - truth| in seconds, over the words Vosk timed.
MAX_MEDIAN_START_ERR = 0.10
MAX_P90_START_ERR = 0.20
MAX_MEDIAN_END_ERR = 0.15
MIN_VOSK_COVERAGE = 0.75


@unittest.skipUnless(MODEL or REQUIRED,
                     "set WORDALIGN_TEST_VOSK_MODEL to run the real-audio test")
class TestRealAudioAccuracy(unittest.TestCase):

    def setUp(self):
        if not MODEL or not Path(MODEL).is_dir():
            self.fail(f"Vosk model not found: {MODEL!r}")
        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            self.fail("ffmpeg/ffprobe are needed for the real-audio test")

    def test_vosk_word_timings_match_ground_truth(self):
        from wordalign.cli_v2 import _parse_args
        from wordalign.core.pipeline import PipelineRunner

        truth = json.loads((AUDIO_DIR / "espeak_en_truth.json").read_text("utf-8"))
        with tempfile.TemporaryDirectory() as d, patch.dict(
                os.environ, {"WORDALIGN_CACHE": os.path.join(d, "cache")}), \
                patch("wordalign.config.PipelineConfig.vosk_model_path",
                      lambda self, language: MODEL):
            cfg = _parse_args([str(AUDIO_DIR / "espeak_en.flac"),
                               "-t", str(AUDIO_DIR / "espeak_en.txt"),
                               "-o", d, "-l", "en", "--no-qwen",
                               "--no-whisperx", "--no-mfa"])
            with contextlib.redirect_stdout(io.StringIO()):
                result = PipelineRunner(cfg).run()

        words = result.aligned_words
        self.assertEqual([w.normalized_text for w in words],
                         [t["word"] for t in truth])
        timed = [(w, t) for w, t in zip(words, truth)
                 if w.timing_source == "Vosk"]
        coverage = len(timed) / len(truth)
        start_err = sorted(abs(w.start - t["start"]) for w, t in timed)
        end_err = sorted(abs(w.end - t["end"]) for w, t in timed)
        p90 = start_err[int(0.9 * (len(start_err) - 1))] if start_err else None
        report = (f"Vosk coverage {coverage:.0%}; start error median "
                  f"{statistics.median(start_err):.3f}s p90 {p90:.3f}s max "
                  f"{start_err[-1]:.3f}s; end error median "
                  f"{statistics.median(end_err):.3f}s"
                  if timed else f"Vosk coverage {coverage:.0%}")
        print(f"\n[accuracy] {report}")

        self.assertGreaterEqual(coverage, MIN_VOSK_COVERAGE, report)
        self.assertLessEqual(statistics.median(start_err), MAX_MEDIAN_START_ERR, report)
        self.assertLessEqual(p90, MAX_P90_START_ERR, report)
        self.assertLessEqual(statistics.median(end_err), MAX_MEDIAN_END_ERR, report)
        self.assertEqual(
            [s.text.replace("\n", " ") for s in result.segments],
            (AUDIO_DIR / "espeak_en.txt").read_text("utf-8").split("\n")[:3])


if __name__ == "__main__":
    unittest.main()
