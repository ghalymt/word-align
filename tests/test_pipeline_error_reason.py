"""A run that stops early says why in PipelineResult.error and .warnings."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.config import PipelineConfig
from wordalign.core.pipeline import PipelineRunner

CLIP = Path(__file__).resolve().parent / "golden" / "audio" / "espeak_en.flac"
ENGINES_OFF = dict(use_vosk=False, use_qwen=False, use_whisperx=False,
                   use_mfa=False)


class TestEarlyExitReason(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        env = patch.dict(os.environ, {"WORDALIGN_CACHE": str(self.dir / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def run_pipeline(self, **kwargs):
        cfg = PipelineConfig(output_dir=str(self.dir / "out"), language="en",
                             **ENGINES_OFF, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return PipelineRunner(cfg).run()

    def test_missing_audio(self):
        # Regression: result.error stayed None, so the GUI job said
        # "Pipeline produced no output" and the batch report "no output
        # produced", without the reason.
        missing = str(self.dir / "nope.wav")
        result = self.run_pipeline(audio_path=missing)
        self.assertEqual(result.error, f"Audio not found: {missing}")
        self.assertIn(f"init: Audio not found: {missing}", result.warnings)

    def test_no_engine_and_no_transcript(self):
        result = self.run_pipeline(audio_path=str(CLIP))
        self.assertEqual(result.output_files, [])
        self.assertIn("ASR engines produced no transcript", result.error)

    def test_successful_run_has_no_error(self):
        transcript = self.dir / "t.txt"
        transcript.write_text("The weather will be cold and windy.\n",
                              encoding="utf-8")
        result = self.run_pipeline(audio_path=str(CLIP),
                                   transcript_path=str(transcript))
        self.assertTrue(result.output_files)
        self.assertIsNone(result.error)

    def test_cli_prints_the_reason(self):
        from wordalign.cli_v2 import main
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main([str(self.dir / "nope.wav"), "-o", str(self.dir),
                         "--no-vosk", "--no-qwen", "--no-whisperx", "--no-mfa"])
        self.assertEqual(code, 1)
        self.assertIn("[error] Audio not found:", out.getvalue())
        self.assertNotIn("Pipeline produced no output", out.getvalue())


if __name__ == "__main__":
    unittest.main()
