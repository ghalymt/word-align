"""CLI: explicit per-engine flags take precedence over --engines."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.pipeline import PipelineRunner


class TestEnginesFlagPrecedence(unittest.TestCase):
    """Explicit --use-mfa / --no-* flags must win over --engines."""

    @staticmethod
    def _config(*args):
        from wordalign.cli_v2 import _parse_args
        return PipelineRunner(_parse_args(["audio.wav", *args])).config

    def test_readme_example_runs_mfa(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk", "--use-mfa")
        self.assertTrue(cfg.use_mfa)
        self.assertTrue(cfg.use_whisperx)
        self.assertTrue(cfg.use_qwen)
        self.assertTrue(cfg.use_vosk)

    def test_flag_order_does_not_matter(self):
        cfg = self._config("--use-mfa", "--engines", "whisperx,qwen,vosk")
        self.assertTrue(cfg.use_mfa)

    def test_no_mfa_wins_over_engines_listing_mfa(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk,mfa", "--no-mfa")
        self.assertFalse(cfg.use_mfa)

    def test_no_engine_flags_win_over_engines_listing(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk",
                           "--no-vosk", "--no-qwen", "--no-whisperx")
        self.assertFalse(cfg.use_vosk)
        self.assertFalse(cfg.use_qwen)
        self.assertFalse(cfg.use_whisperx)

    def test_engines_alone_still_selects_exactly_those(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk")
        self.assertTrue(cfg.use_whisperx)
        self.assertFalse(cfg.use_mfa)
        cfg = self._config("--engines", "whisperx,mfa")
        self.assertTrue(cfg.use_mfa)
        self.assertFalse(cfg.use_vosk)
        self.assertFalse(cfg.use_qwen)



if __name__ == "__main__":
    unittest.main()
