"""Tests for unified ModelPaths configuration (all ASR model dirs)."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wordalign.models.paths import ModelPaths


class TestModelPaths(unittest.TestCase):

    def test_env_var_defaults(self):
        """Fields default to their environment variables."""
        os.environ["WORDALIGN_WHISPER_MODELS"] = "C:/models/whisper"
        try:
            mp = ModelPaths()
            self.assertEqual(mp.whisper_models_dir, "C:/models/whisper")
        finally:
            del os.environ["WORDALIGN_WHISPER_MODELS"]

    def test_auto_detect_fills_all(self):
        """auto_detect() should fill every field (or leave None when absent)."""
        mp = ModelPaths.auto_detect()
        # Every field must be a string or None — never crash
        for field in ("vosk_models_dir", "whisper_models_dir", "qwen_models_dir",
                      "huggingface_cache_dir", "mfa_models_dir"):
            val = getattr(mp, field)
            self.assertTrue(val is None or isinstance(val, str), field)

    def test_custom_path_override(self):
        """custom_paths overrides everything for that model type."""
        with tempfile.TemporaryDirectory() as td:
            mp = ModelPaths(custom_paths={"whisper": td})
            resolved = mp.resolve("whisper")
            self.assertEqual(resolved, Path(td))

    def test_resolve_typed_field(self):
        """resolve() returns the typed field when it exists on disk."""
        with tempfile.TemporaryDirectory() as td:
            mp = ModelPaths(vosk_models_dir=td)
            resolved = mp.resolve("vosk")
            self.assertEqual(resolved, Path(td))

    def test_resolve_missing_returns_none(self):
        """resolve() returns None when nothing is configured."""
        mp = ModelPaths()
        mp._fill_auto()
        # Point all to a non-existent dir and force no fallback
        mp.vosk_models_dir = "Z:/definitely/not/here"
        mp.custom_paths.clear()
        self.assertIsNone(mp.resolve("vosk"))

    def test_resolve_model_whisper(self):
        """resolve_model finds whisper .pt files."""
        with tempfile.TemporaryDirectory() as td:
            Path(td, "large-v3.pt").write_text("x")
            mp = ModelPaths(whisper_models_dir=td)
            p = mp.resolve_model("whisper", "large-v3")
            self.assertIsNotNone(p)
            self.assertEqual(p.name, "large-v3.pt")

    def test_resolve_model_qwen_hf_format(self):
        """resolve_model handles HF cache layout for Qwen."""
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td, "models--Qwen--Qwen3-ASR-1.7B")
            model_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}")
            mp = ModelPaths(qwen_models_dir=td)
            p = mp.resolve_model("qwen", "Qwen/Qwen3-ASR-1.7B")
            self.assertIsNotNone(p)
            self.assertTrue("Qwen3-ASR" in str(p))

    def test_summary_contains_all(self):
        """summary() lists every model type."""
        mp = ModelPaths.auto_detect()
        s = mp.summary()
        for label in ("Vosk", "Whisper", "Qwen", "HF Cache", "MFA"):
            self.assertIn(label, s)

    def test_pipeline_config_uses_model_paths(self):
        """PipelineConfig.vosk_models_dir delegates to model_paths."""
        from wordalign.config import PipelineConfig
        os.environ["WORDALIGN_VOSK_MODELS"] = "C:/custom/vosk"
        try:
            cfg = PipelineConfig("test.mp4")
            self.assertEqual(cfg.vosk_models_dir, "C:/custom/vosk")
            self.assertEqual(cfg.model_paths.vosk_models_dir, "C:/custom/vosk")
            # Setter delegates too
            cfg.vosk_models_dir = "C:/other/vosk"
            self.assertEqual(cfg.model_paths.vosk_models_dir, "C:/other/vosk")
        finally:
            del os.environ["WORDALIGN_VOSK_MODELS"]

    def test_cli_flags_set_model_paths(self):
        """CLI --whisper-models etc. land in model_paths."""
        from wordalign.cli_v2 import _parse_args
        cfg = _parse_args(["test.mp3", "--whisper-models", "D:/w",
                           "--hf-cache", "D:/hf", "--vosk-models", "D:/v",
                           "--mfa-models", "D:/m"])
        self.assertEqual(cfg.model_paths.whisper_models_dir, "D:/w")
        self.assertEqual(cfg.model_paths.huggingface_cache_dir, "D:/hf")
        self.assertEqual(cfg.model_paths.vosk_models_dir, "D:/v")
        self.assertEqual(cfg.model_paths.mfa_models_dir, "D:/m")

    def test_to_dict_roundtrip(self):
        """to_dict() includes custom paths."""
        mp = ModelPaths(custom_paths={"yamnet": "C:/yam"})
        d = mp.to_dict()
        self.assertEqual(d["custom_paths"], {"yamnet": "C:/yam"})
        self.assertIn("vosk_models_dir", d)


if __name__ == "__main__":
    unittest.main()
