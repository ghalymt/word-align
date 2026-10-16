"""The WhisperX worker finds the alignment model in the models folder or its hub/."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.engines import _whisperx_worker as W

NAME = "facebook/wav2vec2-large-960h-lv60-self"
FOLDER = "models--facebook--wav2vec2-large-960h-lv60-self"


class FakeWhisperX:
    def __init__(self, where):
        self.where, self.calls = where, []

    def load_align_model(self, language_code, device, model_name=None, model_dir=None):
        self.calls.append(model_dir)
        if model_name and model_dir and os.path.isdir(os.path.join(model_dir, FOLDER)):
            return "model", {"dir": model_dir}
        if model_name is None:
            return "auto", {}
        raise OSError("not here")


class TestAlignModelDirs(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.root, True))
        os.mkdir(os.path.join(self.root, "hub"))

    def test_model_in_hub_is_found(self):
        os.mkdir(os.path.join(self.root, "hub", FOLDER))
        self.assertEqual(W.align_model_dirs(self.root, NAME)[0], os.path.join(self.root, "hub"))
        fake = FakeWhisperX(self.root)
        model, meta = W._load_align_model(fake, "en", "cpu", self.root)
        self.assertEqual((model, meta["dir"]), ("model", os.path.join(self.root, "hub")))

    def test_model_at_root_is_found(self):
        os.mkdir(os.path.join(self.root, FOLDER))
        fake = FakeWhisperX(self.root)
        model, meta = W._load_align_model(fake, "en", "cpu", self.root)
        self.assertEqual(meta["dir"], self.root)

    def test_missing_model_falls_back_to_auto(self):
        fake = FakeWhisperX(self.root)
        self.assertEqual(W._load_align_model(fake, "en", "cpu", self.root)[0], "auto")

    def test_no_models_dir_uses_default_cache(self):
        self.assertEqual(W.align_model_dirs(None, NAME), [None])


if __name__ == "__main__":
    unittest.main()
