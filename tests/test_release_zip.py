"""build.py --zip: the release ZIP leaves the model store out (known issue 5)."""
import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import build


class TestReleaseZip(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bundle = self.root / "dist" / "WordAlign"
        for rel, data in {"WordAlign.exe": b"MZ", "_internal/lib.dll": b"x",
                          "README.md": b"# hi",
                          "models/vosk/vosk-model-en/am.bin": b"0" * 4096,
                          "models/whisper/large-v3.bin": b"1" * 4096}.items():
            path = self.bundle / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

    def _zip(self, **kwargs):
        with patch.object(build, "ROOT", self.root), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            path = build.make_zip(self.bundle, **kwargs)
        with zipfile.ZipFile(path) as zf:
            return set(zf.namelist()), zf, out.getvalue()

    def test_models_are_left_out_by_default(self):
        # Regression: --zip --copy-models zipped the whole model store
        # (tens of GB; GitHub release assets are capped at 2 GiB).
        names, _, out = self._zip()
        self.assertIn("WordAlign/WordAlign.exe", names)
        self.assertIn("WordAlign/_internal/lib.dll", names)
        self.assertFalse(any("/models/" in n for n in names), names)
        self.assertIn("WordAlign/MODELS.txt", names)
        self.assertIn("models excluded", out)

    def test_models_can_be_included_explicitly(self):
        names, _, _ = self._zip(include_models=True)
        self.assertIn("WordAlign/models/vosk/vosk-model-en/am.bin", names)
        self.assertNotIn("WordAlign/MODELS.txt", names)

    def test_models_readme_explains_where_models_go(self):
        with patch.object(build, "ROOT", self.root), \
                contextlib.redirect_stdout(io.StringIO()):
            path = build.make_zip(self.bundle)
        text = zipfile.ZipFile(path).read("WordAlign/MODELS.txt").decode()
        for folder in ("vosk", "whisper", "huggingface", "mfa", "llm"):
            self.assertIn(f"models\\{folder}\\", text)

    def test_oversized_archive_warns(self):
        with patch.object(build, "GITHUB_ASSET_LIMIT", 10):
            _, _, out = self._zip(include_models=True)
        self.assertIn("2 GiB release-asset limit", out)


if __name__ == "__main__":
    unittest.main()
