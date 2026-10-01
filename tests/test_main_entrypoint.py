"""``python -m wordalign`` must fail loudly when the v2 CLI cannot import."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Make "import wordalign.cli_v2" fail (as a missing or broken subpackage
# would), then run the package entry point exactly like `python -m wordalign`.
SCRIPT = """
import runpy, sys
sys.modules["wordalign.cli_v2"] = None
sys.argv = ["wordalign", "--version"]
runpy.run_module("wordalign", run_name="__main__")
"""


class TestMainEntryPoint(unittest.TestCase):

    def test_import_error_is_not_masked_by_the_v1_cli(self):
        # Regression: __main__ caught any exception and silently ran the old
        # v1 CLI, which hid a checkout missing wordalign/models.
        proc = subprocess.run([sys.executable, "-c", SCRIPT], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("wordalign.cli_v2", proc.stderr)
        self.assertIn("Error", proc.stderr)

    def test_entry_point_runs_the_v2_cli(self):
        proc = subprocess.run([sys.executable, "-m", "wordalign", "--version"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(proc.stdout.startswith("wordalign "), proc.stdout)


if __name__ == "__main__":
    unittest.main()
