"""The CI workflow keeps running the tests on Linux and Windows."""
import unittest
from pathlib import Path


class TestCiWorkflow(unittest.TestCase):
    """main broke from a fresh clone once because nothing ran the tests."""

    WORKFLOW = (Path(__file__).resolve().parent.parent /
                ".github" / "workflows" / "ci.yml")

    def test_workflow_runs_pytest_on_linux_and_windows(self):
        self.assertTrue(self.WORKFLOW.exists(), "CI workflow is missing")
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("ubuntu-latest", text)
        self.assertIn("windows-latest", text)
        self.assertIn("pytest", text)

    def test_workflow_runs_on_pull_requests_and_main(self):
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^on:\s*$")
        self.assertRegex(text, r"(?m)^  pull_request:")
        self.assertRegex(text, r"(?m)^  push:")

    def test_workflow_installs_the_package_with_dev_extra(self):
        # Installing the project itself (not just requirements.txt) is what
        # exercises packaging, so a package missing from the repo fails here.
        text = self.WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('pip install -e ".[dev]"', text)



if __name__ == "__main__":
    unittest.main()
