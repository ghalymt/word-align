"""scripts/release_notes.py: the release workflow's notes come from CHANGELOG.md."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import release_notes  # noqa: E402

from wordalign import __version__  # noqa: E402

CHANGELOG = """# Changelog

## 2.1.0 — 2026-10-01

Intro line one
continues here.

### Added

- **A feature** that wraps
  onto a second line (#1).
- Another one (#2).

## 2.0.0 — 2026-08-10

Old notes.
"""


class TestReleaseNotes(unittest.TestCase):

    def test_section_is_cut_at_the_next_version(self):
        notes = release_notes.release_notes("2.1.0", CHANGELOG)
        self.assertTrue(notes.startswith("Intro line one"))
        self.assertIn("Another one (#2).", notes)
        self.assertNotIn("Old notes", notes)
        self.assertNotIn("## 2.0.0", notes)

    def test_hard_wrapped_lines_are_joined(self):
        notes = release_notes.unwrap(release_notes.release_notes("2.1.0", CHANGELOG))
        self.assertIn("Intro line one continues here.", notes)
        self.assertIn("- **A feature** that wraps onto a second line (#1).", notes)
        self.assertIn("\n- Another one (#2).", notes)
        self.assertIn("\n### Added\n", notes)

    def test_missing_version_fails(self):
        with self.assertRaises(KeyError):
            release_notes.release_notes("9.9.9", CHANGELOG)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(release_notes.main(["9.9.9"]), 1)

    def test_current_version_has_a_changelog_entry(self):
        # The release workflow refuses a tag whose version has no notes;
        # catch it here, before tagging.
        text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        self.assertTrue(release_notes.release_notes(__version__, text).strip())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(release_notes.main([f"v{__version__}"]), 0)
        self.assertTrue(out.getvalue().strip())


if __name__ == "__main__":
    unittest.main()
