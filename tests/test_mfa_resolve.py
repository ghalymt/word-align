"""Finding and running MFA: conda environments before PATH, run as if activated."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.engines import mfa_engine


def make_env(root: Path) -> Path:
    exe = root / "Scripts" / "mfa.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    (root / "conda-meta").mkdir()
    return exe


class TestResolveMfa(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))
        env = {k: v for k, v in os.environ.items() if k not in ("WORDALIGN_MFA", "CONDA_PREFIX")}
        p = patch.dict(os.environ, env, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def test_conda_env_beats_a_pip_mfa_on_path(self):
        env_root = self.tmp / "venvs" / "mfa"
        exe = make_env(env_root)
        with patch.object(mfa_engine, "_candidate_env_roots", lambda: [env_root]), \
                patch("shutil.which", lambda name: r"C:\Users\x\AppData\Roaming\Python\Scripts\mfa.EXE"):
            found, env = mfa_engine.resolve_mfa()
        self.assertEqual(found, str(exe))
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertTrue(env["PATH"].startswith(str(env_root)))
        self.assertIn(str(env_root / "Library" / "bin"), env["PATH"])

    def test_wordalign_mfa_env_root(self):
        env_root = self.tmp / "mfa_env"
        exe = make_env(env_root)
        with patch.dict(os.environ, {"WORDALIGN_MFA": str(env_root)}):
            found, env = mfa_engine.resolve_mfa()
        self.assertEqual(found, str(exe))
        self.assertEqual(env["CONDA_PREFIX"], str(env_root))

    def test_wordalign_mfa_exe_inside_an_env_is_activated_too(self):
        env_root = self.tmp / "mfa_env"
        exe = make_env(env_root)
        with patch.dict(os.environ, {"WORDALIGN_MFA": str(exe)}):
            found, env = mfa_engine.resolve_mfa()
        self.assertEqual(found, str(exe))
        self.assertEqual(env.get("PYTHONNOUSERSITE"), "1")

    def test_path_only_when_no_env(self):
        with patch.object(mfa_engine, "_candidate_env_roots", lambda: []), \
                patch("shutil.which", lambda name: "/usr/bin/mfa"):
            found, env = mfa_engine.resolve_mfa()
        self.assertEqual(found, "/usr/bin/mfa")
        self.assertEqual(env.get("CONDA_PREFIX"), None)

    def test_nothing_found(self):
        with patch.object(mfa_engine, "_candidate_env_roots", lambda: []), \
                patch("shutil.which", lambda name: None):
            self.assertIsNone(mfa_engine.resolve_mfa()[0])


class TestMfaChunks(unittest.TestCase):

    def _words(self):
        return [{"word": "a", "start": 0.0, "end": 0.3},            # voter time, not "matched"
                {"word": "b"},                                        # untimed
                {"word": "c", "start": 1.0, "end": 1.2, "matched": False},
                {"word": "d", "start": 1.3, "end": 1.5},
                {"word": "e", "start": 1.6, "end": 1.8},
                {"word": "f", "start": 1.9, "end": 2.1},
                {"word": "g", "start": 2.2, "end": 2.4},
                {"word": "h", "start": 2.5, "end": 2.7},
                {"word": "i", "start": 2.8, "end": 3.0},
                {"word": "j"}, {"word": "k"},
                {"word": "l", "start": 4.0, "end": 4.3}]

    def test_gaps_with_context(self):
        from wordalign.align import plan_mfa_chunks
        chunks = plan_mfa_chunks(self._words(), 5.0, context=2)
        self.assertEqual([c["gap_indices"] for c in chunks], [[1], [9, 10]])
        self.assertEqual(chunks[0]["text"], "a b c d")
        self.assertAlmostEqual(chunks[0]["start_offset"], 0.0)     # a starts at 0
        self.assertAlmostEqual(chunks[0]["end_offset"], 1.75)      # d ends at 1.5 (+0.25)
        self.assertEqual(chunks[1]["text"], "h i j k l")
        self.assertAlmostEqual(chunks[1]["end_offset"], 4.55)

    def test_overlapping_context_merges(self):
        from wordalign.align import plan_mfa_chunks
        chunks = plan_mfa_chunks(self._words(), 5.0, context=3)
        self.assertEqual(len(chunks), 2)
        chunks = plan_mfa_chunks(self._words(), 5.0, context=4)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["gap_indices"], [1, 9, 10])

    def test_only_gap_words_take_mfa_times(self):
        from types import SimpleNamespace as IV
        from wordalign.align import apply_mfa_intervals, plan_mfa_chunks
        words = self._words()
        chunk = plan_mfa_chunks(words, 5.0, context=2)[0]
        # MFA dropped "c" (an unknown word): the rest still lines up.
        ivs = [IV(mark="a", minTime=0.0, maxTime=0.3), IV(mark="b", minTime=0.5, maxTime=0.8),
               IV(mark="d", minTime=1.3, maxTime=1.5)]
        self.assertEqual(apply_mfa_intervals(words, chunk, ivs), 1)
        self.assertEqual((words[1]["start"], words[1]["source"]), (0.5, "MFA"))
        self.assertEqual(words[0]["start"], 0.0)
        self.assertNotIn("source", words[0])


if __name__ == "__main__":
    unittest.main()
