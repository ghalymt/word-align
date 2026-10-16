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


class TestOnlyUntimedWords(unittest.TestCase):

    def test_timed_words_are_not_realigned(self):
        import contextlib
        import io
        from wordalign import align
        words = [{"word": "a", "start": 0.0, "end": 0.3},            # voter time, not "matched"
                 {"word": "b"},                                        # untimed
                 {"word": "c", "start": 1.0, "end": 1.2, "matched": False},
                 {"word": "d"}, {"word": "e"},
                 {"word": "f", "start": 2.0, "end": 2.3, "matched": True}]
        sliced = []

        class Wrapper:
            corpus_dir = Path(tempfile.mkdtemp())
            output_dir = corpus_dir

            def missing_models(self):
                return []

            def detect_oov_words(self):
                raise RuntimeError("stop after slicing")

            def cleanup(self):
                pass

        self.addCleanup(lambda: __import__("shutil").rmtree(Wrapper.corpus_dir, True))
        with patch.object(align, "create_chunk_wav", lambda *a, **k: sliced.append(a)), \
                patch.object(align, "get_audio_duration", lambda p: 3.0), \
                contextlib.redirect_stdout(io.StringIO()):
            align.run_mfa_on_gaps(words, "audio.wav", Wrapper())
        self.assertEqual(len(sliced), 2)                 # gaps [b] and [d, e] only
        labs = sorted(p.read_text() for p in Wrapper.corpus_dir.glob("*.lab"))
        self.assertEqual(labs, ["b", "d e"])
        self.assertEqual(words[0]["start"], 0.0)


if __name__ == "__main__":
    unittest.main()
