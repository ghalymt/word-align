"""Runtime robustness: console encoding, temp dirs, env, tool discovery."""
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wordalign import align
from wordalign.core.manifest import generate_manifest
from wordalign.engines import whisperx_engine
from wordalign.models import paths


class TestCliOnLegacyConsoleEncoding(unittest.TestCase):

    def test_cli_completes_with_cp1252_stdout(self):
        # Regression: redirected stdout on Windows is cp1252, which has no
        # "→"; a successful run crashed while listing its output files.
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "audio.wav"
            audio.write_bytes(b"not-real-audio")
            transcript = Path(d) / "transcript.txt"
            transcript.write_text("Hello there.\nGeneral Kenobi.\n",
                                  encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="cp1252",
                       WORDALIGN_CACHE=os.path.join(d, "cache"),
                       PYTHONPATH=str(ROOT))
            proc = subprocess.run(
                [sys.executable, "-m", "wordalign", str(audio),
                 "-t", str(transcript), "-o", d, "--no-vosk", "--no-qwen",
                 "--no-whisperx", "--no-mfa"],
                capture_output=True, env=env, timeout=120)
            stderr = proc.stderr.decode("cp1252", "replace")
            self.assertEqual(proc.returncode, 0, stderr[-2000:])
            self.assertNotIn("UnicodeEncodeError", stderr)
            self.assertTrue((Path(d) / "audio_sentence_level.srt").exists())


class TestSurgicalMfaTempDir(unittest.TestCase):

    def _run(self, wrapper_cls, aligned):
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "mfa_surgical_test"
            work.mkdir()
            with patch.object(align.tempfile, "mkdtemp",
                              return_value=str(work)), \
                    patch.object(align, "MFAWrapper", wrapper_cls), \
                    patch("sys.stdout", new=open(os.devnull, "w")) as out:
                align.make_surgical_mfa(aligned, "audio.wav")
                out.close()
            return work.exists()

    def test_temp_dir_removed_when_mfa_missing(self):
        def missing(*args, **kwargs):
            raise FileNotFoundError("Could not locate the MFA executable.")
        self.assertFalse(self._run(missing, [{"word": "a"}]))

    def test_temp_dir_removed_when_there_are_no_gaps(self):
        class Wrapper:
            def __init__(self, work_dir, **kwargs):
                Path(work_dir, "corpus").mkdir(parents=True, exist_ok=True)
        aligned = [{"word": "a", "start": 0.0, "end": 0.2, "matched": True}]
        self.assertFalse(self._run(Wrapper, aligned))


class TestInProcessWhisperxRestoresHfHome(unittest.TestCase):

    def _fake_modules(self):
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False,
                                           empty_cache=lambda: None)
        whisperx = types.ModuleType("whisperx")

        def load_model(*args, **kwargs):
            raise RuntimeError("no model in tests")
        whisperx.load_model = load_model
        return {"torch": torch, "whisperx": whisperx}

    def _call(self):
        with patch.dict(sys.modules, self._fake_modules()), \
                patch("sys.stdout", new=open(os.devnull, "w")) as out, \
                patch("sys.stderr", new=open(os.devnull, "w")) as err:
            result = whisperx_engine._run_whisperx_inproc(
                "a.wav", "en", "tiny", "cpu", "/tmp/whisper-models")
            out.close()
            err.close()
        return result

    def test_previous_hf_home_is_restored(self):
        with patch.dict(os.environ, {"HF_HOME": "/original/hf"}):
            self.assertEqual(self._call(), ([], []))
            self.assertEqual(os.environ["HF_HOME"], "/original/hf")

    def test_unset_hf_home_stays_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HOME", None)
            self._call()
            self.assertNotIn("HF_HOME", os.environ)


class TestLlamaCppDiscovery(unittest.TestCase):

    def test_unix_llama_cli_is_found(self):
        with tempfile.TemporaryDirectory() as d, \
                patch.object(paths, "_project_models_root",
                             return_value=Path(d)), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORDALIGN_LLM_ENGINE", None)
            engine = Path(d) / "llama.cpp"
            engine.mkdir()
            self.assertIsNone(paths._llama_cpp_dir())
            (engine / "llama-cli").write_text("")
            self.assertEqual(paths._llama_cpp_dir(), str(engine))


class TestManifest(unittest.TestCase):

    def test_generated_at_without_deprecation_warning(self):
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "a.wav"
            audio.write_bytes(b"x")
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                manifest = generate_manifest(str(audio))
        self.assertTrue(manifest["generated_at"].endswith("Z"))
        self.assertNotIn("+", manifest["generated_at"])

    def test_pipeline_manifest_records_timestamp_sources(self):
        from wordalign.config import PipelineConfig
        from wordalign.core.pipeline import PipelineRunner
        with tempfile.TemporaryDirectory() as d, patch.dict(
                os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            audio = Path(d) / "audio.wav"
            audio.write_bytes(b"x")
            transcript = Path(d) / "t.txt"
            transcript.write_text("one two three", encoding="utf-8")
            cfg = PipelineConfig(
                audio_path=str(audio), transcript_path=str(transcript),
                output_dir=d, use_vosk=False, use_qwen=False,
                use_whisperx=False, use_mfa=False)
            with patch("sys.stdout", new=open(os.devnull, "w")) as out:
                result = PipelineRunner(cfg).run()
                out.close()
            data = json.loads(Path(result.job_manifest_path).read_text(
                encoding="utf-8"))
        self.assertEqual(data["word_stats"],
                         {"total": 3, "sources": {"Interpolated": 3}})


if __name__ == "__main__":
    unittest.main()
