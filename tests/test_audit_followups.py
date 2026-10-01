"""Follow-up audit fixes: subprocess encoding, legacy ensemble, caching."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.config import PipelineConfig
from wordalign.core.converters import dicts_to_words
from wordalign.core.events import CollectSink, ErrorEvent
from wordalign.core.pipeline import PipelineRunner
from wordalign.engines import qwen_engine, whisperx_engine
from wordalign.models import paths
from wordalign.qa.deterministic import detect_timing_anomalies

FAKE_WORKER = textwrap.dedent('''
    import json, sys
    # Non-Latin log line: on Windows a piped child without UTF-8 stdio
    # crashes here with UnicodeEncodeError.
    print("загрузка модели 模型 — été", file=sys.stderr, flush=True)
    payload = {"language": "fr", "words": [
        {"word": "été", "start": 0.0, "end": 0.4, "conf": 0.9},
        {"word": "模型", "start": 0.5, "end": 0.9, "conf": 0.9}]}
    sys.stdout.write("\\n" + sys.argv[1] + json.dumps(payload, ensure_ascii=False) + "\\n")
''')


class _Quiet:
    def __enter__(self):
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self._stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return self

    def __exit__(self, *exc):
        self._stack.close()


class TestWorkerSubprocessEncoding(unittest.TestCase):

    def _fake_worker(self, d, sentinel):
        script = Path(d) / "worker.py"
        # The sentinel is passed as argv[1]; the real flags follow and are
        # ignored by the fake.
        script.write_text(
            "import sys; sys.argv[1] = %r\n" % sentinel + FAKE_WORKER,
            encoding="utf-8")
        return script

    def test_qwen_worker_output_is_utf8(self):
        with tempfile.TemporaryDirectory() as d:
            worker = self._fake_worker(d, qwen_engine.SENTINEL)
            with patch.object(qwen_engine, "_WORKER", worker), _Quiet():
                words = qwen_engine.run_qwen("a.wav", "fr",
                                             qwen_python=sys.executable)
        self.assertEqual([w["word"] for w in words], ["été", "模型"])

    def test_whisperx_worker_output_is_utf8(self):
        with tempfile.TemporaryDirectory() as d:
            worker = self._fake_worker(d, whisperx_engine.SENTINEL)
            with patch.object(whisperx_engine, "_WORKER", worker), \
                    patch.dict(os.environ,
                               {"WORDALIGN_WHISPERX_PYTHON": sys.executable}), \
                    _Quiet():
                words, _ = whisperx_engine.run_whisperx("a.wav", "fr")
        self.assertEqual([w["word"] for w in words], ["été", "模型"])

    def test_workers_get_utf8_stdio(self):
        seen = {}

        class FakePopen:
            def __init__(self, args, **kwargs):
                seen.update(kwargs)
                self.returncode = 1

            def communicate(self, timeout=None):
                return "", ""

        with patch("subprocess.Popen", FakePopen), _Quiet():
            qwen_engine.run_qwen("a.wav", "fr", qwen_python=sys.executable)
        self.assertEqual(seen["encoding"], "utf-8")
        self.assertEqual(seen["env"]["PYTHONIOENCODING"], "utf-8")

    def test_llama_cli_output_is_decoded_as_utf8(self):
        from wordalign.qa.punctuation import LlamaCppRestorer
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, stdout="1. Été!", stderr="")

        restorer = LlamaCppRestorer(engine="x", model="m.gguf")
        with patch.object(LlamaCppRestorer, "is_ready", return_value=(True, "")), \
                patch.object(LlamaCppRestorer, "cli_path", return_value="llama-cli"), \
                patch("wordalign.qa.punctuation.subprocess.run", fake_run):
            out = restorer.restore_cues(["été"])
        self.assertEqual(seen["encoding"], "utf-8")
        self.assertEqual(out, ["Été!"])

    def test_mfa_subprocess_output_is_decoded_as_utf8(self):
        from wordalign.engines.mfa_engine import MFAWrapper
        seen = {}

        def fake_run(cmd, **kwargs):
            seen.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with tempfile.TemporaryDirectory() as d, \
                patch.object(MFAWrapper, "_resolve_mfa", return_value="mfa"), \
                patch("wordalign.engines.mfa_engine.subprocess.run", fake_run):
            MFAWrapper(work_dir=d).generate_custom_dictionary({"été"})
        self.assertEqual(seen["encoding"], "utf-8")


class TestLegacyEnsembleWithoutTimedEngines(unittest.TestCase):

    def test_reports_error_instead_of_crashing(self):
        # Regression: build_consensus raised ValueError, which escaped run().
        with tempfile.TemporaryDirectory() as d, patch.dict(
                os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            audio = Path(d) / "a.wav"
            audio.write_bytes(b"x")
            cfg = PipelineConfig(audio_path=str(audio), output_dir=d,
                                 use_vosk=False, use_qwen=True,
                                 use_whisperx=True, use_mfa=False,
                                 use_legacy_ensemble=True,
                                 ensemble_engines=("whisperx", "qwen"))
            sink = CollectSink()
            with patch("wordalign.engines.whisperx_engine.run_whisperx",
                       return_value=([], [])), \
                    patch("wordalign.engines.qwen_engine.run_qwen",
                          return_value=[]), _Quiet():
                result = PipelineRunner(cfg, sink=sink).run()
        errors = [e.message for e in sink.events if isinstance(e, ErrorEvent)]
        self.assertTrue(any("no transcript" in m for m in errors), errors)
        self.assertFalse(result.sentence_level_srt_path)


class TestPrimaryRunsAreCached(unittest.TestCase):

    def test_primary_qwen_result_is_reused(self):
        calls = []
        words = [{"word": "hello", "start": 0.0, "end": 0.4, "conf": 0.9},
                 {"word": "world.", "start": 0.5, "end": 0.9, "conf": 0.9}]

        def fake_qwen(*args, **kwargs):
            calls.append(1)
            return words

        with tempfile.TemporaryDirectory() as d, patch.dict(
                os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            audio = Path(d) / "a.wav"
            audio.write_bytes(b"x")
            for _ in range(2):
                cfg = PipelineConfig(audio_path=str(audio), output_dir=d,
                                     use_vosk=False, use_qwen=True,
                                     use_whisperx=False, use_mfa=False)
                with patch("wordalign.engines.qwen_engine.run_qwen", fake_qwen), \
                        _Quiet():
                    result = PipelineRunner(cfg).run()
                self.assertTrue(result.segments)
        self.assertEqual(len(calls), 1)


class TestDefaultLlmModelChoice(unittest.TestCase):

    def test_draft_and_projector_files_are_not_the_main_model(self):
        with tempfile.TemporaryDirectory() as d, \
                patch.object(paths, "_project_models_root", return_value=Path(d)), \
                patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORDALIGN_LLM_MODEL", None)
            llm = Path(d) / "llm"
            llm.mkdir()
            (llm / "qwen3-4b-q4.gguf").write_bytes(b"x" * 10)
            (llm / "mtp-qwen3-4b.gguf").write_bytes(b"x" * 50)
            (llm / "qwen3-mmproj-f16.gguf").write_bytes(b"x" * 90)
            self.assertEqual(Path(paths._default_llm_model_path()).name,
                             "qwen3-4b-q4.gguf")


class TestFastSpeechCheck(unittest.TestCase):

    def test_last_word_is_checked_too(self):
        words = dicts_to_words([
            {"word": "fine", "start": 0.0, "end": 0.3, "source": "Vosk"},
            {"word": "thanks", "start": 0.4, "end": 0.45, "source": "Vosk"}])
        ids = [i.id for i in detect_timing_anomalies(words)]
        self.assertIn("issue_timing_fast_0001", ids)


if __name__ == "__main__":
    unittest.main()
