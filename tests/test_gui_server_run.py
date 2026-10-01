"""GUI server: run parameters and multipart uploads."""
import http.client
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.types import PipelineResult
from wordalign.gui import server as gui_server
from wordalign.gui.server import PipelineAPIHandler


class TestRunEngineFlags(unittest.TestCase):
    """The GUI's engine checkboxes must win over the selected preset."""

    def _effective_config(self, params):
        seen = {}

        def fake_run(runner):
            seen["config"] = runner.config
            return PipelineResult(cancelled=True)

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
                "XDG_DATA_HOME": d, "LOCALAPPDATA": d}), \
                patch("wordalign.core.pipeline.PipelineRunner.run", fake_run):
            from wordalign.core.database import JobRecord, ProjectRecord, ProjectStore
            store = ProjectStore()
            store.create_project(ProjectRecord(id="p1", name="p",
                                               audio_path="a.wav"))
            store.create_job(JobRecord(id="job1", project_id="p1"))
            store.close()
            PipelineAPIHandler._run_pipeline_thread(None, "job1", params)
        return seen["config"]

    def test_unticked_engines_do_not_run_with_a_preset(self):
        # Regression: the GUI sends use_qwen/use_mfa, the server only read
        # qwen/mfa, so the "balanced" preset re-enabled unticked engines.
        cfg = self._effective_config({
            "audio_path": "a.wav", "profile": "balanced",
            "use_vosk": True, "use_qwen": False, "use_whisperx": True,
            "use_mfa": False})
        self.assertFalse(cfg.use_qwen)
        self.assertFalse(cfg.use_mfa)
        self.assertTrue(cfg.use_vosk)
        self.assertTrue(cfg.use_whisperx)

    def test_ticked_engine_runs_even_if_preset_omits_it(self):
        cfg = self._effective_config({
            "audio_path": "a.wav", "profile": "cpu_only",
            "use_vosk": True, "use_qwen": False, "use_whisperx": True,
            "use_mfa": False})
        self.assertTrue(cfg.use_whisperx)

    def test_bare_engine_keys_still_work(self):
        cfg = self._effective_config({
            "audio_path": "a.wav", "profile": "balanced", "qwen": False})
        self.assertFalse(cfg.use_qwen)


class TestMultipartUpload(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        class QuietHandler(PipelineAPIHandler):
            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _upload(self, files):
        boundary = "----wordalignTestBoundary7MA4YWxk"
        body = b""
        for name, data in files:
            body += (f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="file"; '
                     f'filename="{name}"\r\n'
                     "Content-Type: application/octet-stream\r\n\r\n").encode()
            body += data + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        conn = http.client.HTTPConnection("127.0.0.1",
                                          self.server.server_address[1],
                                          timeout=10)
        try:
            conn.request("POST", "/api/upload", body=body, headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(len(body))})
            payload = json.loads(conn.getresponse().read())
        finally:
            conn.close()
        self.addCleanup(self._discard, payload)
        return payload

    @staticmethod
    def _discard(payload):
        upload_dir = gui_server._UPLOAD_DIRS.pop(payload.get("upload_id"), None)
        if upload_dir:
            shutil.rmtree(upload_dir, ignore_errors=True)

    def test_trailing_crlf_and_dashes_in_file_survive(self):
        # Regression: rstrip(b"\r\n-") also removed the file's own trailing
        # CR/LF/'-' bytes (e.g. a transcript ending in a newline).
        data = b"line one\r\nline two --\r\n\r\n"
        payload = self._upload([("transcript.txt", data)])
        self.assertTrue(payload["ok"])
        self.assertEqual(Path(payload["files"][0]).read_bytes(), data)

    def test_binary_file_bytes_are_preserved(self):
        audio = bytes(range(256)) + b"\r\n-\n\r-"
        text = "hello\n".encode()
        payload = self._upload([("a.wav", audio), ("t.txt", text)])
        saved = sorted(payload["files"])
        self.assertEqual(Path(saved[0]).read_bytes(), audio)
        self.assertEqual(Path(saved[1]).read_bytes(), text)


if __name__ == "__main__":
    unittest.main()
