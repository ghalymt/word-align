"""GUI: server data is never parsed as HTML; JSON bodies are bounded."""
import http.client
import json
import re
import socket
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.gui.server import PipelineAPIHandler

APP_HTML = (Path(__file__).resolve().parent.parent
            / "wordalign" / "gui" / "app.html")


class TestJsonBodies(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        class QuietHandler(PipelineAPIHandler):
            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def post(self, path, body: bytes):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("POST", path, body=body,
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read() or b"{}")
        finally:
            conn.close()

    def test_non_object_json_is_treated_as_empty(self):
        # Regression: a JSON array reached body.get(...) and the handler
        # died with AttributeError, dropping the connection with no reply.
        status, data = self.post("/api/run/cancel", b"[1, 2]")
        self.assertEqual(status, 404)
        self.assertEqual(data["error"], "Job not found")

    def test_run_without_audio_is_refused(self):
        # Regression: {} (or a non-object body) queued a job with no audio.
        status, data = self.post("/api/run", b"{}")
        self.assertNotEqual(status, 200)
        self.assertEqual(data["error"], "No audio file given.")
        self.assertNotIn("job_id", data)

    def test_oversized_body_is_refused_unread(self):
        # Regression: the whole body was read into memory, however large.
        with socket.create_connection(("127.0.0.1", self.port),
                                      timeout=5) as sock:
            sock.sendall(
                b"POST /api/run/cancel HTTP/1.1\r\n"
                b"Host: 127.0.0.1:%d\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: 2000000000\r\n\r\n" % self.port)
            reply = b""
            while b"\r\n\r\n" not in reply:
                chunk = sock.recv(4096)      # times out if the server waits
                if not chunk:
                    break
                reply += chunk
        self.assertRegex(reply[:12], rb"HTTP/1\.[01] 413")

    def test_bad_content_length_does_not_crash(self):
        with socket.create_connection(("127.0.0.1", self.port),
                                      timeout=5) as sock:
            sock.sendall(
                b"POST /api/run/cancel HTTP/1.1\r\n"
                b"Host: 127.0.0.1:%d\r\n"
                b"Content-Length: abc\r\n\r\n" % self.port)
            reply = sock.recv(4096)
        self.assertRegex(reply[:12], rb"HTTP/1\.[01] 404")


class TestAppHtmlUsesText(unittest.TestCase):
    """Error messages and file paths come from the server and can hold
    '<' or '&' (a file named "a<b>.mp4" on Linux); they must be inserted
    as text, never parsed as markup."""

    @classmethod
    def setUpClass(cls):
        cls.html = APP_HTML.read_text(encoding="utf-8")

    def test_log_is_never_appended_as_html(self):
        self.assertNotRegex(self.html, r"innerHTML\s*\+=")

    def test_output_paths_are_text(self):
        body = re.search(r"function showOutput\(files\) \{(.*?)\n\}",
                         self.html, re.S).group(1)
        self.assertNotIn("innerHTML", body)
        self.assertIn("textContent = f", body)

    def test_engine_list_escapes_server_values(self):
        body = re.search(r"async function loadEngines\(\) \{(.*?)\n\}",
                         self.html, re.S).group(1)
        for value in re.findall(r"\$\{([^}]*)\}", body):
            if "e.name" in value or "e.id" in value:
                self.assertTrue(value.startswith("esc("), value)


if __name__ == "__main__":
    unittest.main()
