"""GUI server: the local API sends no CORS headers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestGuiServerCors(unittest.TestCase):
    """The local API must not opt in to cross-origin access."""

    @classmethod
    def setUpClass(cls):
        import threading
        from http.server import ThreadingHTTPServer
        from wordalign.gui.server import PipelineAPIHandler

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

    def _request(self, method, path, headers=None):
        import http.client
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(method, path, headers=headers or {})
            resp = conn.getresponse()
            resp.read()
            return resp.status, {k.lower(): v for k, v in resp.getheaders()}
        finally:
            conn.close()

    def assertNoCors(self, headers):
        cors = [k for k in headers if k.startswith("access-control-")]
        self.assertEqual(cors, [], f"unexpected CORS headers: {cors}")

    def test_api_get_has_no_cors_headers(self):
        status, headers = self._request(
            "GET", "/api/health", {"Origin": "https://evil.example"})
        self.assertEqual(status, 200)
        self.assertNoCors(headers)

    def test_error_responses_have_no_cors_headers(self):
        status, headers = self._request(
            "GET", "/api/nope", {"Origin": "https://evil.example"})
        self.assertEqual(status, 404)
        self.assertNoCors(headers)
        status, headers = self._request(
            "POST", "/api/nope", {"Origin": "https://evil.example",
                                  "Content-Length": "0"})
        self.assertEqual(status, 404)
        self.assertNoCors(headers)

    def test_cross_origin_preflight_is_not_granted(self):
        status, headers = self._request("OPTIONS", "/api/run", {
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        self.assertGreaterEqual(status, 400)
        self.assertNoCors(headers)

    def test_gui_is_still_served_same_origin(self):
        status, headers = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["content-type"])
        self.assertNoCors(headers)



if __name__ == "__main__":
    unittest.main()
