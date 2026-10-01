"""GUI API: other websites cannot drive the local server (CSRF / DNS rebinding)."""
import http.client
import os
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.gui.server import PipelineAPIHandler


class TestForeignRequestsAreRefused(unittest.TestCase):

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

    def setUp(self):
        self.cleared = []
        patcher = patch("wordalign.core.cache.StageCache.clear",
                        lambda cache: self.cleared.append(1) or 0)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _request(self, method, path, headers=None, body=b"{}"):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            # skip_host: the test controls the Host header itself.
            conn.putrequest(method, path, skip_host=True,
                            skip_accept_encoding=True)
            headers = dict(headers or {})
            headers.setdefault("Host", f"127.0.0.1:{self.port}")
            if method == "POST":
                headers.setdefault("Content-Type", "text/plain")
                headers["Content-Length"] = str(len(body))
            for k, v in headers.items():
                conn.putheader(k, v)
            conn.endheaders(body if method == "POST" else None)
            resp = conn.getresponse()
            resp.read()
            return resp.status
        finally:
            conn.close()

    # --- CSRF: a "simple" cross-site POST (no preflight) must not act ---

    def test_cross_origin_post_is_refused(self):
        status = self._request("POST", "/api/cache/clear",
                               {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.cleared, [])

    def test_null_origin_post_is_refused(self):
        self.assertEqual(self._request("POST", "/api/cache/clear",
                                       {"Origin": "null"}), 403)
        self.assertEqual(self.cleared, [])

    def test_cross_site_fetch_metadata_is_refused(self):
        self.assertEqual(self._request("POST", "/api/cache/clear",
                                       {"Sec-Fetch-Site": "cross-site"}), 403)
        self.assertEqual(self.cleared, [])

    def test_same_port_other_scheme_or_port_is_refused(self):
        for origin in (f"https://127.0.0.1:{self.port}",
                       f"http://127.0.0.1:{self.port + 1}"):
            self.assertEqual(self._request("POST", "/api/cache/clear",
                                           {"Origin": origin}), 403, origin)

    def test_gui_origin_post_is_allowed(self):
        for name in ("127.0.0.1", "localhost"):
            status = self._request("POST", "/api/cache/clear", {
                "Host": f"{name}:{self.port}",
                "Origin": f"http://{name}:{self.port}",
                "Sec-Fetch-Site": "same-origin"})
            self.assertEqual(status, 200, name)
        self.assertEqual(len(self.cleared), 2)

    def test_scripts_without_origin_are_allowed(self):
        self.assertEqual(self._request("POST", "/api/cache/clear"), 200)

    # --- DNS rebinding: a foreign Host is refused even for reads ---

    def test_foreign_host_header_is_refused(self):
        self.assertEqual(self._request("GET", "/api/health",
                                       {"Host": f"evil.example:{self.port}"}),
                         403)
        self.assertEqual(self._request("POST", "/api/cache/clear",
                                       {"Host": f"evil.example:{self.port}"}),
                         403)
        self.assertEqual(self.cleared, [])

    def test_loopback_hosts_are_served(self):
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}",
                     f"[::1]:{self.port}"):
            self.assertEqual(self._request("GET", "/api/health",
                                           {"Host": host}), 200, host)

    def test_extra_hosts_can_be_allowed_explicitly(self):
        with patch.dict(os.environ, {"WORDALIGN_ALLOWED_HOSTS": "studio-pc"}):
            self.assertEqual(self._request("GET", "/api/health",
                                           {"Host": f"studio-pc:{self.port}"}),
                             200)


if __name__ == "__main__":
    unittest.main()
