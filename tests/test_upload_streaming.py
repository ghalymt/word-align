"""Streaming multipart uploads: correct bytes, bounded memory, bad input."""
import http.client
import io
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

from wordalign.gui import server as gui_server
from wordalign.gui.server import PipelineAPIHandler, _stream_multipart

BOUNDARY = "----wordalignStreamTest4f9e"


def body(parts):
    out = b""
    for name, filename, data in parts:
        disp = f'form-data; name="{name}"'
        if filename is not None:
            disp += f'; filename="{filename}"'
        out += (f"--{BOUNDARY}\r\nContent-Disposition: {disp}\r\n"
                "Content-Type: application/octet-stream\r\n\r\n").encode()
        out += data + b"\r\n"
    return out + f"--{BOUNDARY}--\r\n".encode()


class RecordingReader(io.BytesIO):
    """BytesIO that remembers the largest single read() request."""
    largest = 0

    def read(self, n=-1):
        self.largest = max(self.largest, n)
        return super().read(n)


class TestStreamMultipart(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def parse(self, raw, length=None):
        reader = RecordingReader(raw)
        saved = _stream_multipart(reader, len(raw) if length is None else length,
                                  BOUNDARY.encode(), self.dir)
        return saved, reader

    def test_bytes_survive_tiny_chunks(self):
        # Content full of boundary-like fragments, read 7 bytes at a time so
        # every partial separator straddles a chunk edge.
        tricky = (b"\r\n--" + BOUNDARY[:-1].encode() + b"\r\n-\r\n--\r\n" * 5
                  + bytes(range(256)) + b"\r\n")
        raw = body([("a", "one.bin", tricky), ("b", "two.txt", b"hello\n")])
        with patch.object(gui_server, "_UPLOAD_CHUNK", 7):
            saved, reader = self.parse(raw)
        self.assertEqual([Path(p).read_bytes() for p in saved],
                         [tricky, b"hello\n"])
        self.assertLessEqual(reader.largest, 7)

    def test_large_upload_is_streamed_in_bounded_reads(self):
        payload = os.urandom(3 * 1024 * 1024)
        raw = body([("audio", "big.wav", payload)])
        with patch.object(gui_server, "_UPLOAD_CHUNK", 64 * 1024):
            saved, reader = self.parse(raw)
        self.assertEqual(Path(saved[0]).read_bytes(), payload)
        self.assertLessEqual(reader.largest, 64 * 1024)

    def test_plain_form_fields_are_not_saved_as_files(self):
        saved, _ = self.parse(body([("note", None, b"just text"),
                                    ("f", "x.txt", b"data")]))
        self.assertEqual([Path(p).name for p in saved], ["x.txt"])

    def test_client_paths_and_dot_names_are_neutralised(self):
        saved, _ = self.parse(body([("f", "C:\\fakepath\\clip.wav", b"1"),
                                    ("g", "..", b"2")]))
        self.assertEqual(sorted(Path(p).name for p in saved),
                         ["clip.wav", "uploaded_file"])
        for p in saved:
            self.assertEqual(Path(p).parent, Path(self.dir))

    def test_semicolons_in_a_file_name_are_kept(self):
        # Regression: the header was split on ';' before the quoted file
        # name was read, so "Talk; part 1.srt" was saved as "Talk" -- and
        # without its extension an .srt transcript was read as plain text.
        saved, _ = self.parse(body([("t", "Talk; part 1.srt", b"1"),
                                    ("a", "محاضرة; ١.mp4", b"2")]))
        self.assertEqual([Path(p).name for p in saved],
                         ["Talk; part 1.srt", "محاضرة; ١.mp4"])

    def test_characters_windows_rejects_are_replaced(self):
        saved, _ = self.parse(body([("f", 'a\\"b|c?.wav', b"1")]))
        self.assertEqual(Path(saved[0]).name, "a_b_c_.wav")

    def test_truncated_upload_is_rejected(self):
        raw = body([("f", "x.bin", b"x" * 1000)])
        with self.assertRaises(ValueError):
            self.parse(raw[:500], length=500)


class TestUploadEndpoint(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        class Quiet(PipelineAPIHandler):
            def log_message(self, *args):
                pass
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def post(self, raw):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1],
                                          timeout=10)
        try:
            conn.request("POST", "/api/upload", body=raw, headers={
                "Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
            resp = conn.getresponse()
            return resp.status, json.loads(resp.read())
        finally:
            conn.close()

    def test_malformed_body_gets_400(self):
        status, data = self.post(b"no boundary in here at all")
        self.assertEqual(status, 400)
        self.assertIn("Malformed upload", data["error"])

    def test_upload_round_trip(self):
        payload = os.urandom(200_000)
        status, data = self.post(body([("audio", "a.wav", payload)]))
        self.assertEqual(status, 200)
        self.assertEqual(Path(data["files"][0]).read_bytes(), payload)
        upload_dir = gui_server._UPLOAD_DIRS.pop(data["upload_id"], None)
        shutil.rmtree(upload_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
