"""GUI review queue: review file loading, edits, corrected SRT, audio range."""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.gui import review
from wordalign.qa.triage_calibration import label_cues

SRT = """1
00:00:01,000 --> 00:00:03,000
Speaker 1: we sale to
the island

2
00:00:04,000 --> 00:00:06,000
and stayed there

3
00:00:07,000 --> 00:00:08,000
for a week

"""


def make_job(folder: Path):
    cues = [
        {"index": 1, "start": 1.0, "end": 3.0, "text": "we sale to\nthe island",
         "speaker": "Speaker 1", "tier": "redo", "tier_label": "Redo", "risk": 0.9,
         "reasons": ["voters disagree on 1 word"], "features": {"disputed_any": 1.0},
         "words": [{"word": "sale", "text_conf": 0.4, "alternatives": {"sale": 0.4, "sailed": 0.6}}]},
        {"index": 2, "start": 4.0, "end": 6.0, "text": "and stayed there",
         "speaker": "Speaker 1", "tier": "quick_check", "tier_label": "Quick check",
         "risk": 0.4, "reasons": [], "features": {}, "words": []},
        {"index": 3, "start": 7.0, "end": 8.0, "text": "for a week", "speaker": "Speaker 1",
         "tier": "accept", "tier_label": "Accept", "risk": 0.1, "reasons": [],
         "features": {}, "words": []},
    ]
    (folder / "talk_review.json").write_text(json.dumps(
        {"version": 1, "summary": {}, "queue": [1, 2], "cues": cues}), encoding="utf-8")
    (folder / "talk_sentence_level.srt").write_text(SRT, encoding="utf-8")
    return cues


class TestReviewFiles(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.dir, True))
        self.cues = make_job(self.dir)

    def test_job_ids_are_checked(self):
        self.assertIsNone(review.job_output_dir("../x"))
        self.assertIsNone(review.job_output_dir("a/b"))
        self.assertIsNotNone(review.job_output_dir("ab12-cd"))

    def test_load_without_edits(self):
        data = review.load_review(self.dir)
        self.assertEqual(data["review_file"], "talk_review.json")
        self.assertEqual(data["edits"], {})
        self.assertIsNone(data["audio"])
        self.assertIsNone(review.load_review(self.dir / "missing"))

    def test_edits_make_a_corrected_copy(self):
        original = (self.dir / "talk_sentence_level.srt").read_text(encoding="utf-8")
        out = review.save_edits(self.dir, {
            "1": {"status": "edited", "text": "we sailed to\nthe island"},
            "2": {"status": "accepted"},
            "3": {"status": "edited", "text": "for a week", "start": 7.2, "end": 8.4}})
        self.assertEqual(out["corrected_file"], "talk_corrected.srt")
        self.assertEqual(out["changed"], 3)
        corrected = (self.dir / "talk_corrected.srt").read_text(encoding="utf-8")
        self.assertIn("Speaker 1: we sailed to\nthe island", corrected)   # prefix kept
        self.assertIn("00:00:07,200 --> 00:00:08,400", corrected)
        # The pipeline's own file is untouched.
        self.assertEqual((self.dir / "talk_sentence_level.srt").read_text(encoding="utf-8"),
                         original)
        log = (self.dir / "talk_review_log.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(log), 3)
        self.assertEqual(json.loads(log[0])["after"], "we sailed to\nthe island")
        # Re-saving the same state logs nothing; undo removes the edit.
        self.assertEqual(review.save_edits(self.dir, {"2": {"status": "accepted"}})["changed"], 0)
        out = review.save_edits(self.dir, {"1": None})
        self.assertNotIn("1", out["edits"])
        self.assertIn("we sale to", (self.dir / "talk_corrected.srt").read_text(encoding="utf-8"))
        self.assertEqual(review.load_review(self.dir)["corrected_file"], "talk_corrected.srt")

    def test_deleting_a_cue(self):
        review.save_edits(self.dir, {"2": {"status": "edited", "text": ""}})
        corrected = (self.dir / "talk_corrected.srt").read_text(encoding="utf-8")
        self.assertNotIn("stayed", corrected)
        self.assertTrue(corrected.startswith("1\n"))
        self.assertIn("\n2\n00:00:07,000", corrected)

    def test_invalid_edits_are_refused(self):
        for bad in ({"1": {"status": "nope"}}, {"1": {"status": "edited"}},
                    {"1": {"status": "edited", "text": "x", "start": 5, "end": 4}},
                    {"1": {"status": "edited", "text": "x", "start": "soon"}}):
            with self.assertRaises(ValueError):
                review.save_edits(self.dir, bad)
        self.assertFalse((self.dir / "talk_corrected.srt").exists())

    def test_calibration_labels_ignore_the_speaker_prefix(self):
        review.save_edits(self.dir, {"1": {"status": "edited", "text": "we sailed to\nthe island"},
                                     "2": {"status": "accepted"}})
        import srt
        corrected = [{"start": s.start.total_seconds(), "end": s.end.total_seconds(),
                      "text": s.content}
                     for s in srt.parse((self.dir / "talk_corrected.srt").read_text("utf-8"))]
        self.assertEqual(label_cues(self.cues, corrected), [1, 0, 0])


class TestReviewServer(unittest.TestCase):
    """Real HTTP server: review JSON, saving edits, ranged audio."""

    def setUp(self):
        from http.server import ThreadingHTTPServer
        from wordalign.gui.server import PipelineAPIHandler
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.home, True))
        p = patch("wordalign.gui.review.outputs_root", lambda: self.home)
        p.start()
        self.addCleanup(p.stop)
        job = self.home / "job1"
        job.mkdir()
        make_job(job)
        (job / "review_audio.wav").write_bytes(bytes(range(256)) * 4)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), PipelineAPIHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def _get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()

    def test_review_and_audio(self):
        status, _, body = self._get("/api/job/job1/review")
        data = json.loads(body)
        self.assertEqual(data["audio"], "review_audio.wav")
        self.assertEqual(len(data["review"]["cues"]), 3)
        status, headers, body = self._get("/api/job/job1/audio", {"Range": "bytes=10-19"})
        self.assertEqual(status, 206)
        self.assertEqual(headers["Content-Range"], "bytes 10-19/1024")
        self.assertEqual(body, bytes(range(10, 20)))
        status, _, body = self._get("/api/job/job1/audio")
        self.assertEqual((status, len(body)), (200, 1024))

    def test_save_edits_over_http(self):
        req = urllib.request.Request(
            self.base + "/api/job/job1/review", method="POST",
            data=json.dumps({"edits": {"2": {"status": "accepted"}}}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
        self.assertTrue(data["ok"])
        self.assertTrue((self.home / "job1" / "talk_corrected.srt").exists())

    def test_unknown_job(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self._get("/api/job/nojob/review")
        self.assertEqual(cm.exception.code, 404)


if __name__ == "__main__":
    unittest.main()
