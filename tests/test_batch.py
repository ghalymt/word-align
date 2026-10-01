"""Batch folder processing (wordalign --batch FOLDER)."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.batch import find_media, pair_transcript
from wordalign.cli_v2 import main

ENGINES_OFF = ["--no-vosk", "--no-qwen", "--no-whisperx", "--no-mfa"]
SRT = "1\n00:00:00,000 --> 00:00:02,000\nGood morning.\n"


class TestBatch(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.src = self.root / "media"
        self.out = self.root / "out"
        (self.src / "sub").mkdir(parents=True)
        for name in ("a.wav", "b.mp3", "c.mp4", "sub/d.wav"):
            (self.src / name).write_bytes(b"not-real-audio")
        (self.src / "a.txt").write_text("Hello there.\nHow are you?\n",
                                        encoding="utf-8")
        (self.src / "b.srt").write_text(SRT, encoding="utf-8")
        (self.src / "sub" / "d.txt").write_text("Deep file.\n", encoding="utf-8")
        (self.src / "notes.txt").write_text("not a transcript of anything",
                                            encoding="utf-8")
        env = patch.dict(os.environ,
                         {"WORDALIGN_CACHE": str(self.root / "cache")})
        env.start()
        self.addCleanup(env.stop)

    def run_cli(self, *extra):
        argv = ["--batch", str(self.src), "-o", str(self.out), *ENGINES_OFF,
                *extra]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(argv)
        return code, out.getvalue()

    def report(self):
        return json.loads((self.out / "wordalign_batch_report.json")
                          .read_text(encoding="utf-8"))

    def test_discovery_and_pairing(self):
        names = [p.name for p in find_media(self.src)]
        self.assertEqual(names, ["a.wav", "b.mp3", "c.mp4"])
        self.assertEqual(len(find_media(self.src, recursive=True)), 4)
        self.assertEqual(pair_transcript(self.src / "a.wav").name, "a.txt")
        self.assertEqual(pair_transcript(self.src / "b.mp3").name, "b.srt")
        self.assertIsNone(pair_transcript(self.src / "c.mp4"))

    def test_batch_processes_every_file_and_keeps_going(self):
        code, out = self.run_cli()
        # c.mp4 has no transcript and every engine is off -> it fails, but
        # the others still run and the exit code reports the failure.
        self.assertEqual(code, 1)
        self.assertTrue((self.out / "a_sentence_level.srt").exists())
        self.assertTrue((self.out / "b_sentence_level.srt").exists())
        self.assertIn("Good morning.",
                      (self.out / "b_sentence_level.srt").read_text("utf-8"))
        report = self.report()
        self.assertEqual(report["counts"],
                         {"ok": 2, "skipped": 0, "failed": 1, "cancelled": 0})
        failed = [e for e in report["files"] if e["status"] == "failed"]
        self.assertEqual(Path(failed[0]["media"]).name, "c.mp4")
        self.assertIn("BATCH COMPLETE: 2 ok, 0 skipped, 1 failed", out)
        self.assertFalse((self.out / "d_sentence_level.srt").exists())

    def test_recursive_and_skip_existing(self):
        self.run_cli("--recursive")
        self.assertTrue((self.out / "sub" / "d_sentence_level.srt").exists())
        code, _ = self.run_cli("--recursive", "--skip-existing")
        statuses = {Path(e["media"]).name: e["status"]
                    for e in self.report()["files"]}
        self.assertEqual(statuses, {"a.wav": "skipped", "b.mp3": "skipped",
                                    "c.mp4": "failed", "d.wav": "skipped"})

    def test_recursive_output_mirrors_subfolders(self):
        # Regression: with -o and --recursive every file wrote straight into
        # the output folder, so sub/a.wav overwrote a.wav's subtitles, and
        # --skip-existing then skipped sub/a.wav as already done.
        (self.src / "sub" / "a.wav").write_bytes(b"not-real-audio")
        (self.src / "sub" / "a.txt").write_text("Another file.\n",
                                                encoding="utf-8")
        self.run_cli("--recursive")
        top = (self.out / "a_sentence_level.srt").read_text("utf-8")
        deep = (self.out / "sub" / "a_sentence_level.srt").read_text("utf-8")
        self.assertIn("Hello there.", top)
        self.assertIn("Another file.", deep)
        self.run_cli("--recursive", "--skip-existing")
        statuses = {os.path.relpath(e["media"], self.src): e["status"]
                    for e in self.report()["files"]}
        self.assertEqual(statuses[os.path.join("sub", "a.wav")], "skipped")

    def test_same_name_in_one_folder_is_not_overwritten(self):
        # a.wav and a.mp4 would both write a_sentence_level.srt.
        (self.src / "a.mp4").write_bytes(b"not-real-audio")
        code, out = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("Hello there.", (self.out / "a_sentence_level.srt")
                      .read_text("utf-8"))
        entry = next(e for e in self.report()["files"]
                     if Path(e["media"]).name == "a.wav")
        self.assertEqual(entry["status"], "failed")
        self.assertIn("same output name as a.mp4", entry["error"])

    def test_flags_apply_to_every_file(self):
        code, _ = self.run_cli("--formats", "srt,vtt")
        self.assertTrue((self.out / "a_sentence_level.vtt").exists())
        self.assertTrue((self.out / "b_sentence_level.vtt").exists())

    def test_explicit_transcript_is_rejected(self):
        code, out = self.run_cli("-t", str(self.src / "a.txt"))
        self.assertEqual(code, 2)
        self.assertIn("paired with media files by name", out)

    def test_missing_folder(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            code = main(["--batch", str(self.root / "nope")])
        self.assertEqual(code, 2)
        self.assertIn("not a folder", out.getvalue())


if __name__ == "__main__":
    unittest.main()
