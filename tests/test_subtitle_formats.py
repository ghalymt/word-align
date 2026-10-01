"""WebVTT and ASS export."""
import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import srt

from wordalign.subtitle_formats import (ass_timestamp, compose_ass,
                                        compose_vtt, vtt_timestamp)


def cue(i, start, end, text, speaker=None):
    c = srt.Subtitle(i, timedelta(seconds=start), timedelta(seconds=end), text)
    if speaker:
        c.speaker = speaker
    return c


class TestWebVtt(unittest.TestCase):

    def test_document_structure(self):
        out = compose_vtt([cue(1, 1.5, 3.25, "Hello there,\nhow are you?"),
                           cue(2, 3661.0, 3662.007, "Fine.")])
        self.assertEqual(out, (
            "WEBVTT\n\n"
            "1\n00:00:01.500 --> 00:00:03.250\nHello there,\nhow are you?\n\n"
            "2\n01:01:01.000 --> 01:01:02.007\nFine.\n"))

    def test_markup_characters_are_escaped(self):
        out = compose_vtt([cue(1, 0, 1, "Q&A <live> & more")])
        self.assertIn("Q&amp;A &lt;live&gt; &amp; more", out)

    def test_blank_line_inside_a_cue_does_not_end_it(self):
        # Regression: a blank line ends a WebVTT cue, so the text after it
        # was dropped by players (or read as the next cue's identifier).
        out = compose_vtt([cue(1, 0, 1, "first line\n\nsecond line")])
        self.assertIn("-->", out)
        self.assertIn("first line\nsecond line\n", out)
        self.assertNotIn("\n\nsecond", out)

    def test_speaker_becomes_a_voice_span(self):
        out = compose_vtt([cue(1, 0, 1, "Hi\nthere", speaker="Speaker 1")])
        self.assertIn("<v Speaker 1>Hi\nthere", out)

    def test_timestamp(self):
        self.assertEqual(vtt_timestamp(timedelta(seconds=59.9996)),
                         "00:01:00.000")


class TestAss(unittest.TestCase):

    def test_script_sections_and_events(self):
        out = compose_ass([cue(1, 1.5, 3.256, "Hello there,\nhow are you?")])
        for section in ("[Script Info]", "[V4+ Styles]", "[Events]"):
            self.assertIn(section, out)
        self.assertIn("ScriptType: v4.00+", out)
        self.assertIn("WrapStyle: 2", out)
        self.assertIn("Dialogue: 0,0:00:01.50,0:00:03.25,Default,,0,0,0,,"
                      "Hello there,\\Nhow are you?", out)

    def test_braces_are_escaped_and_speaker_is_the_name(self):
        out = compose_ass([cue(1, 0, 1, "use {braces}", speaker="Ann, host")])
        self.assertIn(",Default,Ann  host,0,0,0,,use \\{braces\\}", out)

    def test_blank_lines_do_not_become_empty_ass_lines(self):
        out = compose_ass([cue(1, 0, 1, "first line\n\nsecond line")])
        self.assertIn(",,first line\\Nsecond line", out)

    def test_centiseconds_truncate(self):
        self.assertEqual(ass_timestamp(timedelta(seconds=3661.999)),
                         "1:01:01.99")


class TestPipelineExport(unittest.TestCase):

    def _run(self, *extra):
        from wordalign.cli_v2 import _parse_args
        from wordalign.core.pipeline import PipelineRunner
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, True))
        audio = Path(d) / "audio.wav"
        audio.write_bytes(b"x")
        transcript = Path(d) / "t.txt"
        transcript.write_text("Hello there & welcome.\nGood to see you.\n",
                              encoding="utf-8")
        with patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            cfg = _parse_args([str(audio), "-t", str(transcript), "-o", d,
                               "--no-vosk", "--no-qwen", "--no-whisperx",
                               "--no-mfa", *extra])
            with contextlib.redirect_stdout(io.StringIO()):
                return PipelineRunner(cfg).run()

    def test_all_three_formats_are_written(self):
        result = self._run("--formats", "srt,vtt,ass")
        self.assertTrue(result.sentence_level_srt_path)
        vtt = Path(result.sentence_level_vtt_path).read_text(encoding="utf-8")
        ass = Path(result.sentence_level_ass_path).read_text(encoding="utf-8-sig")
        self.assertTrue(vtt.startswith("WEBVTT\n"))
        self.assertIn("Hello there &amp; welcome.", vtt)
        self.assertEqual(len(re.findall(r"^Dialogue:", ass, re.M)),
                         len(result.segments))
        for path in (result.sentence_level_vtt_path,
                     result.sentence_level_ass_path):
            self.assertIn(path, result.output_files)

    def test_default_is_srt_only(self):
        result = self._run()
        self.assertTrue(result.sentence_level_srt_path)
        self.assertIsNone(result.sentence_level_vtt_path)
        self.assertIsNone(result.sentence_level_ass_path)

    def test_formats_override_a_profile(self):
        result = self._run("--profile", "fast", "--formats", "vtt")
        self.assertIsNone(result.sentence_level_srt_path)
        self.assertTrue(result.sentence_level_vtt_path)

    def test_unknown_format_is_rejected(self):
        from wordalign.cli_v2 import _parse_args
        with self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()):
            _parse_args(["a.wav", "--formats", "srt,sbv"])


class TestGuiFormats(unittest.TestCase):

    def test_server_passes_formats_to_the_run(self):
        from wordalign.core.types import PipelineResult
        from wordalign.gui.server import PipelineAPIHandler
        seen = {}

        def fake_run(runner):
            seen["runner"] = runner
            return PipelineResult(cancelled=True)

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
                "XDG_DATA_HOME": d, "LOCALAPPDATA": d}), \
                patch("wordalign.core.pipeline.PipelineRunner.run", fake_run):
            from wordalign.core.database import JobRecord, ProjectRecord, ProjectStore
            store = ProjectStore()
            store.create_project(ProjectRecord(id="p", name="p", audio_path="a"))
            store.create_job(JobRecord(id="j", project_id="p"))
            store.close()
            PipelineAPIHandler._run_pipeline_thread(None, "j", {
                "audio_path": "a.wav", "profile": "balanced",
                "formats": ["srt", "vtt", "bogus"]})
        export = seen["runner"].profile.export
        self.assertTrue(export.sentence_srt)
        self.assertTrue(export.vtt)
        self.assertFalse(export.ass)

    def test_gui_offers_the_formats(self):
        html = (Path(__file__).resolve().parent.parent / "wordalign" / "gui" /
                "app.html").read_text(encoding="utf-8")
        self.assertIn('id="format-vtt"', html)
        self.assertIn('id="format-ass"', html)


if __name__ == "__main__":
    unittest.main()
