"""--max-lines: cues may use up to N lines, and never more."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import srt

from wordalign.core.config import SegmentationConfig
from wordalign.segment import (balance_block_enhanced,
                               check_if_text_can_be_split_prettily, get_config,
                               set_config, validate_srt_output)

LONG = ("the committee met on tuesday to review the proposal and agreed "
        "to publish the final report next month")   # 102 chars


class _Layout(unittest.TestCase):
    def setUp(self):
        previous = get_config()
        self.addCleanup(lambda: set_config(SegmentationConfig(**previous)))

    def layout(self, cpl, lines):
        set_config(SegmentationConfig(max_cpl=cpl, max_lines=lines))


class TestBalancer(_Layout):

    def test_three_lines_used_when_two_cannot_hold_the_cue(self):
        # Regression: the balancer only ever made two lines, so with
        # --max-lines 3 a 102-char cue at 42 CPL stayed one long line.
        self.layout(42, 3)
        out = balance_block_enhanced(LONG)
        lines = out.split("\n")
        self.assertEqual(len(lines), 3, out)
        self.assertTrue(all(len(line) <= 42 for line in lines), out)
        self.assertEqual(" ".join(lines), LONG)

    def test_two_lines_still_preferred_when_they_fit(self):
        self.layout(42, 3)
        text = "we should leave now before the rain gets any heavier"
        self.assertEqual(len(balance_block_enhanced(text).split("\n")), 2)

    def test_lines_do_not_end_on_an_article(self):
        self.layout(42, 3)
        for line in balance_block_enhanced(LONG).split("\n")[:-1]:
            self.assertNotIn(line.split()[-1], {"the", "a", "an", "to"})

    def test_one_line_layout_never_adds_a_second_line(self):
        # Regression: with --max-lines 1 the balancer still split cues.
        self.layout(42, 1)
        text = "we should leave now before the rain gets any heavier"
        self.assertNotIn("\n", balance_block_enhanced(text))

    def test_five_lines_found_for_a_cue_that_needs_them(self):
        # Regression: the 3+-line search enumerated cut points in order and
        # stopped after 20,000 of them, all with a one- to three-word first
        # line, so a 195-char cue at 42 CPL / 5 lines was left on one line.
        self.layout(42, 5)
        text = (LONG + " after the board has signed off on the budget "
                "and the auditors have checked every figure twice")
        lines = balance_block_enhanced(text).split("\n")
        self.assertEqual(len(lines), 5, lines)
        self.assertTrue(all(len(line) <= 42 for line in lines), lines)
        self.assertEqual(" ".join(lines), text)

    def test_many_tiny_words_finish_quickly(self):
        import time
        self.layout(42, 8)
        started = time.monotonic()
        lines = balance_block_enhanced(" ".join(["a"] * 150)).split("\n")
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertTrue(all(len(line) <= 42 for line in lines))

    def test_pretty_split_check_honours_max_lines(self):
        self.layout(42, 2)
        self.assertFalse(check_if_text_can_be_split_prettily(LONG))
        self.layout(42, 3)
        self.assertTrue(check_if_text_can_be_split_prettily(LONG))
        self.layout(42, 1)
        self.assertFalse(check_if_text_can_be_split_prettily(
            "we should leave now before the rain gets any heavier"))


class TestValidator(_Layout):

    def test_too_many_lines_are_reported(self):
        self.layout(42, 2)
        entries = [srt.Subtitle(1, timedelta(0), timedelta(seconds=2),
                                "one\ntwo\nthree")]
        with contextlib.redirect_stdout(io.StringIO()):
            issues = validate_srt_output(entries, "test")
        self.assertEqual(issues["lines_high"], 1)


class TestLayoutFlagValidation(unittest.TestCase):

    def test_impossible_layouts_are_rejected(self):
        # Regression: --max-lines 0, --cpl 0 and negative durations were
        # accepted and produced degenerate output without any message.
        from wordalign.cli_v2 import _parse_args
        for args in (["--max-lines", "0"], ["--max-lines", "-1"],
                     ["--cpl", "0"], ["--max-duration-ms", "0"],
                     ["--min-cue-ms", "-5"],
                     ["--min-cue-ms", "3000", "--max-duration-ms", "2000"]):
            with self.subTest(args=args), \
                    contextlib.redirect_stderr(io.StringIO()) as err, \
                    self.assertRaises(SystemExit) as caught:
                _parse_args(["a.wav", *args])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn(args[0], err.getvalue())

    def test_sensible_layouts_are_accepted(self):
        from wordalign.cli_v2 import _parse_args
        cfg = _parse_args(["a.wav", "--max-lines", "1", "--cpl", "16",
                           "--min-cue-ms", "0", "--max-duration-ms", "5000"])
        self.assertEqual((cfg.max_lines, cfg.max_cpl), (1, 16))


class TestPipelineMaxLines(unittest.TestCase):

    def _segments(self, max_lines):
        from wordalign.cli_v2 import _parse_args
        from wordalign.core.pipeline import PipelineRunner
        with tempfile.TemporaryDirectory() as d, patch.dict(
                os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            audio = Path(d) / "audio.wav"
            audio.write_bytes(b"x")
            transcript = Path(d) / "t.txt"
            transcript.write_text(LONG + ".\n", encoding="utf-8")
            cfg = _parse_args([str(audio), "-t", str(transcript), "-o", d,
                               "--max-lines", str(max_lines), "--no-vosk",
                               "--no-qwen", "--no-whisperx", "--no-mfa"])
            with contextlib.redirect_stdout(io.StringIO()):
                return PipelineRunner(cfg).run().segments

    def test_cli_max_lines_three_reaches_the_output(self):
        seg = self._segments(3)[0]
        self.assertEqual(len(seg.text.split("\n")), 3, seg.text)


if __name__ == "__main__":
    unittest.main()
