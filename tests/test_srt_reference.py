"""Reference mode with an .srt transcript: input cues are preserved."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.config import PipelineConfig
from wordalign.core.config import SegmentationConfig
from wordalign.core.pipeline import PipelineRunner
from wordalign.segment import get_config, set_config
from wordalign.utils import time_to_ms

SRT = """1
00:00:00,000 --> 00:00:02,000
[music]

2
00:00:02,000 --> 00:00:03,000
Hello there.

3
00:00:03,000 --> 00:00:05,000
[laughs] How are you?

4
00:00:05,000 --> 00:00:07,000
[door slams]
Fine thanks, and I wanted to say this cue is long enough to need two lines.
"""


class TestSrtReferenceMode(unittest.TestCase):

    def _run(self, srt_text):
        with tempfile.TemporaryDirectory() as d, \
                patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            audio = Path(d) / "audio.wav"
            audio.write_bytes(b"not-real-audio")
            transcript = Path(d) / "transcript.srt"
            transcript.write_text(srt_text, encoding="utf-8")
            cfg = PipelineConfig(
                audio_path=str(audio), transcript_path=str(transcript),
                output_dir=d, use_vosk=False, use_qwen=False,
                use_whisperx=False, use_mfa=False)
            with contextlib.redirect_stdout(io.StringIO()):
                return PipelineRunner(cfg).run()

    def test_tags_do_not_shift_cue_boundaries(self):
        # Regression: "[music]" / "[laughs]" / "[door slams]" occupied word
        # slots, so every later cue started one or more words too early.
        result = self._run(SRT)
        texts = [s.text.replace("\n", " ") for s in result.segments]
        self.assertEqual(texts, [
            "Hello there.",
            "How are you?",
            "Fine thanks, and I wanted to say this cue is long enough to "
            "need two lines.",
        ])

    def test_preserved_cues_respect_cpl(self):
        result = self._run(SRT)
        for seg in result.segments:
            lines = seg.text.split("\n")
            self.assertLessEqual(len(lines), 2)
            for line in lines:
                self.assertLessEqual(len(line), 42, seg.text)

    def test_preserved_cues_do_not_overlap(self):
        runner = PipelineRunner(PipelineConfig("audio.wav"))
        aligned = [
            {"word": "Hello", "start": 0.0, "end": 0.5, "source": "WhisperX"},
            {"word": "there.", "start": 0.5, "end": 2.5, "source": "WhisperX"},
            {"word": "How", "start": 2.0, "end": 2.4, "source": "WhisperX"},
            {"word": "are", "start": 2.4, "end": 2.6, "source": "WhisperX"},
            {"word": "you?", "start": 2.6, "end": 3.0, "source": "WhisperX"},
        ]
        previous = get_config()
        try:
            seg_cfg = SegmentationConfig()
            set_config(seg_cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                segments = runner._segment_transcript(
                    "Hello there.\nHow are you?", aligned, [(0, 2), (2, 5)],
                    seg_cfg)
        finally:
            set_config(SegmentationConfig(**previous))
        self.assertEqual(len(segments), 2)
        self.assertLessEqual(time_to_ms(segments[0]["end"]),
                             time_to_ms(segments[1]["start"]))


if __name__ == "__main__":
    unittest.main()
