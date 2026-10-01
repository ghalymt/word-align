"""Speaker labels: diarization turns -> words -> cues -> SRT/VTT/ASS."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.config import SegmentationConfig
from wordalign.diarize import assign_speakers, speaker_names
from wordalign.document import export_transcript
from wordalign.segment import (get_config, parse_human_transcript_to_srt_segments,
                               run_iterative_merging, set_config)


def words_at(text, step=0.3, speakers=None):
    out = []
    for i, w in enumerate(text.split()):
        d = {"word": w, "start": i * step, "end": i * step + 0.25,
             "source": "Vosk"}
        if speakers:
            d["speaker"] = speakers[i]
        out.append(d)
    return out


class TestAssignSpeakers(unittest.TestCase):

    def test_overlap_nearest_and_unassigned(self):
        words = [{"start": 0.0, "end": 0.5}, {"start": 0.9, "end": 1.4},
                 {"start": 3.1, "end": 3.3}, {"start": 9.0, "end": 9.2}]
        turns = [(0.0, 1.0, "SPK_7"), (1.0, 3.0, "SPK_2")]
        self.assertEqual(assign_speakers(words, turns), 3)
        self.assertEqual(words[0]["speaker"], "Speaker 1")
        self.assertEqual(words[1]["speaker"], "Speaker 2")   # most overlap
        self.assertEqual(words[2]["speaker"], "Speaker 2")   # nearest, 0.1 s
        self.assertNotIn("speaker", words[3])                # 6 s away

    def test_names_follow_first_appearance(self):
        self.assertEqual(speaker_names([(5, 6, "B"), (0, 1, "A"), (2, 3, "B")]),
                         {"A": "Speaker 1", "B": "Speaker 2"})


class TestSegmentationKeepsSpeakersApart(unittest.TestCase):

    def setUp(self):
        previous = get_config()
        self.addCleanup(lambda: set_config(SegmentationConfig(**previous)))
        set_config(SegmentationConfig(max_cpl=42, max_lines=2))

    def _segments(self, text, speakers):
        aligned = words_at(text.replace("\n", " "), speakers=speakers)
        with contextlib.redirect_stdout(io.StringIO()):
            segs = parse_human_transcript_to_srt_segments(text, aligned)
            return run_iterative_merging(segs)

    def test_line_is_split_at_a_speaker_change(self):
        segs = self._segments("hello there how are you I am fine thanks",
                              ["A"] * 5 + ["B"] * 4)
        self.assertEqual([(s["text"].replace("\n", " "), s["speaker"]) for s in segs],
                         [("hello there how are you", "A"),
                          ("I am fine thanks", "B")])

    def test_cues_never_merge_across_speakers(self):
        text = "hello there how are you\nand you my friend"
        same = self._segments(text, ["A"] * 9)
        self.assertEqual(len(same), 1)            # control: would merge
        apart = self._segments(text, ["A"] * 5 + ["B"] * 4)
        self.assertEqual([s["speaker"] for s in apart], ["A", "B"])


class TestPipelineSpeakerLabels(unittest.TestCase):

    TURNS = [(0.0, 1.45, "SPEAKER_00"), (1.45, 30.0, "SPEAKER_01")]

    def _run(self, turns=None, unavailable=None, *flags):
        from wordalign.cli_v2 import _parse_args
        from wordalign.core.events import CollectSink, StageMessage
        from wordalign.core.pipeline import PipelineRunner
        from wordalign.diarize import DiarizationUnavailable

        def fake_turns(*args, **kwargs):
            if unavailable:
                raise DiarizationUnavailable(unavailable)
            return turns

        d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(d, True))
        audio = Path(d) / "talk.wav"
        audio.write_bytes(b"x")
        transcript = Path(d) / "t.txt"
        transcript.write_text("Hello there how are you\nI am fine thanks\n",
                              encoding="utf-8")
        cfg = _parse_args([str(audio), "-t", str(transcript), "-o", d,
                           "--no-vosk", "--no-qwen", "--no-whisperx", "--no-mfa",
                           "--diarize", "--formats", "srt,vtt,ass", *flags])
        sink = CollectSink()
        with patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}), \
                patch("wordalign.diarize.diarize_turns", fake_turns), \
                contextlib.redirect_stdout(io.StringIO()):
            result = PipelineRunner(cfg, sink=sink).run()
        messages = [e.message for e in sink.events if isinstance(e, StageMessage)]
        read = lambda p: Path(p).read_text(encoding="utf-8-sig")
        return result, messages, read

    def test_labels_reach_every_format(self):
        result, _, read = self._run(self.TURNS)
        self.assertEqual([s.speaker for s in result.segments],
                         ["Speaker 1", "Speaker 2"])
        srt_text = read(result.sentence_level_srt_path)
        self.assertIn("Speaker 1: Hello there how are you", srt_text)
        self.assertIn("Speaker 2: I am fine thanks", srt_text)
        vtt = read(result.sentence_level_vtt_path)
        self.assertIn("<v Speaker 2>I am fine thanks", vtt)
        ass = read(result.sentence_level_ass_path)
        self.assertIn(",Default,Speaker 1,0,0,0,,Hello there how are you", ass)

    def test_missing_pyannote_skips_with_a_reason(self):
        result, messages, read = self._run(
            unavailable="pyannote.audio is not installed")
        self.assertTrue(any("Speaker labels skipped" in m for m in messages))
        self.assertNotIn("Speaker", read(result.sentence_level_srt_path))
        self.assertTrue(all(s.speaker is None for s in result.segments))

    def test_cli_flags(self):
        from wordalign.cli_v2 import _parse_args
        cfg = _parse_args(["a.wav", "--diarize", "--speakers", "3"])
        self.assertTrue(cfg.diarize)
        self.assertEqual(cfg.num_speakers, 3)
        self.assertFalse(_parse_args(["a.wav"]).diarize)

    def test_speakers_alone_turns_diarization_on(self):
        # Regression: --speakers 2 without --diarize was silently ignored.
        from wordalign.cli_v2 import _parse_args
        cfg = _parse_args(["a.wav", "--speakers", "2"])
        self.assertTrue(cfg.diarize)
        self.assertEqual(cfg.num_speakers, 2)

    def test_speaker_count_must_be_positive(self):
        from wordalign.cli_v2 import _parse_args
        with contextlib.redirect_stderr(io.StringIO()) as err, \
                self.assertRaises(SystemExit):
            _parse_args(["a.wav", "--diarize", "--speakers", "0"])
        self.assertIn("--speakers must be at least 1", err.getvalue())


class TestTranscriptDocument(unittest.TestCase):

    def test_speaker_turns_start_paragraphs(self):
        words = words_at("Hi there. Hello back. How are things?",
                         speakers=["Speaker 1"] * 2 + ["Speaker 2"] * 2
                         + ["Speaker 1"] * 3)
        with tempfile.TemporaryDirectory() as d, \
                contextlib.redirect_stdout(io.StringIO()):
            paths = export_transcript(os.path.join(d, "t"), "t.wav", words,
                                      "txt", timestamps=False)
            text = Path(paths["transcript_txt_path"]).read_text(encoding="utf-8")
        self.assertEqual(text.strip().split("\n\n"), [
            "Speaker 1: Hi there.", "Speaker 2: Hello back.",
            "Speaker 1: How are things?"])


if __name__ == "__main__":
    unittest.main()
