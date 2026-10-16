"""Timing and tag style learned from a reviewer's finished subtitles."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign import segment as S
from wordalign.core.config import SegmentationConfig
from wordalign.qa.triage import triage_segments
from wordalign.qa.triage_calibration import label_cues
from wordalign.utils import time_to_ms


def seg(a, b, text, words=None, tag=False):
    out = {"start": S.ms_to_time(a), "end": S.ms_to_time(b), "text": text}
    if words is not None:
        out["words"] = words
    if tag:
        out.update(tag=True, words=[])
    return out


def w(word, a, b):
    return {"word": word, "start": a / 1000, "end": b / 1000}


class Base(unittest.TestCase):
    def setUp(self):
        self.previous = S.get_config()
        S.set_config(SegmentationConfig(max_cpl=37, max_lines=2, max_duration_ms=7000,
                                        min_cue_ms=1000))
        self.lang = S.set_language("en")

    def tearDown(self):
        S.set_config(SegmentationConfig(**self.previous))
        S.set_language(self.lang)


class TestTagsInsideSentences(Base):
    def test_tag_in_the_middle_of_a_sentence_stays_inline(self):
        segs = [seg(0, 900, "or not a wave, but in", []), seg(900, 900, "[chuckles]", tag=True),
                seg(1000, 2000, "a tornado.", [])]
        out = S.inline_mid_sentence_tags(segs)
        self.assertEqual([s["text"] for s in out], ["or not a wave, but in [chuckles]", "a tornado."])

    def test_tag_between_sentences_stays_separate(self):
        for before, after in (("Okay.", "Now we start."), ("That was your decision-", "-and mine."),
                              ("15 [?]", "15 is the nadir.")):
            segs = [seg(0, 900, before, []), seg(900, 900, "[laughter]", tag=True),
                    seg(1000, 2000, after, [])]
            self.assertEqual(len(S.inline_mid_sentence_tags(segs)), 3, before)

    def test_tag_cue_fills_its_pause_even_when_short(self):
        segs = [seg(0, 1000, "Good one.", [w("Good", 0, 400), w("one.", 400, 1000)]),
                seg(1000, 1000, "[laughter]", tag=True),
                seg(1700, 2500, "Next part.", [w("Next", 1700, 2000), w("part.", 2000, 2500)])]
        out = S.place_tag_cues(segs, min_ms=1000)
        tag = [s for s in out if s.get("tag")]
        self.assertEqual(len(tag), 1)
        self.assertEqual((time_to_ms(tag[0]["start"]), time_to_ms(tag[0]["end"])), (1001, 1699))


class TestTiming(Base):
    def test_chaining_twice_changes_nothing(self):
        segs = [seg(0, 1000, "one", [w("one", 0, 1000)]),
                seg(5000, 6000, "two", [w("two", 5000, 6000)])]
        once = S.chain_cues(segs)
        self.assertEqual(once, S.chain_cues(once))
        self.assertEqual(time_to_ms(once[0]["end"]), 1250)        # lead-out 250 ms

    def test_sentence_end_boundary_is_never_moved(self):
        a = seg(0, 1500, "We are done here.", [w("We", 0, 300), w("are", 300, 600),
                                                 w("done", 600, 1000), w("here.", 1000, 1500)])
        b = seg(1501, 1900, "So the next thing we will talk about today is long.",
                [w(x, 1501 + i * 40, 1540 + i * 40) for i, x in enumerate(
                    "So the next thing we will talk about today is long.".split())])
        self.assertIsNone(S._resplit_pair(a, b, mode="speed"))


class TestStartsOnWords(Base):
    def test_a_stray_quote_does_not_pull_the_start_early(self):
        piece = {"text": '" "Okay, fine."', "words": [
            {"word": '"', "start": 1.0, "end": 1.1},          # guessed time
            {"word": '"Okay,', "start": 2.2, "end": 2.5},
            {"word": 'fine."', "start": 2.5, "end": 2.9}]}
        S._retime(piece)
        self.assertEqual(time_to_ms(piece["start"]), 2200)


class TestReviewLabels(unittest.TestCase):
    def test_recutting_cues_is_not_a_fix(self):
        cues = [{"start": 0.0, "end": 2.0, "text": "one two three"},
                {"start": 2.0, "end": 4.0, "text": "four five"}]
        corrected = [{"start": 0.0, "end": 1.4, "text": "one two"},
                     {"start": 1.4, "end": 4.5, "text": "three four five"}]
        self.assertEqual(label_cues(cues, corrected), [0, 0])

    def test_changed_word_and_moved_start_are_fixes(self):
        cues = [{"start": 0.0, "end": 2.0, "text": "one two three"},
                {"start": 2.0, "end": 4.0, "text": "four five"}]
        corrected = [{"start": 0.0, "end": 2.0, "text": "one too three"},
                     {"start": 2.6, "end": 4.0, "text": "four five"}]
        self.assertEqual(label_cues(cues, corrected), [1, 1])

    def test_end_cut_before_the_words_is_a_fix(self):
        cues = [{"start": 0.0, "end": 3.0, "text": "one two",
                 "words": [{"end": 1.0}, {"end": 2.8}]}]
        self.assertEqual(label_cues(cues, [{"start": 0.0, "end": 2.0, "text": "one two"}]), [1])
        self.assertEqual(label_cues(cues, [{"start": 0.0, "end": 2.9, "text": "one two"}]), [0])


class TestTriageNearTags(unittest.TestCase):
    def test_cue_next_to_a_sound_tag_is_flagged(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "[applause]"},
                {"start": 1.0, "end": 3.0, "text": "I'll get started with a question"},
                {"start": 9.0, "end": 11.0, "text": "Another line, far away"},
                {"start": 11.0, "end": 12.0, "text": "Yes, [?] question back there."}]
        cues = triage_segments(segs, [], "reference")
        self.assertIn("beside_tag", cues[1].features)
        self.assertNotIn("beside_tag", cues[0].features)           # the tag cue itself
        self.assertNotIn("beside_tag", cues[2].features)
        self.assertNotIn("beside_tag", cues[3].features)           # [?] is not a sound


if __name__ == "__main__":
    unittest.main()
