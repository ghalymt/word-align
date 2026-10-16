"""The final SRT: GoTranscript-style layout, every word and tag, tags clear of speech."""
import contextlib
import io
import sys
import unittest
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import srt

from wordalign.config import PipelineConfig
from wordalign.core.config import SegmentationConfig
from wordalign.core.pipeline import PipelineRunner
from wordalign.linebreak import BAD, break_cost, split_cost
from wordalign import segment as S
from wordalign.qa.subtitle_check import check_subtitles, hard_failures, transcript_stream
from wordalign.utils import time_to_ms

CFG = SegmentationConfig(max_cpl=37, max_lines=2, max_duration_ms=7000, min_cue_ms=1000)


class Layout(unittest.TestCase):
    def setUp(self):
        self.previous = S.get_config()
        S.set_config(CFG)
        self.lang = S.set_language("en")

    def tearDown(self):
        S.set_config(SegmentationConfig(**self.previous))
        S.set_language(self.lang)


def timed_words(text, start=0.0, step=0.3, pauses=None):
    """Aligned words for *text* (tags removed), *step* s apart; *pauses*
    maps a word index to extra silence before it."""
    words, t = [], start
    for i, w in enumerate(S.re.sub(r"\[.*?\]", " ", text).split()):
        t += (pauses or {}).get(i, 0.0)
        words.append({"word": w, "start": round(t, 3), "end": round(t + step - 0.05, 3),
                      "source": "Qwen3-FA"})
        t += step
    return words


def segment(text, words, events=None):
    runner = PipelineRunner(PipelineConfig("x.wav"))
    with contextlib.redirect_stdout(io.StringIO()):
        return runner._segment_transcript(text, words, [], CFG, language="en",
                                          tag_events=events)


def entries(segments):
    return [srt.Subtitle(i + 1, timedelta(milliseconds=time_to_ms(s["start"])),
                         timedelta(milliseconds=time_to_ms(s["end"])), s["text"])
            for i, s in enumerate(segments)]


class TestBreakCost(unittest.TestCase):
    def test_keeps_together(self):
        for left, right in [("Most people are tuned", "out to politics."),
                            ("in which we", "wake up"),
                            ("we have", "been doing this"),
                            ("I really enjoy", "it a lot"),
                            ("I talked to Henry", "Schein about it"),
                            ("we had 21", "patients"),
                            ("I'm meeting Dr.", "Smith"),
                            ("I'll get", "started"),
                            ("that you want to have a", "different relationship")]:
            self.assertGreaterEqual(split_cost(left, right), 10, f"{left} | {right}")

    def test_good_places(self):
        self.assertLess(split_cost("Good morning, everyone.", "Can you hear me?"), 0)
        self.assertLess(split_cost("we analyze it,", "and we fix it"), 0)
        self.assertLess(split_cost("tuned out", "to politics."), 0)
        self.assertLessEqual(split_cost("to have", "a different relationship"), 0)
        self.assertLessEqual(split_cost("back in 2019", "we had"), 0)

    def test_tag_edges_are_fine(self):
        self.assertLess(break_cost(["okay."], ["[laughter]"]), 0)
        self.assertLess(BAD, 30)


class TestLineBalance(Layout):
    def test_one_line_when_it_fits(self):
        self.assertEqual(S.balance_block_enhanced("Good morning, everyone."),
                         "Good morning, everyone.")

    def test_phrasal_verb_kept(self):
        self.assertEqual(S.balance_block_enhanced("Most people are tuned out to politics."),
                         "Most people are tuned out\nto politics.")

    def test_no_article_at_line_end(self):
        out = S.balance_block_enhanced("I'm going to come at it from a slightly different way.")
        self.assertEqual(out, "I'm going to come at it\nfrom a slightly different way.")

    def test_bad_existing_split_is_redone(self):
        out = S.balance_block_enhanced("I'm going to come at it from a\nslightly different way.")
        self.assertEqual(out.split("\n")[0], "I'm going to come at it")

    def test_edge_tag_gets_its_own_line(self):
        self.assertEqual(S.balance_block_enhanced("[applause] I'll get started with a question"),
                         "[applause]\nI'll get started with a question")
        self.assertEqual(S.balance_block_enhanced("Yes, [?] question back there. I have a son."),
                         "Yes, [?] question back there.\nI have a son.")

    def test_never_breaks_inside_a_tag(self):
        out = S.balance_block_enhanced("and then he said [background conversation] nothing")
        for line in out.split("\n"):
            self.assertEqual(line.count("["), line.count("]"), out)


class TestSplitting(Layout):
    def test_long_line_is_split_into_fitting_cues(self):
        text = ("So the thing that we have been talking about for the whole morning, "
                "which is anxiety, is something that every lawyer here knows well.")
        segs = segment(text, timed_words(text))
        for e in entries(segs):
            lines = e.content.split("\n")
            self.assertLessEqual(len(lines), 2)
            self.assertTrue(all(len(l) <= 37 for l in lines), e.content)
        self.assertEqual(transcript_stream(" ".join(s["text"] for s in segs)),
                         transcript_stream(text))

    def test_long_duration_is_split(self):
        text = "Okay so we wait here and then we go on."
        words = timed_words(text, pauses={5: 6.0})
        segs = segment(text, words)
        for e in entries(segs):
            self.assertLessEqual((e.end - e.start).total_seconds(), 7.0, e.content)


class TestTags(Layout):
    TEXT = ("[background conversation]\nGood morning, everyone.\n"
            "Can you all hear me okay?\n[laughter]\nOkay.\nWonderful.\n"
            "That was your decision-\n[chuckles]\n-and mine.\n")

    def _words(self):
        # A long pause before "Good", a short one at [laughter] and none at [chuckles].
        return timed_words(self.TEXT, start=3.0, pauses={4: 0.4})

    def test_every_word_and_tag_in_order_without_overlap(self):
        segs = segment(self.TEXT, self._words())
        report = check_subtitles(entries(segs), 37, 2, transcript=self.TEXT)
        self.assertEqual(report["transcript"]["missing"], 0, report)
        self.assertEqual(report["transcript"]["extra"], 0, report)
        self.assertEqual(hard_failures(report), {}, report)

    def test_tag_in_a_long_pause_gets_its_own_cue(self):
        segs = segment(self.TEXT, self._words())
        self.assertEqual(segs[0]["text"], "[background conversation]")
        self.assertTrue(segs[0].get("tag"))
        self.assertLessEqual(time_to_ms(segs[0]["end"]), time_to_ms(segs[1]["start"]) - 1)

    def test_tag_without_room_shares_a_neighbouring_cue(self):
        segs = segment(self.TEXT, self._words())
        holder = [s for s in segs if "[chuckles]" in s["text"]]
        self.assertEqual(len(holder), 1)
        self.assertFalse(holder[0].get("tag"))     # folded into a speech cue

    def test_sound_event_over_speech_is_pulled_into_the_pause(self):
        text = "Good morning, everyone.\n[laughter]\nNow let us begin today."
        words = timed_words(text, pauses={3: 2.5})
        good_end = words[2]["end"]
        now_start = words[3]["start"]
        # The detector put the laughter half over the speech on both sides.
        events = [{"start": good_end - 1.0, "end": now_start + 1.0, "tag": "Laughter"}]
        segs = segment(text, words, events)
        tag = [s for s in segs if s.get("tag")][0]
        self.assertGreater(time_to_ms(tag["start"]), int(good_end * 1000))
        self.assertLess(time_to_ms(tag["end"]), int(now_start * 1000))
        report = check_subtitles(entries(segs), 37, 2, transcript=text)
        self.assertEqual(hard_failures(report), {}, report)

    def test_speech_only_view(self):
        text = self.TEXT + "Yes, [?] question back there.\n"
        segs = segment(text, timed_words(text, start=3.0, pauses={4: 0.4}))
        speech = S.speech_only(segs)
        flat = " ".join(s["text"] for s in speech)
        self.assertNotIn("[laughter]", flat)
        self.assertNotIn("[chuckles]", flat)
        self.assertIn("[?]", flat)


class TestTiming(Layout):
    def _seg(self, a, b, text="x"):
        return {"start": S.ms_to_time(a), "end": S.ms_to_time(b), "text": text}

    def test_short_cue_keeps_its_start_in_sync_mode(self):
        segs = [self._seg(0, 1000, "first one here"), self._seg(1600, 1900, "Okay."),
                self._seg(1950, 4000, "and then the next one")]
        out = S.enforce_min_duration(segs, 1000)
        okay = [s for s in out if "Okay." in s["text"]][0]
        self.assertEqual(time_to_ms(okay["start"]), 1600)        # never before its word
        self.assertEqual(time_to_ms(okay["end"]), 1949)          # runs on to the next cue

    def test_short_cue_grows_backwards_without_sync(self):
        S.set_config(SegmentationConfig(max_cpl=37, max_lines=2, max_duration_ms=7000,
                                        min_cue_ms=1000, sync_starts=False))
        segs = [self._seg(0, 1000, "first one here"), self._seg(1600, 1900, "Okay."),
                self._seg(1950, 4000, "and then the next one")]
        out = S.enforce_min_duration(segs, 1000)
        okay = [s for s in out if "Okay." in s["text"]][0]
        self.assertGreaterEqual(time_to_ms(okay["end"]) - time_to_ms(okay["start"]), 1000)
        for a, b in zip(out, out[1:]):
            self.assertGreaterEqual(time_to_ms(b["start"]) - time_to_ms(a["end"]), 1)

    def test_chaining(self):
        segs = [self._seg(0, 1000, "one"), self._seg(1800, 2600, "two"),
                self._seg(5000, 6000, "three")]
        out = S.chain_cues(segs, chain_gap_ms=1200, lead_out_ms=500)
        self.assertEqual(time_to_ms(out[0]["end"]), 1799)        # short pause: runs on
        self.assertEqual(time_to_ms(out[1]["end"]), 3100)        # long pause: lingers 500 ms
        self.assertEqual(time_to_ms(out[2]["end"]), 6500)
        self.assertEqual([time_to_ms(s["start"]) for s in out], [0, 1800, 5000])

    def test_fast_cue_lingers_into_the_pause(self):
        segs = [self._seg(0, 1000, "This sentence is far too fast to read."),
                self._seg(4000, 6000, "Next.")]
        out = S.relieve_fast_cues(segs)
        dur = time_to_ms(out[0]["end"]) - time_to_ms(out[0]["start"])
        self.assertGreaterEqual(len(out[0]["text"]) * 1000 / dur, 19.9)
        self.assertLessEqual(len(out[0]["text"]) * 1000 / dur, 20.1)
        self.assertLess(time_to_ms(out[0]["end"]), 4000)

    def test_min_gap_is_kept(self):
        segs = [self._seg(0, 2000, "a"), self._seg(2000, 4000, "b")]
        out = S.resolve_overlaps(segs)
        self.assertGreaterEqual(time_to_ms(out[1]["start"]) - time_to_ms(out[0]["end"]), 1)


class TestCheck(unittest.TestCase):
    def test_reports_missing_words_and_tags(self):
        cues = [srt.Subtitle(1, timedelta(0), timedelta(seconds=2), "Good morning,"),
                srt.Subtitle(2, timedelta(seconds=2), timedelta(seconds=4), "everyone.")]
        report = check_subtitles(cues, 37, 2, transcript="Good morning,\n[laughter]\neveryone. Hi")
        self.assertEqual(report["transcript"]["missing"], 2)
        self.assertIn("[laughter]", report["transcript"]["missing_sample"])
        self.assertEqual(hard_failures(report)["gap_under_min"], 1)

    def test_timecodes_are_not_tags(self):
        self.assertEqual(transcript_stream("[00:01:02] Hello [END OF AUDIO]"), ["hello"])


if __name__ == "__main__":
    unittest.main()
