"""Stages 7-8: cue risk tiers, the review file, and calibration."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.qa.triage import (ACCEPT, DEFAULT_WEIGHTS, QUICK, REDO, fit_weights,
                                 load_weights, review_summary, save_weights,
                                 score_cue, triage_segments, write_review)
from wordalign.qa.triage_calibration import collect_examples, label_cues


def words(n, text_conf=1.0, timing_conf=0.95, t0=0.0, **extra):
    return [dict({"word": f"w{i}", "start": t0 + i * 0.4, "end": t0 + i * 0.4 + 0.3,
                  "text_conf": text_conf, "timing_conf": timing_conf}, **extra)
            for i in range(n)]


class TestScoring(unittest.TestCase):

    def test_clean_cue_is_accepted(self):
        tier, risk, est, reasons, _ = score_cue(words(8), [], 0, "ensemble")
        self.assertEqual(tier, ACCEPT)
        self.assertEqual(reasons, [])

    def test_disputed_words_need_a_check(self):
        ws = words(8)
        ws[2]["text_conf"] = ws[5]["text_conf"] = 0.5
        tier, _, _, reasons, _ = score_cue(ws, [], 0, "ensemble")
        self.assertEqual(tier, QUICK)
        self.assertTrue(any("disagree on 2 words" in r for r in reasons))

    def test_hopeless_cue_is_redo(self):
        tier, _, est, _, _ = score_cue(words(8, text_conf=0.4), ["overlap"], 0, "ensemble")
        self.assertEqual(tier, REDO)
        self.assertGreaterEqual(est, 0.3)

    def test_conditions_raise_risk(self):
        _, quiet, _, _, _ = score_cue(words(8), [], 0, "reference")
        _, noisy, _, reasons, _ = score_cue(words(8), ["very_low_snr", "music"], 0, "reference")
        self.assertGreater(noisy, quiet)
        self.assertIn("music under speech", reasons)

    def test_reference_mode_unheard_words(self):
        ws = words(6, heard_by=0)
        tier, _, est, reasons, _ = score_cue(ws, [], 0, "reference")
        self.assertEqual(tier, REDO)
        self.assertTrue(any("no engine heard" in r for r in reasons))

    def test_interpolated_words_are_reported(self):
        ws = words(6)
        for w in ws[2:4]:
            w["interpolated"] = True
            w["timing_conf"] = 0.2
        _, _, _, reasons, _ = score_cue(ws, [], 0, "reference")
        self.assertTrue(any("interpolated" in r for r in reasons))


class TestReviewFile(unittest.TestCase):

    def test_segments_and_queue(self):
        ws = words(10) + words(10, text_conf=0.4, t0=10.0)
        segs = [{"start": 0.0, "end": 4.0, "text": "a"}, {"start": 10.0, "end": 14.0, "text": "b"}]
        cues = triage_segments(segs, ws, "ensemble",
                               conditions_at=lambda s, e: ["music"] if s >= 10 else [],
                               removals=[{"start": 11.0, "reason": "loop"}],
                               weights=DEFAULT_WEIGHTS)
        self.assertEqual([c.tier for c in cues], [ACCEPT, REDO])
        self.assertTrue(any("hallucination filter" in r for r in cues[1].reasons))
        with tempfile.TemporaryDirectory() as d:
            path = write_review(os.path.join(d, "x_review.json"), cues, {"mode": "ensemble"})
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data["queue"], [2])
        self.assertEqual(data["summary"]["counts"]["redo"], 1)
        self.assertEqual(data["cues"][1]["tier_label"], "Redo")
        self.assertAlmostEqual(review_summary(cues)["share_to_review"], 0.5)


class TestCalibration(unittest.TestCase):

    def test_fit_moves_towards_the_data(self):
        # Cues with music were always corrected; others never.
        ex = []
        for _ in range(60):
            ex.append(({"music": 1.0, "text_error": 0.05}, 1))
            ex.append(({"music": 0.0, "text_error": 0.05}, 0))
        w = fit_weights(ex)
        self.assertGreater(w["music"], DEFAULT_WEIGHTS["music"])

    def test_weights_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"WORDALIGN_HOME": d}):
            save_weights(dict(DEFAULT_WEIGHTS, music=2.5), 10)
            self.assertEqual(load_weights()["music"], 2.5)

    def test_labels_from_corrected_srt(self):
        cues = [{"start": 0.0, "end": 2.0, "text": "hello there"},
                {"start": 3.0, "end": 5.0, "text": "how are you"},
                {"start": 6.0, "end": 8.0, "text": "fine thanks"}]
        corrected = [{"start": 0.0, "end": 2.0, "text": "Hello there."},
                     {"start": 3.0, "end": 5.0, "text": "who are you"},
                     {"start": 6.5, "end": 8.0, "text": "fine thanks"}]
        self.assertEqual(label_cues(cues, corrected), [0, 1, 1])

    def test_collect_examples(self):
        with tempfile.TemporaryDirectory() as d:
            review = {"cues": [{"start": 0.0, "end": 2.0, "text": "hello there",
                                "features": {"music": 1.0}},
                               {"start": 3.0, "end": 5.0, "text": "bye",
                                "features": {}}]}
            Path(d, "job_review.json").write_text(json.dumps(review), encoding="utf-8")
            Path(d, "job_corrected.srt").write_text(
                "1\n00:00:00,000 --> 00:00:02,000\nhello where\n\n"
                "2\n00:00:03,000 --> 00:00:05,000\nbye\n\n", encoding="utf-8")
            examples, used = collect_examples(d)
        self.assertEqual(used, ["job_review.json"])
        self.assertEqual([y for _, y in examples], [1, 0])


if __name__ == "__main__":
    unittest.main()
