"""QA issues and the project store across multiple jobs."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.converters import dicts_to_words
from wordalign.core.database import JobRecord, ProjectRecord, ProjectStore
from wordalign.qa.deterministic import detect_timing_anomalies


class TestProjectStoreAcrossJobs(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = ProjectStore(db_path=os.path.join(tmp.name, "p.db"))
        self.addCleanup(self.store.close)
        self.store.create_project(ProjectRecord(id="p", name="p",
                                                audio_path="a.wav"))
        for job in ("job1", "job2"):
            self.store.create_job(JobRecord(id=job, project_id="p"))

    def test_same_issue_id_in_two_jobs_keeps_both(self):
        # Regression: deterministic ids like "issue_agree_0003" repeat per
        # job; with a global primary key + INSERT OR REPLACE, the second
        # job took the first job's issue away.
        issue = {"id": "issue_agree_0003", "category": "agreement",
                 "start": 1.0, "end": 2.0}
        self.store.save_qa_issue("job1", dict(issue, original_text="first"))
        self.store.save_qa_issue("job2", dict(issue, original_text="second"))
        job1 = self.store.list_qa_issues("job1")
        job2 = self.store.list_qa_issues("job2")
        self.assertEqual([i["original_text"] for i in job1], ["first"])
        self.assertEqual([i["original_text"] for i in job2], ["second"])

    def test_resaving_an_issue_updates_it_in_place(self):
        issue = {"id": "issue_rep_0001_2", "category": "repetition"}
        self.store.save_qa_issue("job1", dict(issue, status="open"))
        saved = self.store.list_qa_issues("job1")[0]
        self.store.save_qa_issue("job1", dict(saved, status="dismissed"))
        issues = self.store.list_qa_issues("job1")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["status"], "dismissed")

    def test_global_glossary_term_is_not_duplicated(self):
        # Regression: UNIQUE(project_id, term) ignores NULL project ids.
        self.store.add_glossary_entry("GitHub", correct_form="GitHub")
        self.store.add_glossary_entry("github", correct_form="GitHub",
                                      context="brand")
        entries = self.store.list_glossary()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["context"], "brand")

    def test_project_glossary_term_is_separate_from_global(self):
        self.store.add_glossary_entry("kubectl")
        self.store.add_glossary_entry("kubectl", project_id="p")
        self.assertEqual(len(self.store.list_glossary("p")), 2)


class TestInterpolatedRunDetection(unittest.TestCase):

    def _words(self, sources):
        return dicts_to_words([
            {"word": f"w{i}", "start": i * 0.5, "end": i * 0.5 + 0.3,
             "source": src}
            for i, src in enumerate(sources)])

    def _interp_issues(self, sources):
        return [i for i in detect_timing_anomalies(self._words(sources))
                if i.id.startswith("issue_timing_interp")]

    def test_trailing_interpolated_run_is_flagged(self):
        # Regression: a run reaching the end of the file was never flushed.
        issues = self._interp_issues(["Vosk"] * 3 + ["Interpolated"] * 6)
        self.assertEqual(len(issues), 1)
        self.assertEqual(len(issues[0].word_ids), 6)

    def test_interior_run_is_still_flagged_once(self):
        issues = self._interp_issues(
            ["Vosk"] + ["Interpolated"] * 5 + ["Vosk"] * 2)
        self.assertEqual(len(issues), 1)

    def test_short_runs_are_not_flagged(self):
        self.assertEqual(
            self._interp_issues(["Vosk"] + ["Interpolated"] * 4), [])


if __name__ == "__main__":
    unittest.main()
