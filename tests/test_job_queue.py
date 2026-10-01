"""GUI job queue, interrupted jobs, and the shared GPU slot."""
import contextlib
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.core.database import JobRecord, ProjectRecord, ProjectStore
from wordalign.core.gpu_scheduler import GPUScheduler
from wordalign.gui import server as gui_server
from wordalign.gui.server import PipelineAPIHandler


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _StoreCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        env = patch.dict(os.environ, {"XDG_DATA_HOME": tmp.name,
                                      "LOCALAPPDATA": tmp.name})
        env.start()
        self.addCleanup(env.stop)

    def status(self, job_id):
        store = ProjectStore()
        try:
            return store.get_job(job_id).status
        finally:
            store.close()


class TestJobQueue(_StoreCase):

    def setUp(self):
        super().setUp()
        self.running = set()
        self.max_running = 0
        self.order = []
        self.release = {}
        self.lock = threading.Lock()

        def fake_run(_self, job_id, params):
            with self.lock:
                self.running.add(job_id)
                self.max_running = max(self.max_running, len(self.running))
                self.order.append(job_id)
            self.release.setdefault(job_id, threading.Event()).wait(5)
            with self.lock:
                self.running.discard(job_id)
            store = ProjectStore()
            store.update_job(job_id, status="completed")
            store.close()

        patcher = patch.object(PipelineAPIHandler, "_run_pipeline_thread",
                               fake_run)
        patcher.start()
        self.addCleanup(patcher.stop)

    def start(self):
        result = PipelineAPIHandler._start_pipeline(None, {"audio_path": "a.wav"})
        self.release.setdefault(result["job_id"], threading.Event())
        return result

    def finish(self, job_id):
        self.release[job_id].set()

    def test_jobs_run_one_at_a_time_in_order(self):
        # Regression: every job got its own thread, so two GPU jobs ran
        # (and loaded models) at the same time.
        jobs = [self.start() for _ in range(3)]
        ids = [j["job_id"] for j in jobs]
        self.assertTrue(_wait_for(lambda: ids[0] in self.running))
        self.assertEqual(jobs[0]["status"], "queued")
        self.assertEqual(self.status(ids[1]), "queued")
        self.assertEqual(gui_server._queue_position(ids[1]), 1)
        self.assertEqual(gui_server._queue_position(ids[2]), 2)
        for job_id in ids:
            self.finish(job_id)
        self.assertTrue(_wait_for(
            lambda: all(self.status(j) == "completed" for j in ids)))
        self.assertEqual(self.order, ids)
        self.assertEqual(self.max_running, 1)

    def test_cancelling_a_queued_job_never_runs_it(self):
        first, second = self.start(), self.start()
        self.assertTrue(_wait_for(lambda: first["job_id"] in self.running))
        handler = PipelineAPIHandler.__new__(PipelineAPIHandler)
        sent = {}
        handler._send_json = lambda data, status=200: sent.update(data)
        handler._read_body = lambda: {"job_id": second["job_id"]}
        handler.path = "/api/run/cancel"
        handler.headers = {"Host": "127.0.0.1:1"}
        handler.server = type("S", (), {"server_address": ("127.0.0.1", 1)})()
        handler.do_POST()
        self.assertTrue(sent.get("ok"))
        self.assertEqual(self.status(second["job_id"]), "cancelled")
        self.assertIsNone(gui_server._queue_position(second["job_id"]))
        self.finish(first["job_id"])
        self.assertTrue(_wait_for(
            lambda: self.status(first["job_id"]) == "completed"))
        time.sleep(0.1)
        self.assertNotIn(second["job_id"], self.order)


class TestInterruptedJobs(_StoreCase):

    def _jobs(self, statuses):
        store = ProjectStore()
        store.create_project(ProjectRecord(id="p", name="p", audio_path="a"))
        for job_id, status in statuses.items():
            store.create_job(JobRecord(id=job_id, project_id="p",
                                       status=status, started_at=time.time()))
        store.close()

    def test_server_start_marks_leftover_jobs_interrupted(self):
        self._jobs({"r": "running", "q": "queued", "d": "completed"})

        class OneShotServer:
            def __init__(self, *args):
                pass

            def serve_forever(self):
                raise KeyboardInterrupt

            def shutdown(self):
                pass

        with patch.object(gui_server, "ThreadingHTTPServer", OneShotServer), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            gui_server.run_server(port=1, open_browser=False)
        self.assertEqual(self.status("r"), "interrupted")
        self.assertEqual(self.status("q"), "interrupted")
        self.assertEqual(self.status("d"), "completed")
        self.assertIn("2 unfinished job(s)", out.getvalue())

    def test_recovery_lists_interrupted_but_not_live_jobs(self):
        from wordalign.core.recovery import CrashRecovery
        self._jobs({"old": "interrupted", "live": "running"})
        store = ProjectStore()
        store.update_job("live", started_at=time.time() - 3600)
        report = CrashRecovery(store).find_incomplete_jobs(
            exclude_job_ids={"live"})
        store.close()
        self.assertEqual([i.job.id for i in report.incomplete], ["old"])

    def test_gui_stops_polling_interrupted_jobs(self):
        html = (Path(__file__).resolve().parent.parent / "wordalign" / "gui" /
                "app.html").read_text(encoding="utf-8")
        self.assertIn("['failed', 'cancelled', 'interrupted'].includes(job.status)",
                      html)
        self.assertIn("job.status === 'queued'", html)


class TestGpuScheduler(unittest.TestCase):

    def test_same_engine_slots_are_tracked_separately(self):
        sched = GPUScheduler(max_concurrent=2)
        a = sched.acquire("whisperx")
        b = sched.acquire("whisperx")
        self.assertTrue(sched.release(a))
        self.assertEqual(sched.active_count, 1)
        self.assertTrue(sched.release(b))
        self.assertEqual(sched.active_count, 0)

    def test_releasing_an_unheld_slot_is_harmless(self):
        # Regression: release() always released the semaphore, so a double
        # release raised ValueError (BoundedSemaphore) or freed a slot that
        # another caller held.
        sched = GPUScheduler(max_concurrent=1)
        slot = sched.acquire("qwen")
        self.assertTrue(sched.release(slot))
        self.assertFalse(sched.release(slot))
        self.assertFalse(sched.release("whisperx"))
        sched.acquire("whisperx", timeout_seconds=0.1)

    def _runner(self, d, loader):
        from wordalign.config import PipelineConfig
        from wordalign.core.pipeline import PipelineRunner
        audio = Path(d) / "a.wav"
        audio.write_bytes(b"x")
        runner = PipelineRunner(PipelineConfig(audio_path=str(audio)))
        return runner, lambda: runner._cached_engine(
            "whisperx", "large-v3", "en", {}, loader)

    def test_gpu_engines_from_concurrent_runs_never_overlap(self):
        active, peak = [0], [0]
        lock = threading.Lock()

        def loader():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            with lock:
                active[0] -= 1
            return ([], [])

        with tempfile.TemporaryDirectory() as d, \
                patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}), \
                patch("wordalign.core.gpu_scheduler._SHARED",
                      GPUScheduler(max_concurrent=1)):
            calls = [self._runner(d, loader)[1] for _ in range(3)]
            threads = [threading.Thread(target=c) for c in calls]
            for t in threads:
                t.start()
            for t in threads:
                t.join(5)
        self.assertEqual(peak[0], 1)

    def test_cancel_while_waiting_for_the_gpu(self):
        from wordalign.core.errors import PipelineCancelled
        sched = GPUScheduler(max_concurrent=1)
        held = sched.acquire("qwen")
        with tempfile.TemporaryDirectory() as d, \
                patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}), \
                patch("wordalign.core.gpu_scheduler._SHARED", sched):
            runner, call = self._runner(d, lambda: self.fail("must not run"))
            threading.Timer(0.3, runner.cancel).start()
            with self.assertRaises(PipelineCancelled):
                call()
        sched.release(held)

    def test_cpu_device_does_not_take_the_gpu_slot(self):
        sched = GPUScheduler(max_concurrent=1)
        held = sched.acquire("qwen")
        with tempfile.TemporaryDirectory() as d, \
                patch.dict(os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}), \
                patch("wordalign.core.gpu_scheduler._SHARED", sched):
            runner, call = self._runner(d, lambda: ([], []))
            runner.config.device = "cpu"
            self.assertEqual(call(), (([], []), False))
        sched.release(held)


if __name__ == "__main__":
    unittest.main()
