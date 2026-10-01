import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.config import PipelineConfig
from wordalign.core.cache import StageCache
from wordalign.core.config import PipelineProfile
from wordalign.core.database import JobRecord, ProjectRecord, ProjectStore
from wordalign.core.events import CancelledEvent, CollectSink, StageCompleted, StageStarted
from wordalign.core.pipeline import PipelineRunner
from wordalign.document import export_transcript
from wordalign.gui.server import _JobSink


class TestPipelineIntegration(unittest.TestCase):
    def test_profile_controls_engine_selection(self):
        cfg = PipelineConfig("audio.wav", profile_name="cpu_only")
        runner = PipelineRunner(cfg)
        self.assertTrue(cfg.use_vosk)
        self.assertFalse(cfg.use_qwen)
        self.assertFalse(cfg.use_whisperx)
        self.assertTrue(cfg.qa)
        self.assertEqual(runner.profile.name, "CPU Only")

    def test_explicit_profile_overrides_win(self):
        from wordalign.cli_v2 import _parse_args
        cfg = _parse_args([
            "audio.wav", "--profile", "fast", "--max-cpl", "32",
            "--doc", "none"])
        runner = PipelineRunner(cfg)
        self.assertEqual(runner.profile.segmentation.max_cpl, 32)
        self.assertEqual(runner.profile.export.transcript_format, "none")

    def test_transcript_media_mismatch_stops_before_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.mp4"
            transcript = Path(directory) / "transcript.txt"
            audio.write_bytes(b"video")
            transcript.write_text(" ".join(["word"] * 1000))
            cfg = PipelineConfig(
                audio_path=str(audio), transcript_path=str(transcript),
                output_dir=directory, use_vosk=False, use_qwen=False,
                use_whisperx=False, use_mfa=False)
            with patch("wordalign.core.pipeline.get_audio_duration",
                       return_value=10.0):
                result = PipelineRunner(cfg).run()
            self.assertIn("Transcript/media mismatch", result.error)
            self.assertEqual(result.output_files if hasattr(result, "output_files") else [], [])
            self.assertFalse(result.word_level_srt_path)

    def test_cli_exposes_transcript_mismatch_override(self):
        from wordalign.cli_v2 import _parse_args
        cfg = _parse_args(["audio.mp4", "-t", "transcript.txt",
                           "--allow-transcript-mismatch"])
        self.assertTrue(cfg.allow_transcript_mismatch)

    def test_cache_accepts_absolute_model_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = StageCache(cache_dir=directory)
            cache.put(r"C:\models\vosk\vosk-model-en", {"words": []})
            self.assertIsNotNone(cache.get(r"C:\models\vosk\vosk-model-en"))

    def test_profile_object_can_be_passed_directly(self):
        profile = PipelineProfile.from_dict({
            "name": "Custom",
            "transcription": {"engines": [{"plugin": "vosk", "enabled": True}]},
            "segmentation": {"max_cpl": 30},
        })
        runner = PipelineRunner(PipelineConfig("audio.wav"), profile=profile)
        self.assertTrue(runner.config.use_vosk)
        self.assertFalse(runner.config.use_whisperx)
        self.assertEqual(runner.profile.segmentation.max_cpl, 30)

    def test_primary_timings_and_manifest_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"not-real-audio")
            cfg = PipelineConfig(
                audio_path=str(audio), output_dir=directory,
                use_vosk=False, use_qwen=False, use_whisperx=True,
                use_mfa=False)
            sink = CollectSink()
            runner = PipelineRunner(cfg, sink=sink)
            runner._run_primary_transcript = lambda config, language: (
                [{"word": "Hello", "start": 1.0, "end": 1.2, "conf": 0.9},
                 {"word": "world", "start": 1.2, "end": 1.5, "conf": 0.9}],
                "Hello world", [], "whisperx")
            result = runner.run()
            self.assertEqual([word.timing_source for word in result.aligned_words],
                             ["WhisperX", "WhisperX"])
            self.assertTrue(result.transcript_txt_path)
            self.assertTrue(result.job_manifest_path)
            self.assertEqual(len(result.segments), 1)
            stage_names = {event.stage for event in sink.events
                           if isinstance(event, (StageStarted, StageCompleted))}
            self.assertIn("primary", stage_names)
            self.assertIn("output", stage_names)

    def test_pre_cancelled_run_emits_cancelled_event(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"audio")
            sink = CollectSink()
            runner = PipelineRunner(PipelineConfig(
                audio_path=str(audio), output_dir=directory,
                use_vosk=False, use_qwen=False, use_whisperx=False,
                use_mfa=False), sink=sink)
            runner.cancel()
            runner.run()
            self.assertTrue(any(isinstance(event, CancelledEvent)
                                for event in sink.events))

    def test_cancellation_during_primary_is_terminal_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"audio")
            sink = CollectSink()
            runner = PipelineRunner(PipelineConfig(
                audio_path=str(audio), output_dir=directory,
                use_vosk=False, use_qwen=True, use_whisperx=False,
                use_mfa=False), sink=sink)

            def cancel_primary(config, language):
                runner.cancel()
                return [], "", [], None

            runner._run_primary_transcript = cancel_primary
            result = runner.run()
            self.assertTrue(result.cancelled)
            self.assertTrue(any(isinstance(event, CancelledEvent)
                                for event in sink.events))

    def test_legacy_ensemble_path_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "audio.wav"
            audio.write_bytes(b"audio")
            cfg = PipelineConfig(
                audio_path=str(audio), output_dir=directory,
                use_legacy_ensemble=True, use_vosk=False, use_qwen=False,
                use_whisperx=False, use_mfa=False)
            runner = PipelineRunner(cfg)
            runner._run_ensemble = lambda config, language: (
                [{"word": "Hello", "start": 0.0, "end": 0.4, "conf": 0.9}],
                "Hello", [], [])
            result = runner.run()
            self.assertEqual(result.aligned_words[0].timing_source,
                             "Legacy ensemble")

    def test_transcript_export_returns_written_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = str(Path(directory) / "audio")
            paths = export_transcript(
                base, "audio.wav",
                [{"word": "Hello.", "start": 0.0, "end": 0.4, "conf": 0.9}])
            self.assertTrue(Path(paths["transcript_txt_path"]).exists())


class TestGuiPersistence(unittest.TestCase):
    def test_job_sink_records_stage_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProjectStore(str(Path(directory) / "jobs.db"))
            store.create_project(ProjectRecord(
                id="project", name="Project", audio_path="audio.wav"))
            store.create_job(JobRecord(
                id="job", project_id="project", status="running"))
            store.update_job("job", warnings=["qwen: skipped"])
            self.assertEqual(store.get_job("job").warnings, ["qwen: skipped"])
            sink = _JobSink("job", store)
            sink.emit(StageStarted(stage="vosk", message="start"))
            sink.emit(StageCompleted(stage="vosk", duration_seconds=1.5))
            stages = store.list_stages("job")
            self.assertEqual(len(stages), 1)
            self.assertEqual(stages[0]["status"], "completed")
            self.assertEqual(stages[0]["duration_seconds"], 1.5)
            store.close()


class TestPackaging(unittest.TestCase):
    def test_no_argument_cli_launches_gui(self):
        from wordalign.cli_v2 import main
        with patch("wordalign.gui.server.run_server") as run_server:
            self.assertEqual(main([]), 0)
            run_server.assert_called_once()

    def test_frozen_build_skips_unconfigured_optional_backends(self):
        from wordalign.engines.qwen_engine import run_qwen
        from wordalign.engines.whisperx_engine import run_whisperx
        with patch.object(sys, "frozen", True, create=True), \
             patch.dict("os.environ", {}, clear=True):
            self.assertEqual(run_whisperx("audio.wav", "en"), ([], []))
            self.assertEqual(run_qwen("audio.wav", "en"), [])

    def test_qwen_worker_normalizes_media_with_ffmpeg(self):
        from wordalign.engines._qwen_worker import _normalize_audio
        with tempfile.TemporaryDirectory() as directory:
            ffmpeg = Path(directory) / "ffmpeg.exe"
            ffmpeg.write_bytes(b"fake")
            output = {}

            def fake_run(command, **_kwargs):
                Path(command[-1]).write_bytes(b"RIFF" + b"0" * 128)
                return None

            with patch.dict(os.environ, {"WORDALIGN_FFMPEG": str(ffmpeg)}), \
                 patch("wordalign.engines._qwen_worker.subprocess.run",
                       side_effect=fake_run):
                output["path"] = _normalize_audio("input.mp4")
            self.assertTrue(Path(output["path"]).exists())
            os.remove(output["path"])

    def test_project_metadata_is_present(self):
        data = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
        self.assertIn('name = "wordalign"', data)
        self.assertIn('wordalign = "wordalign.cli_v2:main"', data)
        self.assertTrue((Path(__file__).resolve().parent.parent /
                         "scripts" / "Launch WordAlign.vbs").exists())

    def test_gui_persists_jobs_and_exposes_mismatch_override(self):
        html = (Path(__file__).resolve().parent.parent /
                "wordalign" / "gui" / "app.html").read_text(encoding="utf-8")
        self.assertIn("ACTIVE_JOB_KEY", html)
        self.assertIn("resumeStoredJob", html)
        self.assertIn("allow-mismatch", html)
        self.assertIn("job.warnings", html)

    def test_profile_json_qa_is_loaded(self):
        path = Path(__file__).resolve().parent.parent / "wordalign" / "profiles" / "balanced.json"
        profile = PipelineProfile.from_json(path)
        self.assertTrue(profile.qa.enabled)


class TestEnginesFlagPrecedence(unittest.TestCase):
    """Explicit --use-mfa / --no-* flags must win over --engines."""

    @staticmethod
    def _config(*args):
        from wordalign.cli_v2 import _parse_args
        return PipelineRunner(_parse_args(["audio.wav", *args])).config

    def test_readme_example_runs_mfa(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk", "--use-mfa")
        self.assertTrue(cfg.use_mfa)
        self.assertTrue(cfg.use_whisperx)
        self.assertTrue(cfg.use_qwen)
        self.assertTrue(cfg.use_vosk)

    def test_flag_order_does_not_matter(self):
        cfg = self._config("--use-mfa", "--engines", "whisperx,qwen,vosk")
        self.assertTrue(cfg.use_mfa)

    def test_no_mfa_wins_over_engines_listing_mfa(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk,mfa", "--no-mfa")
        self.assertFalse(cfg.use_mfa)

    def test_no_engine_flags_win_over_engines_listing(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk",
                           "--no-vosk", "--no-qwen", "--no-whisperx")
        self.assertFalse(cfg.use_vosk)
        self.assertFalse(cfg.use_qwen)
        self.assertFalse(cfg.use_whisperx)

    def test_engines_alone_still_selects_exactly_those(self):
        cfg = self._config("--engines", "whisperx,qwen,vosk")
        self.assertTrue(cfg.use_whisperx)
        self.assertFalse(cfg.use_mfa)
        cfg = self._config("--engines", "whisperx,mfa")
        self.assertTrue(cfg.use_mfa)
        self.assertFalse(cfg.use_vosk)
        self.assertFalse(cfg.use_qwen)


if __name__ == "__main__":
    unittest.main()
