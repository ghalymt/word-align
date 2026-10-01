"""MFA is skipped for languages it has no model for -- never English timing."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign import align
from wordalign.config import PipelineConfig
from wordalign.core.events import CollectSink, StageMessage, StageStarted
from wordalign.core.pipeline import PipelineRunner
from wordalign.engines.mfa_engine import (MFAWrapper, get_mfa_models,
                                          mfa_supports_language)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


class TestLanguageMap(unittest.TestCase):

    def test_unsupported_language_has_no_model(self):
        # Regression: unknown languages silently used english_us_arpa.
        self.assertIsNone(get_mfa_models("is"))
        self.assertFalse(mfa_supports_language("is"))

    def test_regional_codes_use_the_primary_subtag(self):
        self.assertEqual(get_mfa_models("zh-CN")["acoustic"],
                         "mandarin_chinese_mfa")
        self.assertEqual(get_mfa_models("pt_BR"),
                         get_mfa_models("pt"))
        self.assertTrue(mfa_supports_language("EN"))

    def test_wrapper_refuses_unsupported_language(self):
        with tempfile.TemporaryDirectory() as d, \
                patch.object(MFAWrapper, "_resolve_mfa", return_value="mfa"):
            with self.assertRaises(ValueError):
                MFAWrapper(work_dir=d, language="is")


class TestSurgicalMfaSkips(unittest.TestCase):

    def _gappy_words(self):
        return [{"word": "a", "start": 0.0, "end": 0.2, "matched": True},
                {"word": "b"},
                {"word": "c", "start": 1.0, "end": 1.2, "matched": True}]

    def test_unsupported_language_does_not_slice_audio(self):
        sliced = []
        with patch.object(MFAWrapper, "_resolve_mfa", return_value="mfa"), \
                patch.object(align, "create_chunk_wav",
                             lambda *a, **k: sliced.append(a)), _quiet():
            words = self._gappy_words()
            align.make_surgical_mfa(words, "audio.wav", language="is")
        self.assertEqual(sliced, [])
        self.assertFalse(words[1].get("matched"))

    def test_missing_model_files_skip_before_slicing(self):
        sliced = []
        with tempfile.TemporaryDirectory() as empty, \
                patch.object(MFAWrapper, "_resolve_mfa", return_value="mfa"), \
                patch("wordalign.models.paths._project_models_root",
                      return_value=Path(empty)), \
                patch.dict(os.environ, {"MFA_ROOT_DIR": empty}), \
                patch("pathlib.Path.home", return_value=Path(empty)), \
                patch.object(align, "create_chunk_wav",
                             lambda *a, **k: sliced.append(a)), _quiet() as out:
            align.make_surgical_mfa(self._gappy_words(), "audio.wav",
                                    language="en")
        self.assertEqual(sliced, [])
        self.assertIn("MFA skipped", out.getvalue())


class TestPipelineMfaStage(unittest.TestCase):

    def test_pipeline_reports_skip_for_unsupported_language(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(
                os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            audio = Path(d) / "a.wav"
            audio.write_bytes(b"x")
            transcript = Path(d) / "t.txt"
            transcript.write_text("halló heimur", encoding="utf-8")
            cfg = PipelineConfig(audio_path=str(audio),
                                 transcript_path=str(transcript),
                                 output_dir=d, language="is",
                                 use_vosk=False, use_qwen=False,
                                 use_whisperx=False, use_mfa=True)
            sink = CollectSink()
            ran = []
            with patch("wordalign.core.pipeline.make_surgical_mfa",
                       lambda *a, **k: ran.append(a)), _quiet():
                PipelineRunner(cfg, sink=sink).run()
        self.assertEqual(ran, [])
        msgs = [e.message for e in sink.events if isinstance(e, StageMessage)]
        self.assertTrue(any("MFA skipped" in m for m in msgs), msgs)
        self.assertFalse(any(isinstance(e, StageStarted) and e.stage == "mfa"
                             for e in sink.events))


class TestLanguageDetection(unittest.TestCase):

    def test_detected_region_suffix_is_dropped(self):
        from wordalign.utils import detect_language
        with patch("langdetect.detect", return_value="zh-cn"), _quiet():
            self.assertEqual(detect_language("这是一个测试"), "zh")


class TestMfaDefaults(unittest.TestCase):

    def test_gui_balanced_preset_matches_profile(self):
        root = Path(__file__).resolve().parent.parent
        html = (root / "wordalign" / "gui" / "app.html").read_text(encoding="utf-8")
        self.assertIn("balanced: ['vosk', 'qwen_asr', 'whisperx', 'mfa']", html)
        import json
        balanced = json.loads((root / "wordalign" / "profiles" /
                               "balanced.json").read_text(encoding="utf-8"))
        self.assertIn("mfa", balanced["timing"]["order"])

    def test_api_run_without_mfa_flag_matches_cli_default(self):
        from wordalign.core.types import PipelineResult
        from wordalign.gui.server import PipelineAPIHandler
        from wordalign.cli_v2 import _parse_args
        seen = {}

        def fake_run(runner):
            seen["cfg"] = runner.config
            return PipelineResult(cancelled=True)

        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
                "XDG_DATA_HOME": d, "LOCALAPPDATA": d}), \
                patch("wordalign.core.pipeline.PipelineRunner.run", fake_run):
            from wordalign.core.database import JobRecord, ProjectRecord, ProjectStore
            store = ProjectStore()
            store.create_project(ProjectRecord(id="p", name="p", audio_path="a"))
            store.create_job(JobRecord(id="j", project_id="p"))
            store.close()
            PipelineAPIHandler._run_pipeline_thread(None, "j",
                                                    {"audio_path": "a.wav"})
        self.assertTrue(seen["cfg"].use_mfa)
        self.assertTrue(_parse_args(["a.wav"]).use_mfa)


if __name__ == "__main__":
    unittest.main()
