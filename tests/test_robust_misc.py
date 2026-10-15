"""Robust flow plumbing: CLI/profile settings, backends, dash cues, SDH
proposals, overlap regions, and the benchmark's metrics and suite builder."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestSettings(unittest.TestCase):

    def test_cli_flags(self):
        from wordalign.cli_v2 import _parse_args
        cfg = _parse_args(["a.wav", "--cleanup", "auto", "--voters", "cohere,qwen",
                           "--overlap-map", "--no-triage", "--timing", "fill_only"])
        self.assertEqual(cfg.cleanup, "auto")
        self.assertEqual(cfg.voters, ("cohere", "qwen"))
        self.assertTrue(cfg.overlap_map)
        self.assertFalse(cfg.triage)
        self.assertEqual(cfg.timing_strategy, "fill_only")
        self.assertEqual(cfg.robust_overrides["voters"], ("cohere", "qwen"))

    def test_unknown_voter_is_rejected(self):
        from wordalign.cli_v2 import _parse_args
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            _parse_args(["a.wav", "--voters", "cohere,nope"])

    def test_defaults(self):
        from wordalign.config import PipelineConfig
        cfg = PipelineConfig(audio_path="a.wav")
        self.assertTrue(cfg.analysis and cfg.hallucination_filter and cfg.triage)
        self.assertEqual(cfg.timing_strategy, "anchor")
        self.assertEqual(cfg.cleanup, "off")
        self.assertFalse(cfg.overlap_asr)

    def test_robust_profile_and_flag_precedence(self):
        from wordalign.config import PipelineConfig
        from wordalign.core.config import PipelineProfile
        from wordalign.core.pipeline import PipelineRunner
        for profile in (PipelineProfile.preset("robust"),
                        PipelineProfile.from_json(ROOT / "wordalign" / "profiles" / "robust.json")):
            cfg = PipelineConfig(audio_path="a.wav", robust_overrides={"cleanup": "off"})
            PipelineRunner(cfg, profile=profile)
            self.assertEqual(cfg.voters, ("cohere", "qwen", "parakeet", "whisperx"))
            self.assertTrue(cfg.overlap_map)
            self.assertEqual(cfg.cleanup, "off")          # the CLI flag wins

    def test_overlap_asr_turns_on_speaker_labels(self):
        from wordalign.config import PipelineConfig
        from wordalign.core.pipeline import PipelineRunner
        cfg = PipelineConfig(audio_path="a.wav", overlap_asr=True)
        PipelineRunner(cfg)
        self.assertTrue(cfg.diarize)


class TestBackends(unittest.TestCase):

    def test_env_var_then_venv_folder(self):
        from wordalign.backends import backend_python
        with tempfile.TemporaryDirectory() as d:
            py = Path(d, "cohere", "Scripts", "python.exe")
            py.parent.mkdir(parents=True)
            py.write_bytes(b"")
            with patch.dict(os.environ, {"WORDALIGN_VENVS_DIR": d}, clear=False):
                os.environ.pop("WORDALIGN_COHERE_PYTHON", None)
                self.assertEqual(backend_python("cohere"), str(py))
                # sed shares the whisperx venv, which is absent here.
                os.environ.pop("WORDALIGN_SED_PYTHON", None)
                os.environ.pop("WORDALIGN_WHISPERX_PYTHON", None)
                self.assertIsNone(backend_python("sed"))
                wx = Path(d, "whisperx", "Scripts", "python.exe")
                wx.parent.mkdir(parents=True)
                wx.write_bytes(b"")
                self.assertEqual(backend_python("sed"), str(wx))

    def test_models_dir_override(self):
        from wordalign.models.paths import _project_models_root
        with patch.dict(os.environ, {"WORDALIGN_MODELS_DIR": "/x/models"}):
            self.assertEqual(str(_project_models_root()).replace("\\", "/"), "/x/models")

    def test_doctor_lists_every_stage(self):
        from wordalign.doctor import collect
        stages = {row[0] for row in collect()}
        for stage in ("1 analysis", "2 cleanup", "3 voting", "4 crosstalk", "5 timing"):
            self.assertIn(stage, stages)


class TestCuesAndTags(unittest.TestCase):

    def test_dash_cues(self):
        from wordalign.core.config import SegmentationConfig
        from wordalign.segment import get_config, merge_dash_cues, set_config
        previous = get_config()
        self.addCleanup(lambda: set_config(SegmentationConfig(**previous)))
        set_config(SegmentationConfig(max_cpl=42, max_lines=2))
        segs = [{"start": "00:00:01,000", "end": "00:00:02,000", "text": "Are you coming?",
                 "speaker": "Speaker 1"},
                {"start": "00:00:02,100", "end": "00:00:03,000", "text": "Yes.",
                 "speaker": "Speaker 2"},
                {"start": "00:00:09,000", "end": "00:00:10,000", "text": "Later.",
                 "speaker": "Speaker 1"}]
        out = merge_dash_cues(segs, style="-")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["text"], "-Are you coming?\n-Yes.")
        self.assertTrue(out[0]["dash_cue"])
        self.assertEqual(merge_dash_cues(segs, style="- ")[0]["text"],
                         "- Are you coming?\n- Yes.")

    def test_sed_proposals(self):
        from wordalign.tags import sed_detections, sed_proposals
        events = [{"label": "Music", "start": 0.0, "end": 5.0, "score": 0.9},
                  {"label": "Music", "start": 5.5, "end": 9.0, "score": 0.8},
                  {"label": "Laughter", "start": 12.0, "end": 12.2, "score": 0.9},
                  {"label": "Laughter", "start": 20.0, "end": 21.0, "score": 0.9},
                  {"label": "Speech", "start": 0.0, "end": 30.0, "score": 0.99}]
        props = sed_proposals(events)
        self.assertEqual([(p["tag"], p["start"]) for p in props],
                         [("music", 0.0), ("laughter", 20.0)])
        self.assertEqual(sed_detections(events)[0]["tag"], "Music")

    def test_overlap_regions(self):
        from wordalign.diarize import overlap_regions
        turns = [(0.0, 5.0, "A"), (4.0, 8.0, "B"), (7.5, 9.0, "A"), (20.0, 21.0, "B")]
        self.assertEqual(overlap_regions(turns), [(4.0, 5.0), (7.5, 8.0)])


class TestBenchmark(unittest.TestCase):

    def test_wer_and_timing(self):
        from wordalign.robustness import metrics
        r = metrics.wer("The cat sat on the mat.", "the cat sat on a mat")
        self.assertAlmostEqual(r["wer"], 1 / 6)
        truth = [{"word": "a", "start": 0.0}, {"word": "b", "start": 1.0}]
        hyp = [{"word": "a", "start": 0.04}, {"word": "b", "start": 1.2}]
        t = metrics.timing_errors(truth, hyp)
        self.assertEqual(t["matched"], 2)
        self.assertAlmostEqual(t["mean_ms"], 120.0)
        self.assertEqual(t["within_100ms"], 0.5)

    def test_mix_at_snr(self):
        from wordalign.robustness import degrade
        rng = np.random.default_rng(0)
        speech = rng.standard_normal(16000).astype(np.float32) * 0.1
        noise = rng.standard_normal(16000).astype(np.float32)
        mixed = degrade.mix_at_snr(speech, noise, 0.0, rng)
        resid = mixed - speech
        snr = 10 * np.log10(np.mean(speech ** 2) / np.mean(resid ** 2))
        self.assertAlmostEqual(snr, 0.0, delta=1.0)

    def test_suite_build_and_evaluate(self):
        from wordalign.audio.io import write_wav
        from wordalign.robustness import runner, suite
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            sp = Path(d, "src", "speech")
            sp.mkdir(parents=True)
            index = []
            for k in range(6):
                uid = f"{100 + k % 3}-1-{k}"
                write_wav(str(sp / f"{uid}.wav"), (rng.standard_normal(3 * 16000) * 0.1).astype(np.float32))
                (sp / f"{uid}.words.json").write_text(json.dumps(
                    [{"word": f"W{k}x{i}", "start": 0.5 + i * 0.5, "end": 0.9 + i * 0.5}
                     for i in range(4)]), encoding="utf-8")
                index.append({"id": uid, "speaker": uid.split("-")[0], "sex": "F", "seconds": 3.0})
            fake = {"speech": index, "noise": [], "music": [], "dest": str(Path(d, "src"))}
            with patch("wordalign.robustness.suite.ensure_sources", lambda *a, **k: fake), \
                    patch("wordalign.robustness.degrade.codec", lambda x, kind: x):
                manifest = suite.build_suite(str(Path(d, "suite")), str(Path(d, "src")),
                                             conditions=["clean", "noise_5dB", "overlap_5dB",
                                                         "only_noise"],
                                             block_seconds=5.0, log=lambda m: None)
            self.assertEqual(manifest["blocks"], 4)
            truth = json.loads(Path(d, "suite", "bench.words.json").read_text())
            blocks = json.loads(Path(d, "suite", "bench.blocks.json").read_text())
            self.assertTrue(any(w.get("interferer") for w in truth))
            self.assertTrue(Path(d, "suite", "bench.txt").read_text().strip())
            # A perfect hypothesis plus two invented words in the noise block.
            hyp = [dict(w) for w in truth]
            nb = blocks[3]
            hyp += [{"word": "thanks", "start": nb["start"] + 1, "end": nb["start"] + 1.3},
                    {"word": "bye", "start": nb["start"] + 2, "end": nb["start"] + 2.3}]
            scores = runner.evaluate(truth, blocks, hyp, "free")
            self.assertEqual(scores["overall"]["wer"], 0.0)
            self.assertEqual(scores["overall"]["mean_shift_ms"], 0.0)
            self.assertGreater(scores["overall"]["invented_per_10min"], 0)
            report = {"suite": "x", "variants": {"v2.2": {"free": scores, "flags": []}}}
            md = runner.render_markdown(report)
            self.assertIn("| v2.2 | free | 0.0%", md)


class TestFetchModels(unittest.TestCase):

    def test_list_without_network(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_models.py"),
                                  "--models-dir", d, "--list"],
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("vad", out.stdout)


if __name__ == "__main__":
    unittest.main()
