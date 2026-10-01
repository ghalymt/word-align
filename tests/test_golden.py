"""Golden-file tests: recorded engine output -> exact subtitle files.

Each folder under tests/golden/ is a scenario: ``scenario.json`` (CLI flags,
optional transcript, and which recorded engine output to feed each engine)
plus ``expected_*`` files. The real pipeline runs with the engines replaced
by their recordings, and every expected file must match byte for byte.

These pin the whole chain -- matching, refinement, interpolation,
segmentation, layout, export -- so regressions like "the whole file is one
cue" or "the refinement engines change nothing" fail loudly.

After an intentional output change, regenerate and review the diff:

    WORDALIGN_REGEN_GOLDEN=1 python -m pytest tests/test_golden.py
"""
import contextlib
import difflib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN = Path(__file__).resolve().parent / "golden"
REGEN = os.environ.get("WORDALIGN_REGEN_GOLDEN") == "1"
# Outputs compared per scenario (the manifest has timestamps, so it is not).
OUTPUT_SUFFIXES = ("_sentence_level.srt", "_word_level.srt",
                   "_sentence_level.vtt", "_sentence_level.ass",
                   "_transcript.txt")


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_scenario(folder: Path, out_dir: Path) -> dict:
    """Run one scenario; returns {expected-file-name: produced text}."""
    from wordalign.cli_v2 import _parse_args
    from wordalign.core.pipeline import PipelineRunner

    scenario = _load(folder / "scenario.json")
    engines = {name: _load(folder / f)
               for name, f in scenario.get("engines", {}).items()}
    audio = out_dir / "media.wav"
    audio.write_bytes(b"recorded-engines-need-no-audio")
    argv = [str(audio), "-o", str(out_dir), *scenario["args"]]
    if scenario.get("transcript"):
        argv += ["-t", str(folder / scenario["transcript"])]
    cfg = _parse_args(argv)
    for name in ("whisperx", "qwen", "vosk"):
        if name not in engines:
            setattr(cfg, f"use_{name}", False)

    patches = [
        patch.dict(os.environ, {"WORDALIGN_CACHE": str(out_dir / "cache")}),
        patch("wordalign.engines.whisperx_engine.run_whisperx",
              lambda *a, **k: (engines.get("whisperx", []), [])),
        patch("wordalign.engines.qwen_engine.run_qwen",
              lambda *a, **k: engines.get("qwen", [])),
        patch("wordalign.engines.vosk_engine.run_vosk_parallel",
              lambda *a, **k: engines.get("vosk", [])),
        patch("wordalign.engines.vosk_engine.VOSK_AVAILABLE", True),
        patch("wordalign.config.PipelineConfig.vosk_model_path",
              lambda self, language: "/recorded/vosk"),
        # No ffprobe dependency: the recordings define the timeline.
        patch("wordalign.core.pipeline.get_audio_duration", lambda p: 0.0),
    ]
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        result = PipelineRunner(cfg).run()
    if result.error:
        raise AssertionError(f"{folder.name}: pipeline error: {result.error}")
    produced = {}
    for suffix in OUTPUT_SUFFIXES:
        path = out_dir / f"media{suffix}"
        if path.exists():
            produced[f"expected{suffix}"] = path.read_text(encoding="utf-8-sig")
    return produced


class TestGoldenFiles(unittest.TestCase):

    def test_every_scenario_matches_its_golden_files(self):
        scenarios = sorted(p.parent for p in GOLDEN.glob("*/scenario.json"))
        self.assertTrue(scenarios, "no golden scenarios found")
        for folder in scenarios:
            with self.subTest(scenario=folder.name):
                out_dir = Path(tempfile.mkdtemp())
                self.addCleanup(shutil.rmtree, out_dir, True)
                produced = run_scenario(folder, out_dir)
                if REGEN:
                    for old in folder.glob("expected*"):
                        old.unlink()
                    for name, text in produced.items():
                        (folder / name).write_text(text, encoding="utf-8",
                                                   newline="\n")
                    continue
                expected = {p.name: p.read_text(encoding="utf-8")
                            for p in folder.glob("expected*")}
                self.assertTrue(expected, f"{folder.name}: no expected files "
                                "(run with WORDALIGN_REGEN_GOLDEN=1)")
                self.assertEqual(sorted(produced), sorted(expected),
                                 f"{folder.name}: different set of outputs")
                for name, want in expected.items():
                    got = produced[name].replace("\r\n", "\n")
                    if got != want:
                        diff = "".join(difflib.unified_diff(
                            want.splitlines(True), got.splitlines(True),
                            f"golden/{folder.name}/{name}", "produced"))
                        self.fail(f"{folder.name}/{name} differs from its "
                                  f"golden file:\n{diff}")


if __name__ == "__main__":
    unittest.main()
