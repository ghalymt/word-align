"""LLM punctuation restore: batching, word preservation, job layout."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.config import PipelineConfig
from wordalign.core.pipeline import PipelineRunner
from wordalign.qa.punctuation import LlamaCppRestorer, restore_segments
from wordalign.segment import get_config


class FakeLlamaCli:
    """Stands in for subprocess.run(llama-cli ...).

    Echoes the prompt's cue block back line by line, the way a model that
    keeps the input's line structure would: numbered lines are capitalised
    and get a "!" (so tests can tell LLM output from the rules' "."),
    unnumbered lines are repeated as-is.
    """

    def __init__(self, reword=None):
        self.prompts = []
        self.reword = reword or {}

    def __call__(self, cmd, **kwargs):
        prompt = cmd[cmd.index("-p") + 1]
        self.prompts.append(prompt)
        body = prompt.split("\n\n", 1)[1]
        out = []
        for line in body.splitlines():
            num, sep, text = line.partition(". ")
            if sep and num.isdigit():
                for old, new in self.reword.items():
                    text = text.replace(old, new)
                out.append(f"{num}. {text[:1].upper()}{text[1:]}!")
            else:
                out.append(line)
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(out),
                                           stderr="")


class TestLlamaCppRestoreCues(unittest.TestCase):

    def _restore(self, cues, fake, **kwargs):
        restorer = LlamaCppRestorer(engine="x", model="m.gguf", **kwargs)
        with patch.object(LlamaCppRestorer, "is_ready",
                          return_value=(True, "ready")), \
                patch.object(LlamaCppRestorer, "cli_path",
                             return_value="llama-cli"), \
                patch("wordalign.qa.punctuation.subprocess.run", fake):
            return restorer.restore_cues(cues)

    def test_two_line_cue_keeps_every_word(self):
        # Regression: "1. hello there\nhow are you" -- the parser kept only
        # the numbered line, silently dropping the cue's second line.
        out = self._restore(["hello there\nhow are you"], FakeLlamaCli())
        self.assertEqual(" ".join(out[0].split()), "Hello there how are you!")

    def test_cues_are_restored_in_batches(self):
        fake = FakeLlamaCli()
        cues = [f"cue number {i}" for i in range(95)]
        out = self._restore(cues, fake, batch_size=40)
        self.assertEqual(len(fake.prompts), 3)
        for prompt in fake.prompts:
            numbered = [l for l in prompt.splitlines() if l[:1].isdigit()]
            self.assertLessEqual(len(numbered), 40)
        self.assertEqual(len(out), 95)
        self.assertTrue(all(o.endswith("!") for o in out))
        self.assertEqual(out[94], "Cue number 94!")

    def test_reworded_cue_falls_back_to_rules(self):
        fake = FakeLlamaCli(reword={"gonna": "going to"})
        out = self._restore(["we are gonna win", "i think so"], fake)
        self.assertEqual(out[0], "We are gonna win.")   # rules, words kept
        self.assertEqual(out[1], "I think so!")         # LLM

    def test_cue_starting_like_commentary_is_still_restored(self):
        # The old commentary filter rejected any cue starting "I think",
        # "Wait", "Note"..., failing the whole batch over to rules.
        out = self._restore(["wait what", "note the date", "fine"],
                            FakeLlamaCli())
        self.assertEqual(out, ["Wait what!", "Note the date!", "Fine!"])

    def test_commentary_instead_of_a_cue_falls_back_to_rules(self):
        fake = FakeLlamaCli(reword={"we lost": "Wait, I can't change words"})
        out = self._restore(["we lost", "we won"], fake)
        self.assertEqual(out, ["We lost.", "We won!"])


class TestRestoreSegmentsLayout(unittest.TestCase):

    def test_layout_is_applied_to_flattened_text(self):
        segments = [{"text": "hello there\nhow are you"}]
        restore_segments(segments, lambda cues: [c.upper() for c in cues],
                         layout=lambda text: f"<{text}>")
        self.assertEqual(segments[0]["text"], "<HELLO THERE HOW ARE YOU>")


class TestPipelinePunctuationStage(unittest.TestCase):

    def _run(self, cfg_kwargs, primary_text):
        words, t = [], 0.0
        for w in primary_text.split():
            words.append({"word": w, "start": t, "end": t + 0.3, "conf": 0.9})
            t += 0.4
        with tempfile.TemporaryDirectory() as d, patch.dict(
                os.environ, {"WORDALIGN_CACHE": os.path.join(d, "c")}):
            os.environ.pop("WORDALIGN_LLM_MODEL", None)
            audio = Path(d) / "audio.wav"
            audio.write_bytes(b"not-real-audio")
            cfg = PipelineConfig(audio_path=str(audio), output_dir=d,
                                 use_vosk=False, use_qwen=False,
                                 use_whisperx=True, use_mfa=False,
                                 **cfg_kwargs)
            runner = PipelineRunner(cfg)
            runner._run_primary_transcript = lambda c, l: (
                words, " ".join(w["word"] for w in words), [], "whisperx")
            with contextlib.redirect_stdout(io.StringIO()):
                return runner.run()

    def test_result_segments_include_restored_punctuation(self):
        # Regression: result.segments was captured before the punctuation
        # stage, so the GUI/API saw the unpunctuated text.
        result = self._run({"punctuation": True}, "hello there everyone")
        self.assertEqual(result.segments[0].text, "Hello there everyone.")

    def test_srt_is_validated_against_the_jobs_layout(self):
        # Regression: validation ran after the module layout was reset to
        # the defaults (32 CPL / 3000 ms), not the job's 42 / 7000.
        seen = []
        with patch("wordalign.core.pipeline.validate_srt_output",
                   lambda entries, name: seen.append(get_config())):
            self._run({}, "hello there everyone")
        self.assertEqual(seen[0]["max_cpl"], 42)
        self.assertEqual(seen[0]["max_duration_ms"], 7000)


if __name__ == "__main__":
    unittest.main()
