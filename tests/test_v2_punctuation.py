"""Tests for punctuation/capitalization restoration (smart transcriber)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wordalign.qa.punctuation import (
    RuleBasedRestorer,
    LlamaCppRestorer,
    create_restorer,
    restore_segments,
    _strip_thinking,
    _wrap_text,
)


class TestRuleBasedRestorer(unittest.TestCase):

    def test_capitalizes_first_letter(self):
        r = RuleBasedRestorer()
        self.assertEqual(r.restore("hello world"), "Hello world.")

    def test_fixes_lowercase_i(self):
        r = RuleBasedRestorer()
        self.assertEqual(r.restore("i think i'm ready"),
                         "I think I'm ready.")

    def test_capitalizes_after_period(self):
        r = RuleBasedRestorer()
        self.assertEqual(r.restore("it was hard. it got easier"),
                         "It was hard. It got easier.")

    def test_no_double_period(self):
        r = RuleBasedRestorer()
        self.assertEqual(r.restore("hello."), "Hello.")

    def test_empty_text(self):
        r = RuleBasedRestorer()
        self.assertEqual(r.restore(""), "")


class TestWrap(unittest.TestCase):

    def test_short_cue_unchanged(self):
        self.assertEqual(_wrap_text("Short cue.", 42), "Short cue.")

    def test_sentence_boundary_wrap(self):
        out = _wrap_text("to be in charge at first. I was afraid of change.", 42)
        self.assertEqual(out, "to be in charge at first.\nI was afraid of change.")

    def test_word_wrap_long_sentence(self):
        out = _wrap_text(
            "I had a good mentor see the potential that a district manager "
            "pushed me out there.", 42)
        lines = out.split("\n")
        for line in lines:
            self.assertLessEqual(len(line), 42)
        self.assertGreater(len(lines), 1)

    def test_strip_thinking(self):
        text = "1. Hello there.\n[Start thinking]\nThe user said hello.\n[End thinking]\n2. How are you?"
        self.assertNotIn("thinking", _strip_thinking(text).lower())
        self.assertIn("Hello there.", _strip_thinking(text))
        self.assertIn("How are you?", _strip_thinking(text))


class TestLlamaCppRestorer(unittest.TestCase):

    def test_is_ready_missing_model(self):
        # Self-contained: a temp "engine" dir holding a fake llama-cli, so
        # the CLI lookup succeeds on any machine and the model check is what
        # fails. (is_ready only checks that the file exists; never runs it.)
        with tempfile.TemporaryDirectory() as tmp:
            cli = "llama-cli.exe" if os.name == "nt" else "llama-cli"
            open(os.path.join(tmp, cli), "w").close()
            r = LlamaCppRestorer(
                engine=tmp, model=os.path.join(tmp, "nonexistent.gguf"))
            ok, msg = r.is_ready()
        self.assertFalse(ok)
        self.assertIn("model not found", msg)
        self.assertNotIn("llama-cli", msg)

    def test_parse_rejects_commentary(self):
        r = LlamaCppRestorer(engine="x", model="y")
        cleaned = (
            "1. I got scared.\n"
            "2. Wait, I can't change words.\n"
            "3. [Note: cue 3 missing]\n"
            "4. Now I lead the team.\n"
        )
        parsed = r._parse_cues(cleaned, 4)
        # Commentary lines (2, 3) must be rejected → not all 4 → None
        self.assertIsNone(parsed)

    def test_parse_accepts_clean(self):
        r = LlamaCppRestorer(engine="x", model="y")
        cleaned = "1. I got scared.\n2. Now I lead the team."
        parsed = r._parse_cues(cleaned, 2)
        self.assertEqual(parsed, ["I got scared.", "Now I lead the team."])


class TestFactoryAndSegments(unittest.TestCase):

    def test_create_restorer_rules_without_llm(self):
        from wordalign.config import PipelineConfig
        cfg = PipelineConfig("test.mp4", punctuation=True)
        restorer = create_restorer(cfg)
        out = restorer(["i am ready"])
        self.assertEqual(out, ["I am ready."])

    def test_create_restorer_llm_when_configured(self):
        from wordalign.config import PipelineConfig
        cfg = PipelineConfig("test.mp4", punctuation=True,
                             llm_model="Z:/model.gguf",
                             llm_engine="D:/llama.cpp")
        restorer = create_restorer(cfg)
        # Falls back to rules when the model file doesn't exist
        out = restorer(["i am ready"])
        self.assertEqual(out, ["I am ready."])

    def test_restore_segments_in_place(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "i got scared"},
                    {"start": 1.0, "end": 2.0, "text": "i ran away"}]
        r = RuleBasedRestorer()
        restore_segments(segments, r.restore_cues)
        self.assertEqual(segments[0]["text"], "I got scared.")
        self.assertEqual(segments[1]["text"], "I ran away.")


if __name__ == "__main__":
    unittest.main()
