"""Logic tests for the parts that need no GPU, audio, or model weights.

Run with:  python -m pytest tests/ -q     (or plain: python tests/test_pipeline.py)
"""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wordalign.align import interpolate_timestamps, match_timestamps
from wordalign.document import (group_into_paragraphs, group_into_sentences,
                                write_txt)
from wordalign.ensemble import build_consensus, consensus_to_text
from wordalign.segment import (parse_human_transcript_to_srt_segments,
                               run_iterative_merging, validate_srt_output)
from wordalign.utils import (has_terminal_punctuation, normalize_word,
                             strip_tags)


def W(word, t, conf=0.9, dur=0.28):
    return {"word": word, "start": t, "end": t + dur, "conf": conf}


class Cue:
    """Minimal stand-in for srt.Subtitle -- the validator reads only these."""

    def __init__(self, start, end, content):
        self.start = timedelta(seconds=start)
        self.end = timedelta(seconds=end)
        self.content = content


def test_ensemble_corrects_backbone_error():
    """Two independent engines agreeing should overturn a backbone mistake."""
    wx = [W("The", 0.0), W("speed", 0.35), W("of", 0.70),
          W("night", 1.05, 0.55), W("is", 1.40), W("constant.", 1.75)]
    pk = [W("The", 0.02), W("speed", 0.36), W("of", 0.71),
          W("light", 1.06, 0.95), W("is", 1.41), W("constant", 1.76)]
    vk = [W("the", 0.01), W("speed", 0.34), W("off", 0.69, 0.4),
          W("light", 1.04, 0.8), W("is", 1.42), W("constant", 1.74)]
    cq = [{"word": w, "start": None, "end": None, "conf": 0.95}
          for w in "The speed of light is constant.".split()]
    cons = build_consensus({"whisperx": wx, "parakeet": pk,
                            "vosk": vk, "canary": cq})
    text = consensus_to_text(cons)
    assert "light" in text and "night" not in text
    # Backbone must be WhisperX -> its timestamps survive verbatim.
    assert cons[0]["start"] == 0.0
    assert all(0 < c["conf"] <= 1 for c in cons)


def test_ensemble_margin_protects_backbone():
    """A single low-confidence dissenter must not overturn the backbone."""
    cons = build_consensus({"whisperx": [W("Cairo", 0.0, 0.9)],
                            "vosk": [W("Kyoto", 0.02, 0.5)]})
    assert cons[0]["word"] == "Cairo"


def test_ensemble_backbone_is_whisperx_not_highest_prior():
    """Regression: priors weight votes; they must not pick the backbone."""
    cons = build_consensus({"parakeet": [W("alpha", 5.0)],
                            "whisperx": [W("alpha", 1.0)]})
    assert cons[0]["start"] == 1.0


def test_waterfall_is_fill_only():
    """Later engines fill gaps; they never overwrite earlier matches."""
    human = "Hello world this is a timing test".split()
    aligned = [{"word": w} for w in human]
    norm = [normalize_word(w) for w in human]
    match_timestamps(aligned, [W("Hello", 0.0), W("world", 0.4),
                               W("this", 0.8)], "A", norm)
    match_timestamps(aligned, [W("a", 1.6), W("timing", 2.0),
                               W("test", 2.4)], "B", norm)
    assert aligned[0]["source"] == "A" and aligned[5]["source"] == "B"
    assert aligned[3].get("start") is None
    before = aligned[0]["start"]
    match_timestamps(aligned, [W("Hello", 9.0)], "Late", norm)
    assert aligned[0]["start"] == before and aligned[0]["source"] == "A"


def test_interpolation_stays_monotonic():
    human = "one two three four five".split()
    aligned = [{"word": w} for w in human]
    norm = [normalize_word(w) for w in human]
    match_timestamps(aligned, [W("one", 0.0), W("five", 2.0)], "A", norm)
    interpolate_timestamps(aligned)
    starts = [w["start"] for w in aligned]
    assert starts == sorted(starts), starts
    assert aligned[2]["source"] == "Interpolated"


def test_segmentation_respects_cpl():
    transcript = ("We started the migration in March.\n"
                  "It went well.\n"
                  "Then we hit the database issue,\n"
                  "which took two weeks to resolve.\n")
    words = transcript.replace("\n", " ").split()
    t, aligned = 0.0, []
    for w in words:
        aligned.append({"word": w, "start": t, "end": t + 0.30,
                        "matched": True})
        t += 0.36
    segs = parse_human_transcript_to_srt_segments(transcript, aligned)
    assert len(segs) == 4
    final = run_iterative_merging(segs)
    assert final
    for s in final:
        for line in s["text"].split("\n"):
            assert len(line) <= 32, f"CPL violated: {line!r}"


def test_tag_stripping_regex():
    """Regression: the tag regex must strip [tags], not eat real words."""
    assert strip_tags("Hello [laughs] world").strip() == "Hello  world".strip()
    assert "laughs" not in strip_tags("Hello [laughs] world")
    assert strip_tags("no tags here") == "no tags here"


def test_document_paragraphing_and_export(tmp_path=None):
    import tempfile
    words = [W("Hello", 0.0), W("there.", 0.4)]
    # 4s silence -> must start a new paragraph
    words += [W("Second", 5.0), W("paragraph.", 5.4)]
    sentences = group_into_sentences(words)
    assert len(sentences) == 2
    paras = group_into_paragraphs(sentences)
    assert len(paras) == 2, paras
    out = Path(tempfile.mkdtemp()) / "t.txt"
    write_txt(str(out), paras, timestamps=True)
    body = out.read_text()
    assert "[00:00:00]" in body and "[00:00:05]" in body


def test_validator_measures_cpl_per_line():
    """Regression: CPL is per line, not per block.

    Phase 3 balances long blocks across two lines. Measuring len(content)
    counted both lines plus the newline, so every correctly balanced cue
    was reported as a CPL violation.
    """
    balanced = "x" * 30 + "\n" + "y" * 30      # 61 chars total, 30 per line
    issues = validate_srt_output([Cue(0.0, 2.0, balanced),
                                  Cue(2.5, 4.5, balanced)], "cpl")
    assert issues["cpl_high"] == 0, issues


def test_validator_checks_the_final_entry():
    """Regression: per-entry checks skipped the last cue.

    The loop ran to len(entries) - 1 because it needed a successor for the
    overlap/gap checks, which silently excluded the final cue from the
    duration and CPL checks too.
    """
    legal = Cue(0.0, 1.0, "short line")
    too_long = Cue(2.0, 3.0, "z" * 40)         # single line, over MAX_CPL
    too_slow = Cue(4.0, 12.0, "short line")    # 8s, over MAX_DURATION_MS
    assert validate_srt_output([legal, too_long], "last-cpl")["cpl_high"] == 1
    assert validate_srt_output([legal, too_slow],
                               "last-duration")["duration_long"] == 1


def test_untimed_engine_reinforces_but_cannot_correct():
    """Characterization: untimed engines defend the backbone, never fix it.

    difflib matches on *equal* tokens, so a lexical vote always lands on
    the backbone's own token. This pins down two consequences that are easy
    to regress on and easy to misread from the module docstring.
    """
    # 1. Canary disagreeing with the backbone changes nothing: there is no
    #    matching block, so no vote is cast at all.
    wx = [W("teh", 0.0, conf=0.5)]
    cq = [{"word": "The", "start": None, "end": None, "conf": 0.95}]
    assert build_consensus({"whisperx": wx, "canary": cq})[0]["word"] == "teh"

    # 2. Canary agreeing with the backbone is decisive: without its vote a
    #    timed challenger takes the slot, with it the backbone holds.
    wx2 = [W("Cairo", 0.0, conf=0.6)]
    pk2 = [W("Kyoto", 0.02, conf=0.95)]
    cq2 = [{"word": "Cairo", "start": None, "end": None, "conf": 0.95}]
    assert build_consensus({"whisperx": wx2,
                            "parakeet": pk2})[0]["word"] == "Kyoto"
    assert build_consensus({"whisperx": wx2, "parakeet": pk2,
                            "canary": cq2})[0]["word"] == "Cairo"


def test_terminal_punctuation_survives_closing_quotes():
    """Regression: the trailing class ["\\"] collapsed to one straight quote.

    Curly-quoted and guillemet-quoted dialogue therefore never registered
    as a sentence ending, so phase 1 merged straight through it.
    """
    assert has_terminal_punctuation('He said "Go."')
    assert has_terminal_punctuation("He said “Go.”")
    assert has_terminal_punctuation("Она сказала «Иди.»")
    assert has_terminal_punctuation("It ended.")
    assert not has_terminal_punctuation("and it trailed off")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    print("\n" + ("ALL TESTS PASSED" if not failures
                  else f"{failures} TEST(S) FAILED"))
    sys.exit(1 if failures else 0)
