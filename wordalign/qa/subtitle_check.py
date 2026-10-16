"""Check finished subtitles against the layout and timing rules, and against
the transcript they came from.

The final SRT promises every word and every ``[tag]`` of the transcript, in
order, within the job's limits (lines, characters per line including spaces
and punctuation, 1-7 s on screen by default, at least 1 ms between cues, no
overlaps). :func:`check_subtitles` measures exactly that, so a run reports
what (if anything) still needs a look.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence

_TAG_RE = re.compile(r"\[[^\]]*\]")
_TIMECODE_RE = re.compile(r"\d{2}:\d{2}:\d{2}")
_STREAM_RE = re.compile(r"\[[^\]]*\]|[^\s\[\]]+")


def _keep_tag(tag: str) -> bool:
    inner = tag.strip()[1:-1].strip()
    return bool(inner) and not _TIMECODE_RE.match(inner) and "END OF AUDIO" not in inner.upper()


def transcript_stream(text: str) -> List[str]:
    """Words (lower case, no punctuation) and ``[tags]``, in order."""
    out: List[str] = []
    for tok in _STREAM_RE.findall(text or ""):
        if tok.startswith("["):
            if _keep_tag(tok):
                out.append(" ".join(tok.lower().split()))
        else:
            word = re.sub(r"[^\w']", "", tok.replace("’", "'")).lower()
            if word:
                out.append(word)
    return out


def check_subtitles(cues: Sequence, max_cpl: int, max_lines: int,
                    min_ms: int = 1000, max_ms: int = 7000, min_gap_ms: int = 1,
                    max_cps: float = 25.0, transcript: Optional[str] = None) -> Dict:
    """Rule check of *cues* (objects with ``start``/``end`` timedelta and
    ``content``, as the ``srt`` package makes them).

    Returns counts and the first few cue numbers for each problem, the
    reading-speed spread, and -- given the *transcript* -- the words and
    tags that are missing, extra or out of order.
    """
    problems: Dict[str, List] = {k: [] for k in (
        "too_many_lines", "line_too_long", "should_be_one_line", "too_short",
        "too_long", "overlap", "gap_under_min", "empty", "fast", "short_in_sync")}
    cps_values: List[float] = []
    prev_end = None
    for n, cue in enumerate(cues, 1):
        lines = cue.content.split("\n")
        flat = " ".join(line.strip() for line in lines).strip()
        dur_ms = (cue.end - cue.start).total_seconds() * 1000.0
        if not flat:
            problems["empty"].append(n)
        if len(lines) > max_lines:
            problems["too_many_lines"].append(n)
        if any(len(line) > max_cpl for line in lines):
            problems["line_too_long"].append(n)
        if len(lines) > 1 and len(flat) <= max_cpl:
            problems["should_be_one_line"].append(n)
        if dur_ms < min_ms - 0.5:
            # Short but running on to the next cue: the words are just short,
            # and a longer cue would have to start before them.
            nxt = cues[n] if n < len(cues) else None
            chained = nxt is not None and (nxt.start - cue.end).total_seconds() * 1000.0 \
                <= min_gap_ms + 0.5
            problems["short_in_sync" if chained else "too_short"].append(n)
        if dur_ms > max_ms + 0.5:
            problems["too_long"].append(n)
        if prev_end is not None:
            gap_ms = (cue.start - prev_end).total_seconds() * 1000.0
            if gap_ms < 0:
                problems["overlap"].append(n)
            elif gap_ms < min_gap_ms - 0.5:
                problems["gap_under_min"].append(n)
        tag_only = bool(re.fullmatch(r"\s*(\[[^\]]*\]\s*)+", flat))
        if not tag_only and dur_ms > 0:
            cps = len(flat) * 1000.0 / dur_ms
            cps_values.append(cps)
            if cps > max_cps:
                problems["fast"].append(n)
        prev_end = cue.end
    report: Dict = {"cues": len(cues),
                    "problems": {k: {"count": len(v), "cues": v[:20]} for k, v in problems.items()}}
    if cps_values:
        ordered = sorted(cps_values)
        report["cps"] = {"median": round(ordered[len(ordered) // 2], 1),
                         "max": round(ordered[-1], 1),
                         "share_over_20": round(sum(c > 20 for c in cps_values) / len(cps_values), 4),
                         "share_over_max": round(len(problems["fast"]) / len(cps_values), 4)}
    if transcript is not None:
        ref = transcript_stream(transcript)
        hyp = transcript_stream(" ".join(c.content for c in cues))
        sm = SequenceMatcher(None, ref, hyp, autojunk=False)
        missing, extra = [], []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op in ("delete", "replace"):
                missing.extend(ref[i1:i2])
            if op in ("insert", "replace"):
                extra.extend(hyp[j1:j2])
        report["transcript"] = {
            "words": sum(1 for t in ref if not t.startswith("[")),
            "tags": sum(1 for t in ref if t.startswith("[")),
            "missing": len(missing), "extra": len(extra),
            "missing_sample": missing[:12], "extra_sample": extra[:12]}
    return report


def hard_failures(report: Dict) -> Dict[str, int]:
    """The problems that break the promise (everything except reading speed
    and short cues that run on to the next one)."""
    out = {k: v["count"] for k, v in report["problems"].items()
           if v["count"] and k not in ("fast", "short_in_sync")}
    tr = report.get("transcript")
    if tr and tr["missing"]:
        out["missing_from_transcript"] = tr["missing"]
    if tr and tr["extra"]:
        out["not_in_transcript"] = tr["extra"]
    return out


def summary_line(report: Dict) -> str:
    parts = [f"{report['cues']} cues"]
    tr = report.get("transcript")
    if tr:
        parts.append(f"all {tr['words']} words and {tr['tags']} tags present" if not tr["missing"]
                     else f"{tr['missing']} word(s)/tag(s) of the transcript missing")
    fails = hard_failures(report)
    parts.append("all rules met" if not fails else
                 "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in fails.items()))
    short = report["problems"]["short_in_sync"]["count"]
    if short:
        parts.append(f"{short} short cue(s) kept in sync with their words")
    cps = report.get("cps")
    if cps:
        parts.append(f"reading speed median {cps['median']} cps, "
                     f"{cps['share_over_20'] * 100:.1f}% over 20")
    return ", ".join(parts)
