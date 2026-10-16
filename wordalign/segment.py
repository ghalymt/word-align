"""Sentence segmentation: transcript lines -> broadcast-quality subtitle blocks.

Three phases run inside an iterative sweep with a growing character budget
(1 up to lines x per-line CPL), which lets small merges happen early and
prevents greedy over-merging:
    phase 1  merge lines up to sentence boundaries (capitalization-aware)
    phase 2  merge short fragments under CPL/duration limits
    phase 3  balance each block into visually even lines
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .config import ITERATION_START, MAX_CPL, MAX_DURATION_MS, MIN_LINE_RATIO
from .linebreak import break_cost
from .utils import (TIMECODE_TAG_RE, count_visible_characters,
                    ends_with_nonterminal_abbreviation,
                    has_terminal_punctuation, ms_to_time, normalize_word,
                    time_to_ms)

# ---------------------------------------------------------------------------
# Runtime layout configuration (set once by the CLI before segmenting).
#
# The subtitle standard is expressed as two knobs -- characters *per line* and
# number of *lines*. A cue may hold up to `_max_lines` lines of `_max_cpl`
# characters, so the merge sweep grows its budget to `_max_cpl * _max_lines`
# (e.g. 42x2 = 84 for regular video, 32x2 = 64 for vertical/social) while each
# individual line is still capped at `_max_cpl`.
# ---------------------------------------------------------------------------
_max_cpl = MAX_CPL                    # characters per line
_max_lines = 2                        # lines per cue
_max_duration_ms = MAX_DURATION_MS    # max on-screen duration per cue
_min_cue_ms = 700                     # minimum on-screen duration per cue
_language = "en"                      # which line-break rules apply

# Second-attempt ratio for the balancer: prefer a slightly uneven split over
# leaving a line too long, but never force an outright ugly break.
RELAXED_LINE_RATIO = 0.20


def set_layout(max_cpl: Optional[int] = None, max_lines: Optional[int] = None,
               max_duration_ms: Optional[int] = None) -> None:
    """Configure the per-line / per-cue limits used throughout segmentation."""
    global _max_cpl, _max_lines, _max_duration_ms
    if max_cpl is not None:
        _max_cpl = int(max_cpl)
    if max_lines is not None:
        _max_lines = int(max_lines)
    if max_duration_ms is not None:
        _max_duration_ms = int(max_duration_ms)


def set_config(config) -> None:
    """Set segmentation parameters from a SegmentationConfig dataclass.

    This is the v2.0 entry point. It replaces multiple set_layout() calls
    with a single config-driven setup, making multi-job operation safe.
    """
    global _max_cpl, _max_lines, _max_duration_ms, _min_cue_ms
    _max_cpl = int(config.max_cpl)
    _max_lines = int(config.max_lines)
    _max_duration_ms = int(config.max_duration_ms)
    _min_cue_ms = int(config.min_cue_ms)


def set_language(language: Optional[str]) -> str:
    """Choose the line-break rules (English keeps GoTranscript's); returns the old value."""
    global _language
    previous = _language
    _language = (language or "en").lower()
    return previous


def get_config() -> dict:
    """Return current segmentation parameters as a dict.

    Useful for testing and for saving job state.
    """
    return {
        "max_cpl": _max_cpl,
        "max_lines": _max_lines,
        "max_duration_ms": _max_duration_ms,
        "min_cue_ms": _min_cue_ms,
    }


def _cue_char_budget() -> int:
    """Total characters a cue may hold: per-line limit times line count."""
    return _max_cpl * _max_lines


# The sweep's current character budget (module-level by design: the phase
# functions consult it every call while the sweep drives it from 1 upward).
_current_max_chars = _max_cpl


def set_max_chars(value: int) -> None:
    global _current_max_chars
    _current_max_chars = value


_ISOLATED_TAG_RE = re.compile(r"^\s*\[[^\]]+\]\s*$")


def is_isolated_tag(text: str) -> bool:
    """A cue that is nothing but a single ``[bracketed]`` tag (``[music]``,
    ``[laughter]`` …) should stay on its own line, never merged into dialogue."""
    return bool(_ISOLATED_TAG_RE.match(text or ""))


_TAGS_ONLY_RE = re.compile(r"^\s*(\[[^\]]*\]\s*)+$")


def _is_tag(seg: Dict) -> bool:
    """A sound-tag cue (``[laughter]``, ``[laughter] [applause]``)."""
    return bool(seg.get("tag")) or is_isolated_tag(seg.get("text", ""))


def _keepable_tag(line: str) -> bool:
    """A transcript tag line that belongs in the subtitles (not a timecode
    or an end-of-audio marker)."""
    inner = [t.strip() for t in re.findall(r"\[(.*?)\]", line)]
    return bool(inner) and all(t and not TIMECODE_TAG_RE.match(t)
                               and "END OF AUDIO" not in t.upper() for t in inner)


def is_protected_phrase(text: str, split_pos: int) -> bool:
    protected_phrases = ["a little bit"]
    lower_text = text.lower()
    for phrase in protected_phrases:
        start = 0
        while True:
            idx = lower_text.find(phrase, start)
            if idx == -1:
                break
            if idx < split_pos < idx + len(phrase):
                return True
            start = idx + 1
    return False


def evaluate_split(text: str, pos: int, total_len: int,
                   min_ratio: float = MIN_LINE_RATIO) -> Optional[dict]:
    if is_protected_phrase(text, pos):
        return None
    line1 = text[:pos].strip()
    line2 = text[pos:].strip()
    l1 = count_visible_characters(line1)
    l2 = count_visible_characters(line2)
    if l1 > _max_cpl or l2 > _max_cpl:
        return None
    if l2 < 2:
        return None
    if line2 and line2[0] in ",;:!?…。！？‥":
        return None
    if line1 and line1[-1] in "([{«":
        return None
    if l1 < (total_len * min_ratio) or l2 < (total_len * min_ratio):
        return None
    brk = break_cost(line1.split(), line2.split(), _language)
    return {"pos": pos, "score": abs(l1 - l2) + brk, "brk": brk, "l1": line1, "l2": line2}


# A merge is only worth it if the merged cue can break into lines somewhere
# that keeps words together (break_cost below this; BAD is 15).
CLEAN_BREAK_LIMIT = 10.0


def check_if_text_can_be_split_prettily(text: str) -> bool:
    text = text.replace("\n", " ").strip()
    total_len = count_visible_characters(text)
    if total_len <= _max_cpl:
        return True
    if total_len > _cue_char_budget() or _max_lines < 2:
        return False
    for i in range(1, len(text)):
        if text[i] != " ":
            continue
        c = evaluate_split(text, i, total_len)
        if c is not None and c["brk"] < CLEAN_BREAK_LIMIT:
            return True
    return any(_best_multiline_split(text, n) is not None
               for n in range(3, _max_lines + 1))


# Extra cost of a split whose shorter line is under MIN_LINE_RATIO (but at
# least RELAXED_LINE_RATIO) of the text: a slightly uneven pair is better
# than a break that splits a verb, a name or a number from its unit.
UNEVEN_PENALTY = 6.0


def _best_two_line_split(flat: str, total: int) -> Optional[dict]:
    best = None
    for i in range(1, len(flat)):
        if flat[i] != " ":
            continue
        c = evaluate_split(flat, i, total, RELAXED_LINE_RATIO)
        if c is None:
            continue
        shorter = min(count_visible_characters(c["l1"]), count_visible_characters(c["l2"]))
        if shorter < total * MIN_LINE_RATIO:
            c["score"] += UNEVEN_PENALTY
        if best is None or c["score"] < best["score"]:
            best = c
    return best


def find_word_boundary_near_limit(text: str, max_pos: int) -> int:
    pos = min(max_pos, len(text) - 1)
    while pos > 0:
        if text[pos] == " ":
            return pos + 1
        pos -= 1
    return max_pos


def balance_block_enhanced(text: str) -> str:
    """Balance a cue into two lines -- but never introduce a bad break.

    Strategy (ported from the manual-review-friendly balancer): keep it one
    line if it fits; keep an already-valid two-line split untouched; otherwise
    try a balanced (gold) split, then a relaxed (silver) one. If no clean split
    exists, leave the text as a single long line so the validator flags it for
    a human, rather than forcing an ugly mid-phrase break.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    # An existing, already-valid two-line split is left exactly as it is.
    if "\n" in stripped and _max_lines >= 2:
        lines = stripped.split("\n")
        if len(lines) == 2:
            l1 = count_visible_characters(lines[0])
            l2 = count_visible_characters(lines[1])
            ratio = l1 / (l1 + l2) if (l1 + l2) else 0
            if (l1 <= _max_cpl and l2 <= _max_cpl
                    and MIN_LINE_RATIO <= ratio <= 1 - MIN_LINE_RATIO):
                return stripped

    flat = " ".join(stripped.replace("\n", " ").split())
    total = count_visible_characters(flat)
    if total <= _max_cpl or _max_lines < 2:
        return flat   # one line allowed: never invent a second one

    best = _best_two_line_split(flat, total)
    if best is not None:
        return f"{best['l1']}\n{best['l2']}"

    # Two lines cannot hold it cleanly; use more if the layout allows them.
    for n_lines in range(3, _max_lines + 1):
        lines = _best_multiline_split(flat, n_lines)
        if lines is not None:
            return "\n".join(lines)

    return flat   # no clean split -> leave long, visible for manual review


# Upper bound on search steps per cue for 3+ lines. The search below only
# follows splits whose lines fit, so real cues finish in a few thousand
# steps; this guards against pathological input (hundreds of one-letter
# tokens), where the best split found so far is used.
_MAX_SPLIT_STEPS = 100_000

_NO_LINE_START = ",;:!?…。！？‥"
_NO_LINE_END = "([{«"


def _split_score(lines: List[str]) -> float:
    lengths = [count_visible_characters(line) for line in lines]
    score = float(max(lengths) - min(lengths))
    for prev, nxt in zip(lines, lines[1:]):
        score += break_cost(prev.split(), nxt.split(), _language)
    return score


def _best_multiline_split(text: str, n_lines: int) -> Optional[List[str]]:
    """Best split of *text* into exactly *n_lines* lines, each <= CPL.

    Scored like the two-line balancer: the spread between the longest and
    shortest line, with the same preferences for where a line may end
    (articles/prepositions start a line rather than end it, conjunctions
    are good line starts). Lines may not start with closing punctuation or
    end with an opening bracket. Returns None if no split fits.

    The search extends a line one word at a time and abandons it as soon as
    it is too long or the rest cannot fit on the remaining lines, so it
    only ever visits splits that can work -- enumerating every combination
    of cut points blindly gave up on five-line cues before reaching one.
    """
    words = text.split()
    if len(words) < n_lines:
        return None
    # Visible length of words[a:b] joined by spaces, in O(1).
    prefix = [0]
    for word in words:
        prefix.append(prefix[-1] + count_visible_characters(word))

    def width(a: int, b: int) -> int:
        return prefix[b] - prefix[a] + (b - a - 1)

    best: List = [None]           # [(score, lines)]
    steps = [0]

    def search(start: int, lines_left: int, cuts: List[int]) -> None:
        if lines_left == 1:
            last = width(start, len(words))
            if last > _max_cpl or last < 2 or words[start][0] in _NO_LINE_START:
                return
            bounds = [0, *cuts, len(words)]
            lines = [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:])]
            score = _split_score(lines)
            if best[0] is None or score < best[0][0]:
                best[0] = (score, lines)
            return
        if cuts and words[start][0] in _NO_LINE_START:
            return
        # Leave at least one word for each of the remaining lines.
        for end in range(start + 1, len(words) - lines_left + 2):
            steps[0] += 1
            if steps[0] > _MAX_SPLIT_STEPS:
                return
            line_width = width(start, end)
            if line_width > _max_cpl:
                break                     # longer lines only get longer
            if line_width < 2 or words[end - 1][-1] in _NO_LINE_END:
                continue
            rest_lines = lines_left - 1
            if width(end, len(words)) > rest_lines * _max_cpl + rest_lines - 1:
                continue                  # a longer line leaves less behind
            search(end, rest_lines, cuts + [end])

    search(0, n_lines, [])
    return best[0][1] if best[0] else None


def is_eligible_sentence(sentence_text: str, start_time: str,
                         end_time: str) -> bool:
    if count_visible_characters(sentence_text) >= _current_max_chars:
        return False
    if not check_if_text_can_be_split_prettily(sentence_text):
        return False
    duration_ms = time_to_ms(end_time) - time_to_ms(start_time)
    if duration_ms >= _max_duration_ms:
        return False
    return True


def shift_dangling_words(segments: List[Dict]) -> List[Dict]:
    for i in range(len(segments) - 1):
        if segments[i].get("speaker") != segments[i + 1].get("speaker"):
            continue   # never hand a word to another speaker's cue
        if _is_tag(segments[i]) or _is_tag(segments[i + 1]):
            continue
        curr_text = segments[i]["text"].strip()
        next_text = segments[i + 1]["text"].strip()
        match = re.search(r"(.*,\s)([^,.\s!?]+)$", curr_text, re.DOTALL)
        if match:
            prefix = match.group(1).strip()
            dangler = match.group(2).strip()
            if not dangler or len(dangler) > 20:
                continue
            words_i = segments[i].get("words")
            if words_i is not None and segments[i + 1].get("words") is not None:
                # The word's timing moves with it.
                if len(words_i) < 2 or _word_tokens(dangler) != 1:
                    continue
                segments[i + 1]["words"] = [words_i[-1]] + segments[i + 1]["words"]
                segments[i]["words"] = words_i[:-1]
                _retime(segments[i])
                _retime(segments[i + 1])
            segments[i]["text"] = prefix
            segments[i + 1]["text"] = f"{dangler} {next_text}"
    return segments


def _token_is_word(tokens: List[str]) -> List[bool]:
    """Which whitespace tokens are words (not part of a ``[tag]``).

    Mirrors how the transcript is matched to aligned words: tags are
    removed, then the rest is split on whitespace.
    """
    flags, depth = [], 0
    for tok in tokens:
        outside = []
        for ch in tok:
            if ch == "[":
                depth += 1
            elif ch == "]" and depth:
                depth -= 1
            elif not depth:
                outside.append(ch)
        flags.append(bool("".join(outside).strip()))
    return flags


def _word_tokens(text: str) -> int:
    return sum(_token_is_word(text.split()))


def _retime(seg: Dict) -> None:
    """Set a segment's times from the first and last of its timed words."""
    timed = [w for w in seg.get("words") or []
             if w.get("start") is not None and w.get("end") is not None]
    if not timed:
        return
    seg["start"] = ms_to_time(int(timed[0]["start"] * 1000))
    seg["end"] = ms_to_time(int(timed[-1]["end"] * 1000))
    seg["start_source"] = timed[0].get("source")
    seg["end_source"] = timed[-1].get("source")


def _joined(first: Dict, last: Dict, text: str, group: List[Dict]) -> Dict:
    """One segment from the run *group* (first .. last), with *text*."""
    out = {"start": first["start"], "end": last["end"],
           "start_source": first.get("start_source"),
           "end_source": last.get("end_source"),
           "text": text, "speaker": first.get("speaker"),
           "index": first.get("index")}
    if all(g.get("words") is not None for g in group):
        out["words"] = [w for g in group for w in g["words"]]
    return out


def _span_speaker(words: List[Dict]) -> Optional[str]:
    """Most common speaker among *words* (None when diarization is off)."""
    counts: Dict[str, int] = {}
    for w in words:
        if w.get("speaker"):
            counts[w["speaker"]] = counts.get(w["speaker"], 0) + 1
    return max(counts, key=counts.get) if counts else None


def _speaker_runs(words: List[Dict]) -> List[Tuple[int, int]]:
    """(start, end) index runs of *words* that share one speaker."""
    runs: List[Tuple[int, int]] = []
    start, current = 0, None
    for i, word in enumerate(words):
        speaker = word.get("speaker")      # unlabelled words join the run
        if speaker and current and speaker != current:
            runs.append((start, i))
            start = i
        if speaker:
            current = speaker
    if words:
        runs.append((start, len(words)))
    return runs


def parse_human_transcript_to_srt_segments(
        transcript_text: str, aligned_words: List[Dict],
        cue_boundaries: Optional[List[Tuple[int, int]]] = None,
        keep_tags: bool = False) -> List[Dict]:
    """Map each transcript line onto its aligned word span.

    If ``cue_boundaries`` is provided, those (start, end) word-index ranges
    are used to build segments directly — this preserves the segmentation
    of an input SRT transcript instead of letting the iterative merger
    collapse neighbouring phrases together.

    Each segment carries its aligned ``words``, so later passes can split
    it at a word boundary. With *keep_tags*, a line that is only ``[tags]``
    becomes a ``tag`` segment in its place (see :func:`place_tag_cues`);
    otherwise it is skipped.
    """
    print("Parsing transcript into segments...")
    segments: List[Dict] = []
    lines = [ln.strip() for ln in transcript_text.split("\n") if ln.strip()]
    cursor = 0

    # Fast path: respect the human's cue boundaries from an input SRT.
    if cue_boundaries:
        for cue_idx, (start, end) in enumerate(cue_boundaries):
            if end > len(aligned_words):
                # Out of range — transcript has more words than aligned.
                # Fall back to whatever range is still available.
                end = min(end, len(aligned_words))
            if start >= end:
                continue
            first = aligned_words[start]
            last = aligned_words[end - 1]
            t0 = first.get("start")
            t1 = last.get("end")
            if t0 is None or t1 is None:
                print(f"  [skip] cue {cue_idx + 1}: missing timestamps")
                continue
            segments.append({
                "index": cue_idx + 1,
                "start": ms_to_time(int(t0 * 1000)),
                "end": ms_to_time(int(t1 * 1000)),
                "start_source": first.get("source"),
                "end_source": last.get("source"),
                "text": " ".join(w["word"] for w in aligned_words[start:end]),
                "speaker": _span_speaker(aligned_words[start:end]),
                "words": list(aligned_words[start:end]),
            })
        print(f"[ok] Created {len(segments)} segments from SRT cue boundaries")
        return segments

    for line_idx, line in enumerate(lines):
        if re.fullmatch(r"\s*\[.*?\]\s*", line):
            if keep_tags and _keepable_tag(line):
                anchor = (segments[-1]["end"] if segments else ms_to_time(0))
                segments.append({"index": line_idx + 1, "start": anchor, "end": anchor,
                                 "text": " ".join(line.split()), "speaker": None,
                                 "tag": True, "words": [],
                                 "start_source": "tag", "end_source": "tag"})
            continue
        clean_line = re.sub(r"\[.*?\]", "", line).strip()
        words = clean_line.split()
        if not words:
            continue
        words_norm = [normalize_word(w) for w in words]
        pos = -1
        search_limit = min(cursor + 500,
                           len(aligned_words) - len(words_norm) + 1)
        for i in range(cursor, search_limit):
            if all(i + j < len(aligned_words)
                   and normalize_word(aligned_words[i + j]["word"]) == norm
                   for j, norm in enumerate(words_norm)):
                pos = i
                break
        if pos != -1:
            span = aligned_words[pos:pos + len(words)]
            cursor = pos + len(words)
            runs = _speaker_runs(span)
            if len(runs) > 1:
                # Diarization found a speaker change inside this line: one
                # segment per speaker, so no cue mixes two people.
                for a, b in runs:
                    first_word, last_word = span[a], span[b - 1]
                    if first_word.get("start") is None or last_word.get("end") is None:
                        continue
                    segments.append({
                        "index": line_idx + 1,
                        "start": ms_to_time(int(first_word["start"] * 1000)),
                        "end": ms_to_time(int(last_word["end"] * 1000)),
                        "start_source": first_word.get("source"),
                        "end_source": last_word.get("source"),
                        "text": " ".join(words[a:b]),
                        "speaker": _span_speaker(span[a:b]),
                        "words": list(span[a:b]),
                    })
                continue
            first_word = aligned_words[pos]
            last_word = aligned_words[pos + len(words) - 1]
            start = first_word.get("start")
            end = last_word.get("end")
            if start is not None and end is not None:
                segments.append({
                    "index": line_idx + 1,
                    "start": ms_to_time(int(start * 1000)),
                    "end": ms_to_time(int(end * 1000)),
                    # Which engine produced the boundary timestamps -- carried
                    # through the merge phases so overlap resolution can avoid
                    # touching Vosk times (see resolve_overlaps).
                    "start_source": first_word.get("source"),
                    "end_source": last_word.get("source"),
                    "text": line,
                    "speaker": _span_speaker(span),
                    "words": list(span),
                })
            else:
                print(f"  [skip] line {line_idx + 1}: missing timestamps")
        else:
            print(f"  [skip] line {line_idx + 1}: no word match found")
    print(f"[ok] Created {len(segments)} initial segments from transcript")
    return segments


def process_phase1(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    merged: List[Dict] = []
    i = 0
    while i < len(segments):
        if _is_tag(segments[i]):
            merged.append(segments[i])
            i += 1
            continue
        if has_terminal_punctuation(segments[i]["text"]):
            merged.append(segments[i])
            i += 1
            continue
        group = [segments[i]]
        parts = [segments[i]["text"]]
        start = segments[i]["start"]
        j = i + 1
        boundary_found = False
        while j < len(segments):
            next_txt = segments[j]["text"]
            if _is_tag(segments[j]):        # don't pull a tag into dialogue
                break
            if segments[j].get("speaker") != segments[i].get("speaker"):
                break                       # never merge two speakers
            prev_txt = segments[j - 1]["text"]
            if next_txt and next_txt[0].isupper() and next_txt[0] != "I":
                # A capitalised next word signals a new sentence -- unless the
                # previous text merely trailed off on a comma/colon/dash, or
                # ended on an abbreviation like "St." before a proper noun.
                if (not prev_txt.strip().endswith((",", ":", ";", "-", "—"))
                        and not ends_with_nonterminal_abbreviation(prev_txt)):
                    merged.extend(group)
                    i = j
                    boundary_found = True
                    break
            pot_text = " ".join(parts + [segments[j]["text"]])
            if count_visible_characters(pot_text) >= _current_max_chars:
                break
            if not check_if_text_can_be_split_prettily(pot_text):
                break
            if time_to_ms(segments[j]["end"]) - time_to_ms(start) >= _max_duration_ms:
                break                       # would stay on screen too long
            group.append(segments[j])
            parts.append(segments[j]["text"])
            if has_terminal_punctuation(segments[j]["text"]):
                merged.append(_joined(segments[i], segments[j], " ".join(parts), group))
                i = j + 1
                boundary_found = True
                break
            j += 1
        if not boundary_found:
            full_text = " ".join(parts)
            if len(group) > 1 and is_eligible_sentence(
                    full_text, start, segments[j - 1]["end"]):
                merged.append(_joined(segments[i], segments[j - 1], full_text, group))
            else:
                merged.extend(group)
            i = j
    return merged


def process_phase2(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    merged: List[Dict] = []
    i = 0
    while i < len(segments):
        curr = segments[i]
        if _is_tag(curr):
            merged.append(curr)
            i += 1
            continue
        if has_terminal_punctuation(curr["text"]):
            merged.append(curr)
            i += 1
            continue
        acc = curr
        j = i + 1
        while j < len(segments):
            nxt = segments[j]
            if _is_tag(nxt):
                break
            if nxt.get("speaker") != acc.get("speaker"):
                break
            combined = acc["text"] + " " + nxt["text"]
            if has_terminal_punctuation(acc["text"]):
                break
            if count_visible_characters(combined) >= _current_max_chars:
                break
            if not check_if_text_can_be_split_prettily(combined):
                break
            if (time_to_ms(nxt["end"]) - time_to_ms(acc["start"])) >= _max_duration_ms:
                break
            acc = _joined(acc, nxt, combined, [acc, nxt])
            if has_terminal_punctuation(acc["text"]):
                j += 1
                break
            j += 1
        merged.append(acc)
        i = j
    return merged


def process_phase3(segments: List[Dict]) -> List[Dict]:
    return [dict(s, index=s.get("index", i + 1), text=balance_block_enhanced(s["text"]))
            for i, s in enumerate(segments)]


def run_iterative_merging(segments: List[Dict]) -> List[Dict]:
    """Drive the phase sweep with the growing character budget."""
    print("\n" + "=" * 60 + "\nAPPLYING ITERATIVE MERGING & BALANCING")
    for i in range(ITERATION_START, _cue_char_budget() + 1):
        set_max_chars(i)
        if i % 10 == 0:
            print(f"   ... iteration limit: {i}")
        segments = process_phase1(segments)
        segments = process_phase2(segments)
        segments = shift_dangling_words(segments)
        segments = process_phase3(segments)
    print(f"[ok] Iterative processing complete: {len(segments)} final segments")
    return segments


# Minimum gap (ms) left between two cues so they never share an exact boundary.
OVERLAP_GAP_MS = 1


def _resolve_overlaps_once(segments: List[Dict]) -> int:
    """One forward pass of overlap removal; returns how many cues it adjusted.

    Priority rules, tuned to how the engines actually behave:
      1. Never move a Vosk timestamp -- Vosk is the timing authority (Whisper
         is strong on words, weak on times).
      2. Prefer preserving *start* times; they are empirically the most
         accurate anchor. So resolve by trimming the earlier cue's END.
      3. Only when the earlier END is Vosk (must be kept) and the later START
         is not, push the later START forward instead.

    When a boundary's source is unknown the code falls back to rule 2 (trim the
    end, keep the start), which is already safe. The word-level SRT is never
    touched -- this adjusts sentence-cue display times only.
    """
    fixed = 0
    for i in range(len(segments) - 1):
        cur, nxt = segments[i], segments[i + 1]
        cur_start = time_to_ms(cur["start"])
        cur_end = time_to_ms(cur["end"])
        nxt_start = time_to_ms(nxt["start"])
        nxt_end = time_to_ms(nxt["end"])
        if cur_end <= nxt_start:
            continue  # no overlap

        end_is_vosk = cur.get("end_source") == "Vosk"
        start_is_vosk = nxt.get("start_source") == "Vosk"

        if not end_is_vosk:
            # Trim the (less reliable) earlier end back behind the next start,
            # keeping the cue non-degenerate.
            new_end = max(cur_start + OVERLAP_GAP_MS, nxt_start - OVERLAP_GAP_MS)
            if new_end < cur_end:
                cur["end"] = ms_to_time(new_end)
                fixed += 1
        elif not start_is_vosk:
            # Earlier end is Vosk -> keep it; nudge the later start to just
            # after it, without crossing that cue's own end.
            new_start = min(cur_end + OVERLAP_GAP_MS, nxt_end - OVERLAP_GAP_MS)
            if new_start > nxt_start:
                nxt["start"] = ms_to_time(new_start)
                fixed += 1
        else:
            # Both boundaries are Vosk yet still overlap (an upstream artefact).
            # Honour rule 2: preserve the start, clip the earlier end.
            new_end = max(cur_start + OVERLAP_GAP_MS, nxt_start - OVERLAP_GAP_MS)
            if new_end < cur_end:
                cur["end"] = ms_to_time(new_end)
                fixed += 1
    return fixed


def resolve_overlaps(segments: List[Dict], max_passes: int = 6) -> List[Dict]:
    """Iterate overlap removal to a fixed point.

    A single forward pass can leave a residual overlap when resolving one pair
    nudges a boundary into its neighbour; repeat until nothing changes (or a
    safety cap) so overlaps reach zero.
    """
    total = 0
    for _ in range(max_passes):
        fixed = _resolve_overlaps_once(segments)
        total += fixed
        if fixed == 0:
            break
    if total:
        print(f"[ok] Resolved {total} cue overlap(s)")
    return segments


# Minimum on-screen time a cue may have. Sub-frame cues (e.g. from ASR words
# whose timestamps collapsed onto one instant) are extended or merged away.
MIN_CUE_MS = 700

# Reading speed, in characters per second (spaces and punctuation count).
# Cues faster than TARGET_CPS borrow free time around them; cues still faster
# than MAX_CPS are joined with a neighbour when that slows them down.
TARGET_CPS = 20.0
MAX_CPS = 25.0
# How far a cue may appear before its first word for reading speed (a
# cue's end lingers into the pause after it first).
MAX_LEAD_IN_MS = 500
# On-screen time for a [tag] cue when no sound event says how long it lasts.
TAG_DEFAULT_MS = 2000

# Splitting a cue that is too long: every extra cue costs PIECE_COST, more
# than the best break can earn back, so a cue is split into as few pieces as
# will fit and the breaks are chosen among those.
PIECE_COST = 30.0
SHORT_PIECE_CHARS = 12


def _ms(seg: Dict, key: str) -> int:
    return time_to_ms(seg[key])


def _dur(seg: Dict) -> int:
    return _ms(seg, "end") - _ms(seg, "start")


def _flat(text: str) -> str:
    return " ".join((text or "").split())


_LAYOUT_CACHE: Dict[tuple, Tuple[bool, float]] = {}


def _layout(text: str) -> Tuple[bool, float]:
    """(fits, break cost) of *text* laid out by :func:`balance_block_enhanced`:
    whether it fits ``_max_lines`` lines of ``_max_cpl``, and the
    :func:`break_cost` of its line breaks (0 for one line)."""
    flat = _flat(text)
    n = count_visible_characters(flat)
    if n <= _max_cpl:
        return True, 0.0
    if n > _cue_char_budget() or _max_lines < 2:
        return False, 0.0
    key = (flat, _max_cpl, _max_lines, _language)
    hit = _LAYOUT_CACHE.get(key)
    if hit is None:
        lines = balance_block_enhanced(flat).split("\n")
        fits = len(lines) <= _max_lines and all(
            count_visible_characters(line) <= _max_cpl for line in lines)
        cost = sum(break_cost(a.split(), b.split(), _language)
                   for a, b in zip(lines, lines[1:]))
        hit = (fits, cost)
        if len(_LAYOUT_CACHE) > 50_000:
            _LAYOUT_CACHE.clear()
        _LAYOUT_CACHE[key] = hit
    return hit


def _text_fits(text: str, clean: bool = False) -> bool:
    """Whether *text* lays out in at most ``_max_lines`` lines of ``_max_cpl``
    (*clean*: with line breaks that keep words together)."""
    fits, cost = _layout(text)
    return fits and (not clean or cost < CLEAN_BREAK_LIMIT)


def _cps(seg: Dict) -> float:
    dur = _dur(seg)
    chars = len(_flat(seg["text"]))
    return chars * 1000.0 / dur if dur > 0 else float("inf")


# -- splitting cues that are too long ------------------------------------------
def _needs_split(seg: Dict) -> bool:
    words = seg.get("words")
    if _is_tag(seg) or not words:
        return False
    if not _text_fits(seg["text"]):
        return True
    return len(words) > 1 and _dur(seg) > _max_duration_ms


def _split_segment(seg: Dict) -> List[Dict]:
    """Split *seg* into the fewest cues that fit the layout and duration.

    Among the splits with that many cues, the one with the best breaks wins
    (:func:`break_cost`, plus a bonus for a pause between the words). A
    single word is always a valid cue.
    """
    tokens = _flat(seg["text"]).split()
    is_word = _token_is_word(tokens)
    words = seg.get("words") or []
    if len(tokens) < 2 or sum(is_word) != len(words):
        return [seg]
    cum = [0]
    for flag in is_word:
        cum.append(cum[-1] + (1 if flag else 0))
    starts = [w.get("start") for w in words]
    ends = [w.get("end") for w in words]

    def span_ms(wa: int, wb: int) -> int:
        s = next((x for x in starts[wa:wb] if x is not None), None)
        e = next((x for x in reversed(ends[wa:wb]) if x is not None), None)
        return int((e - s) * 1000) if s is not None and e is not None else 0

    def cut_cost(a: int) -> float:
        cost = break_cost(tokens[max(0, a - 3):a], tokens[a:a + 3], _language)
        before, after = cum[a] - 1, cum[a]
        if 0 <= before < len(words) and after < len(words) \
                and ends[before] is not None and starts[after] is not None:
            pause = (starts[after] - ends[before]) * 1000.0
            cost -= max(0.0, min(12.0, pause / 50.0))
        return cost

    n = len(tokens)
    lens = [count_visible_characters(t) for t in tokens]
    budget = _cue_char_budget()
    inf = float("inf")
    best = [inf] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for b in range(1, n + 1):
        width = -1
        for a in range(b - 1, -1, -1):
            width += lens[a] + 1
            if width > budget and b - a > 1:
                break
            if best[a] == inf:
                continue
            wa, wb = cum[a], cum[b]
            if wb == wa:
                continue                      # a cue needs a word
            span = span_ms(wa, wb)
            fits, layout_cost = _layout(" ".join(tokens[a:b]))
            if wb - wa > 1 and (span > _max_duration_ms or not fits):
                continue
            cost = best[a] + PIECE_COST + max(0.0, layout_cost)
            if width < SHORT_PIECE_CHARS:
                cost += SHORT_PIECE_CHARS - width
            if span < _min_cue_ms:
                cost += 4.0
            if a > 0:
                cost += cut_cost(a)
            if cost < best[b]:
                best[b], back[b] = cost, a
    if best[n] == inf:
        return [seg]
    bounds = []
    b = n
    while b > 0:
        bounds.append((back[b], b))
        b = back[b]
    pieces = []
    for a, b in reversed(bounds):
        piece = dict(seg, text=balance_block_enhanced(" ".join(tokens[a:b])),
                     words=words[cum[a]:cum[b]])
        _retime(piece)
        pieces.append(piece)
    return pieces


def split_long_segments(segments: List[Dict]) -> List[Dict]:
    """Split cues that cannot fit ``_max_lines`` x ``_max_cpl`` or stay on
    screen longer than ``_max_duration_ms`` -- a long transcript line, say --
    at their best break points. Needs each segment's ``words``."""
    out: List[Dict] = []
    split = 0
    for seg in segments:
        if _needs_split(seg):
            pieces = _split_segment(seg)
            split += len(pieces) > 1
            out.extend(pieces)
        else:
            out.append(seg)
    if split:
        print(f"[ok] Split {split} over-long cue(s) at their best break points")
    return out


# -- the editor's [tags] ---------------------------------------------------------
def _tag_labels(text: str) -> List[str]:
    return [t.strip().lower() for t in re.findall(r"\[(.*?)\]", text or "")]


def _matching_event(events, labels: List[str], lo: int, hi: int) -> Optional[Tuple[int, int]]:
    """The sound event (ms) that best matches *labels* around [lo, hi]."""
    if not events:
        return None
    from .config import TAG_MAP
    wanted = set()
    for label in labels:
        wanted.update({label, label.rstrip("s"), TAG_MAP.get(label, label).lower()})
    best, best_overlap = None, 0
    for ev in events:
        if str(ev.get("tag", "")).lower() not in wanted:
            continue
        s, e = int(float(ev["start"]) * 1000), int(float(ev["end"]) * 1000)
        overlap = min(e, hi + 1000) - max(s, lo - 1000)
        if overlap > best_overlap:
            best, best_overlap = (s, e), overlap
    return best


def _join_text(a: str, b: str) -> str:
    return balance_block_enhanced(f"{_flat(a)} {_flat(b)}".strip())


def _fold_tag(out: List[Dict], segments: List[Dict], j: int, text: str,
              prefer_next: bool) -> bool:
    """Put *text* (a tag) into the cue before it (``out[-1]``) or after it
    (``segments[j]``) -- splitting that cue if it is full. False if neither
    neighbour can take it."""
    prev = out[-1] if out and not _is_tag(out[-1]) else None
    nxt = segments[j] if j < len(segments) and not _is_tag(segments[j]) else None
    order = [("next", nxt), ("prev", prev)] if prefer_next else [("prev", prev), ("next", nxt)]
    for side, cue in order:
        if cue is None:
            continue
        joined = f"{_flat(cue['text'])} {text}" if side == "prev" else f"{text} {_flat(cue['text'])}"
        if _text_fits(joined):
            new = dict(cue, text=balance_block_enhanced(joined))
            if side == "prev":
                out[-1] = new
            else:
                segments[j] = new
            return True
    for side, cue in order:
        if cue is None or cue.get("words") is None:
            continue
        joined = f"{_flat(cue['text'])} {text}" if side == "prev" else f"{text} {_flat(cue['text'])}"
        pieces = _split_segment(dict(cue, text=joined))
        if len(pieces) > 1:
            if side == "prev":
                out[-1:] = pieces
            else:
                segments[j:j + 1] = pieces
            return True
    return False


def place_tag_cues(segments: List[Dict], events=None,
                   min_ms: Optional[int] = None) -> List[Dict]:
    """Time the transcript's ``[tag]`` lines (``tag`` segments from
    :func:`parse_human_transcript_to_srt_segments`).

    A tag sits between the cue before it and the cue after it, where the
    editor put it. When that pause is long enough for a cue it gets one --
    timed by a matching sound event in *events* (``start``/``end`` seconds,
    ``tag`` label) if there is one; otherwise the tag is folded into the cue
    before (or after) it, which keeps every cue at least *min_ms* long and
    never overlaps speech.
    """
    if min_ms is None:
        min_ms = _min_cue_ms
    segments = list(segments)
    out: List[Dict] = []
    placed = folded = 0
    i = 0
    while i < len(segments):
        if not segments[i].get("tag"):
            out.append(segments[i])
            i += 1
            continue
        j = i
        while j < len(segments) and segments[j].get("tag"):
            j += 1
        group = segments[i:j]
        text = " ".join(_flat(g["text"]) for g in group)
        prev = out[-1] if out and not _is_tag(out[-1]) else None
        nxt = segments[j] if j < len(segments) else None
        lo = _ms(prev, "end") + OVERLAP_GAP_MS if prev else 0
        hi = (_ms(nxt, "start") - OVERLAP_GAP_MS) if nxt else lo + TAG_DEFAULT_MS
        event = _matching_event(events, _tag_labels(text), lo, hi)
        if hi - lo >= min_ms:
            if event:
                start = min(max(event[0], lo), hi - min_ms)
                end = min(max(event[1], start + min_ms), hi)
            else:
                start, end = lo, min(hi, lo + max(min_ms, TAG_DEFAULT_MS))
            end = min(end, start + _max_duration_ms)
            out.append({"index": group[0].get("index"), "start": ms_to_time(start),
                        "end": ms_to_time(end), "text": balance_block_enhanced(text),
                        "speaker": None, "tag": True, "words": [],
                        "start_source": "tag", "end_source": "tag"})
            placed += 1
        else:
            prefer_next = bool(event and prev is not None and nxt is not None
                               and event[0] >= _ms(prev, "end"))
            if not _fold_tag(out, segments, j, text, prefer_next):
                # No neighbour can take it: show it in whatever time there is
                # rather than lose the editor's tag.
                start = max(lo, 0)
                end = max(start + 1, hi)
                out.append({"index": group[0].get("index"), "start": ms_to_time(start),
                            "end": ms_to_time(end), "text": balance_block_enhanced(text),
                            "speaker": None, "tag": True, "words": [],
                            "start_source": "tag", "end_source": "tag"})
            folded += 1
        i = j
    if placed or folded:
        print(f"[ok] Transcript tags: {placed} on their own cue, {folded} folded into a "
              f"neighbouring cue")
    return out


# -- on-screen time ----------------------------------------------------------------
def _can_join(a: Optional[Dict], b: Optional[Dict]) -> bool:
    if a is None or b is None or _is_tag(a) or _is_tag(b):
        return False
    if a.get("dash_cue") or b.get("dash_cue") or a.get("speaker") != b.get("speaker"):
        return False
    if max(_ms(a, "end"), _ms(b, "end")) - _ms(a, "start") > _max_duration_ms:
        return False
    return _text_fits(f"{_flat(a['text'])} {_flat(b['text'])}", clean=True)


def _join_two(a: Dict, b: Dict) -> Dict:
    later = b if _ms(b, "end") >= _ms(a, "end") else a
    out = dict(a, text=_join_text(a["text"], b["text"]), end=later["end"],
               end_source=later.get("end_source"))
    if a.get("words") is not None and b.get("words") is not None:
        out["words"] = list(a["words"]) + list(b["words"])
    return out


def _grow(segs: List[Dict], i: int, need: int, max_lead_in: int) -> None:
    """Give cue *i* up to *need* more ms: its end first, into the pause after
    it, then its start (at most *max_lead_in* earlier), never past a
    neighbour or the duration limit."""
    seg = segs[i]
    s, e = _ms(seg, "start"), _ms(seg, "end")
    limit = s + _max_duration_ms
    nxt = segs[i + 1] if i + 1 < len(segs) else None
    prev = segs[i - 1] if i > 0 else None
    if nxt is None:
        ceiling = e + need
    else:
        ceiling = _ms(nxt, "start") - OVERLAP_GAP_MS
        if _is_tag(nxt):      # a [tag] cue may start later, down to its minimum
            ceiling = max(ceiling, _ms(nxt, "end") - _min_cue_ms - OVERLAP_GAP_MS)
    e2 = max(e, min(e + need, ceiling, limit))
    if nxt is not None and e2 + OVERLAP_GAP_MS > _ms(nxt, "start"):
        nxt["start"] = ms_to_time(e2 + OVERLAP_GAP_MS)
    need -= e2 - e
    s2 = s
    if need > 0:
        floor = _ms(prev, "end") + OVERLAP_GAP_MS if prev is not None else 0
        if prev is not None and _is_tag(prev):
            floor = min(floor, _ms(prev, "start") + _min_cue_ms + OVERLAP_GAP_MS)
        s2 = min(s, max(s - need, s - max_lead_in, floor, e2 - _max_duration_ms))
        if prev is not None and s2 - OVERLAP_GAP_MS < _ms(prev, "end"):
            prev["end"] = ms_to_time(s2 - OVERLAP_GAP_MS)
    seg["start"], seg["end"] = ms_to_time(s2), ms_to_time(e2)


def _extend_short(segs: List[Dict], min_ms: int) -> None:
    for i, seg in enumerate(segs):
        if not _is_tag(seg) and _dur(seg) < min_ms:
            _grow(segs, i, min_ms - _dur(seg), max_lead_in=min_ms)


def enforce_min_duration(segments: List[Dict],
                         min_ms: int = None) -> List[Dict]:
    """Guarantee a readable minimum duration for every cue.

    A too-short cue first grows into the free time around it -- its end
    into the pause after it, then its start into the pause before it --
    never overlapping a neighbour. A cue that is still short is joined with
    the neighbour it is closest to, when the joined cue still fits the
    layout and duration limits. (A near-invisible one is joined with the
    previous cue even when it does not fit.) The word-level SRT is untouched.
    """
    if min_ms is None:
        min_ms = _min_cue_ms
    if not segments:
        return segments
    segs = [dict(s) for s in segments]
    _extend_short(segs, min_ms)
    out: List[Dict] = []
    merged = 0
    i = 0
    while i < len(segs):
        seg = segs[i]
        if _is_tag(seg) or _dur(seg) >= min_ms:
            out.append(seg)
            i += 1
            continue
        prev = out[-1] if out else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        options = []
        if _can_join(prev, seg):
            options.append((_ms(seg, "start") - _ms(prev, "end"), 0, "prev"))
        if _can_join(seg, nxt):
            options.append((_ms(nxt, "start") - _ms(seg, "end"), 1, "next"))
        if options:
            side = min(options)[2]
            if side == "prev":
                out[-1] = _join_two(prev, seg)
            else:
                segs[i + 1] = _join_two(seg, nxt)
            merged += 1
        elif (prev is not None and not _is_tag(prev) and _dur(seg) < min_ms // 3
              and prev.get("speaker") == seg.get("speaker")):
            out[-1] = _join_two(prev, seg)
            merged += 1
        else:
            out.append(seg)
        i += 1
    _extend_short(out, min_ms)
    if merged:
        print(f"[ok] Joined {merged} short cue(s) with a neighbour")
    return out


def _resplit_pair(a: Dict, b: Dict) -> Optional[Tuple[Dict, Dict]]:
    """Move the boundary between neighbouring cues *a* and *b* to another
    word when that evens out their reading speed. Times follow the words, so
    sync is kept; the new break must be as clean as the old one or clean
    enough (below ``CLEAN_BREAK_LIMIT``), and both cues must still fit."""
    if _is_tag(a) or _is_tag(b) or a.get("dash_cue") or b.get("dash_cue"):
        return None
    if a.get("speaker") != b.get("speaker") or a.get("words") is None or b.get("words") is None:
        return None
    ta, tb = _flat(a["text"]).split(), _flat(b["text"]).split()
    tokens = ta + tb
    words = list(a["words"]) + list(b["words"])
    is_word = _token_is_word(tokens)
    if sum(is_word) != len(words) or len(words) < 2:
        return None
    cum = [0]
    for flag in is_word:
        cum.append(cum[-1] + (1 if flag else 0))
    start, end = _ms(a, "start"), _ms(b, "end")
    old_max = max(_cps(a), _cps(b))
    old_brk = break_cost(ta[-3:], tb[:3], _language)
    best = None
    for k in range(1, len(tokens)):
        wa = cum[k]
        if k == len(ta) or wa == 0 or wa == len(words):
            continue
        last, first = words[wa - 1], words[wa]
        if last.get("end") is None or first.get("start") is None:
            continue
        brk = break_cost(tokens[max(0, k - 3):k], tokens[k:k + 3], _language)
        if brk >= CLEAN_BREAK_LIMIT and brk > old_brk:
            continue
        left, right = " ".join(tokens[:k]), " ".join(tokens[k:])
        a_end = max(int(last["end"] * 1000), int(first["start"] * 1000) - OVERLAP_GAP_MS)
        b_start = max(int(first["start"] * 1000), a_end + OVERLAP_GAP_MS)
        if min(a_end - start, end - b_start) < _min_cue_ms \
                or max(a_end - start, end - b_start) > _max_duration_ms:
            continue
        new_max = max(len(left) * 1000.0 / (a_end - start),
                      len(right) * 1000.0 / (end - b_start))
        if new_max > old_max - 1.0:
            continue
        if not (_text_fits(left, clean=True) and _text_fits(right, clean=True)):
            continue
        score = new_max + 0.2 * max(0.0, brk - old_brk)
        if best is None or score < best[0]:
            best = (score, k, a_end, b_start)
    if best is None:
        return None
    _, k, a_end, b_start = best
    wa = cum[k]
    new_a = dict(a, text=balance_block_enhanced(" ".join(tokens[:k])), words=words[:wa],
                 end=ms_to_time(a_end), end_source=words[wa - 1].get("source"))
    new_b = dict(b, text=balance_block_enhanced(" ".join(tokens[k:])), words=words[wa:],
                 start=ms_to_time(b_start), start_source=words[wa].get("source"))
    return new_a, new_b


def _resplit_fast(segs: List[Dict], target_cps: float) -> int:
    moved = 0
    for i in range(len(segs)):
        if _is_tag(segs[i]) or _cps(segs[i]) <= target_cps:
            continue
        options = []
        if i + 1 < len(segs):
            pair = _resplit_pair(segs[i], segs[i + 1])
            if pair:
                options.append((max(map(_cps, pair)), i, pair))
        if i > 0:
            pair = _resplit_pair(segs[i - 1], segs[i])
            if pair:
                options.append((max(map(_cps, pair)), i - 1, pair))
        if options:
            _, j, pair = min(options, key=lambda o: o[0])
            segs[j], segs[j + 1] = pair
            moved += 1
    return moved


def relieve_fast_cues(segments: List[Dict], target_cps: float = TARGET_CPS,
                      max_cps: float = MAX_CPS) -> List[Dict]:
    """Slow down cues that read faster than *target_cps*.

    Each one first takes free time around it (its end lingers into the
    pause after it; its start may come up to ``MAX_LEAD_IN_MS`` early), only
    as much as it needs. A cue still faster than *max_cps* is joined with a
    neighbour when the joined cue fits and reads slower.
    """
    if not segments:
        return segments
    segs = [dict(s) for s in segments]

    def ease(cues: List[Dict]) -> None:
        for i, seg in enumerate(cues):
            if _is_tag(seg):
                continue
            need = int(len(_flat(seg["text"])) * 1000.0 / target_cps) - _dur(seg)
            if need > 0:
                _grow(cues, i, need, MAX_LEAD_IN_MS)

    ease(segs)
    moved = _resplit_fast(segs, target_cps)
    ease(segs)
    out: List[Dict] = []
    merged = 0
    i = 0
    while i < len(segs):
        seg = segs[i]
        own = _cps(seg)
        if _is_tag(seg) or own <= max_cps:
            out.append(seg)
            i += 1
            continue
        prev = out[-1] if out else None
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        best = None
        for side, a, b in (("prev", prev, seg), ("next", seg, nxt)):
            if not _can_join(a, b):
                continue
            other = a if side == "prev" else b
            joined = _join_two(a, b)
            cps = _cps(joined)
            if cps < own and (cps <= max_cps or _cps(other) > max_cps) \
                    and (best is None or cps < best[0]):
                best = (cps, side, joined)
        if best is None:
            out.append(seg)
        elif best[1] == "prev":
            out[-1] = best[2]
            merged += 1
        else:
            segs[i + 1] = best[2]
            merged += 1
        i += 1
    ease(out)
    if moved:
        print(f"[ok] Moved {moved} cue boundary(ies) to even out reading speed")
    if merged:
        print(f"[ok] Joined {merged} fast cue(s) with a neighbour")
    return out


def longest_line_length(content: str) -> int:
    """Longest single display line of a (possibly two-line) subtitle block.

    CPL is a *per-line* constraint. Measuring ``len(content)`` on a block
    that phase 3 already balanced across two lines counts both lines plus
    the newline, so every well-formed two-line cue looks like a violation.
    """
    return max((len(line) for line in content.splitlines()), default=0)


def validate_srt_output(entries, name: str) -> Dict[str, int]:
    print(f"\nValidating {name}...")
    issues = {"overlaps": 0, "gaps_large": 0, "duration_long": 0, "cpl_high": 0,
              "lines_high": 0}
    # Pairwise checks: only defined for entries that have a successor.
    for i in range(len(entries) - 1):
        current = entries[i]
        next_entry = entries[i + 1]
        if current.end > next_entry.start:
            issues["overlaps"] += 1
        if (next_entry.start - current.end).total_seconds() > 5.0:
            issues["gaps_large"] += 1
    # Per-entry checks: must cover every entry, including the last one.
    for entry in entries:
        if (entry.end - entry.start).total_seconds() * 1000 > _max_duration_ms:
            issues["duration_long"] += 1
        if longest_line_length(entry.content) > _max_cpl:
            issues["cpl_high"] += 1
        if len(entry.content.splitlines()) > _max_lines:
            issues["lines_high"] += 1
    print(f"  Issues found: {sum(issues.values())}")
    for issue_type, count in issues.items():
        if count > 0:
            print(f"    - {issue_type}: {count}")
    return issues


def merge_dash_cues(segments: List[Dict], overlaps=None, style: str = "-",
                    max_gap_ms: int = 300) -> List[Dict]:
    """Join quick two-speaker exchanges into one dash cue (SDH convention).

    Two consecutive cues by different speakers become one cue with one line
    each, each line led by *style* ("-" per Netflix's English style; "- "
    for a spaced dash), when they overlap or follow within *max_gap_ms*,
    each fits on one line, and the joined cue respects the duration limit.
    Only used when ``--dash-cues`` is set and diarization labelled speakers.
    """
    if _max_lines < 2:
        return segments
    out: List[Dict] = []
    i = 0
    while i < len(segments):
        a = segments[i]
        if i + 1 < len(segments):
            b = segments[i + 1]
            sa, sb = a.get("speaker"), b.get("speaker")
            a0, a1 = time_to_ms(a["start"]), time_to_ms(a["end"])
            b0, b1 = time_to_ms(b["start"]), time_to_ms(b["end"])
            ta = " ".join(a["text"].split())
            tb = " ".join(b["text"].split())
            near = b0 - a1 <= max_gap_ms
            if overlaps and not near:
                near = any(os_ * 1000 - max_gap_ms <= b0 <= oe * 1000 + max_gap_ms
                           for os_, oe in overlaps)
            fits = (len(style) + len(ta) <= _max_cpl and len(style) + len(tb) <= _max_cpl
                    and max(a1, b1) - a0 <= _max_duration_ms)
            if sa and sb and sa != sb and near and fits:
                out.append({**a, "end": ms_to_time(max(a1, b1)),
                            "end_source": b.get("end_source"),
                            "text": f"{style}{ta}\n{style}{tb}",
                            "speaker": f"{sa} / {sb}", "dash_cue": True})
                i += 2
                continue
        out.append(a)
        i += 1
    for k, seg in enumerate(out):
        seg["index"] = k + 1
    return out
