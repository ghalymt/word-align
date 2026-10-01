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

from .config import (CLAUSE_MARKERS, ITERATION_START, MAX_CPL,
                     MAX_DURATION_MS, MIN_LINE_RATIO, PREFER_NEW_LINE_WORDS)
from .utils import (count_visible_characters,
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
    balance_penalty = abs(l1 - l2)
    w2 = line2.split()[0].lower().rstrip(",.;") if line2 else ""
    w1 = line1.split()[-1].lower().rstrip(",.;") if line1 else ""
    score = balance_penalty
    if w2 in PREFER_NEW_LINE_WORDS:
        score -= 5
    if w1 in PREFER_NEW_LINE_WORDS:
        score += 15
    if w2 in CLAUSE_MARKERS["conjunctions"]:
        score -= 3
    return {"pos": pos, "score": score, "l1": line1, "l2": line2}


def check_if_text_can_be_split_prettily(text: str) -> bool:
    text = text.replace("\n", " ").strip()
    total_len = count_visible_characters(text)
    if total_len <= _max_cpl:
        return True
    if total_len > _cue_char_budget() or _max_lines < 2:
        return False
    for i in range(1, len(text)):
        if text[i] == " " and evaluate_split(text, i, total_len) is not None:
            return True
    return any(_best_multiline_split(text, n) is not None
               for n in range(3, _max_lines + 1))


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

    for ratio in (MIN_LINE_RATIO, RELAXED_LINE_RATIO):
        candidates = []
        for i in range(1, len(flat)):
            if flat[i] == " ":
                c = evaluate_split(flat, i, total, ratio)
                if c:
                    candidates.append(c)
        if candidates:
            best = min(candidates, key=lambda x: x["score"])
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
        w1 = prev.split()[-1].lower().rstrip(",.;")
        w2 = nxt.split()[0].lower().rstrip(",.;")
        if w2 in PREFER_NEW_LINE_WORDS:
            score -= 5
        if w1 in PREFER_NEW_LINE_WORDS:
            score += 15
        if w2 in CLAUSE_MARKERS["conjunctions"]:
            score -= 3
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
        curr_text = segments[i]["text"].strip()
        next_text = segments[i + 1]["text"].strip()
        match = re.search(r"(.*,\s)([^,.\s!?]+)$", curr_text, re.DOTALL)
        if match:
            prefix = match.group(1).strip()
            dangler = match.group(2).strip()
            if not dangler or len(dangler) > 20:
                continue
            segments[i]["text"] = prefix
            segments[i + 1]["text"] = f"{dangler} {next_text}"
    return segments


def parse_human_transcript_to_srt_segments(
        transcript_text: str, aligned_words: List[Dict],
        cue_boundaries: Optional[List[Tuple[int, int]]] = None) -> List[Dict]:
    """Map each transcript line onto its aligned word span.

    If ``cue_boundaries`` is provided, those (start, end) word-index ranges
    are used to build segments directly — this preserves the segmentation
    of an input SRT transcript instead of letting the iterative merger
    collapse neighbouring phrases together.
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
            })
        print(f"[ok] Created {len(segments)} segments from SRT cue boundaries")
        return segments

    for line_idx, line in enumerate(lines):
        if re.fullmatch(r"\s*\[.*?\]\s*", line):
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
            first_word = aligned_words[pos]
            last_word = aligned_words[pos + len(words) - 1]
            start = first_word.get("start")
            end = last_word.get("end")
            cursor = pos + len(words)
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
        if is_isolated_tag(segments[i]["text"]):
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
            if is_isolated_tag(next_txt):   # don't pull a tag into dialogue
                break
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
            group.append(segments[j])
            parts.append(segments[j]["text"])
            if has_terminal_punctuation(segments[j]["text"]):
                merged.append({"start": start, "end": segments[j]["end"],
                               "start_source": segments[i].get("start_source"),
                               "end_source": segments[j].get("end_source"),
                               "text": " ".join(parts),
                               "index": segments[i]["index"]})
                i = j + 1
                boundary_found = True
                break
            j += 1
        if not boundary_found:
            full_text = " ".join(parts)
            if len(group) > 1 and is_eligible_sentence(
                    full_text, start, segments[j - 1]["end"]):
                merged.append({"start": start, "end": segments[j - 1]["end"],
                               "start_source": segments[i].get("start_source"),
                               "end_source": segments[j - 1].get("end_source"),
                               "text": full_text,
                               "index": segments[i]["index"]})
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
        if is_isolated_tag(curr["text"]):
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
            if is_isolated_tag(nxt["text"]):
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
            acc = {"start": acc["start"], "end": nxt["end"],
                   "start_source": acc.get("start_source"),
                   "end_source": nxt.get("end_source"),
                   "text": combined, "index": acc["index"]}
            if has_terminal_punctuation(acc["text"]):
                j += 1
                break
            j += 1
        merged.append(acc)
        i = j
    return merged


def process_phase3(segments: List[Dict]) -> List[Dict]:
    return [{"index": s.get("index", i + 1), "start": s["start"],
             "end": s["end"], "text": balance_block_enhanced(s["text"]),
             "start_source": s.get("start_source"),
             "end_source": s.get("end_source")}
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


def enforce_min_duration(segments: List[Dict],
                         min_ms: int = None) -> List[Dict]:
    """Guarantee a readable minimum duration for every cue.

    Two passes: first extend a too-short cue's END into the free gap before the
    next cue (never creating an overlap); then merge any cue that is *still*
    degenerate (crammed with no room) into the previous cue, folding its text in
    and re-balancing. The word-level SRT is untouched.
    """
    if min_ms is None:
        min_ms = _min_cue_ms
    if not segments:
        return segments

    # Pass 1: extend short cues into available space.
    for i in range(len(segments)):
        s = time_to_ms(segments[i]["start"])
        e = time_to_ms(segments[i]["end"])
        if e - s >= min_ms:
            continue
        if i + 1 < len(segments):
            ceiling = time_to_ms(segments[i + 1]["start"]) - OVERLAP_GAP_MS
        else:
            ceiling = s + min_ms
        new_e = max(e, min(s + min_ms, ceiling))
        segments[i]["end"] = ms_to_time(new_e)

    # Pass 2: merge any cue still far too short into the previous one.
    out: List[Dict] = []
    merged = 0
    for seg in segments:
        dur = time_to_ms(seg["end"]) - time_to_ms(seg["start"])
        if out and dur < min_ms // 3:
            prev = out[-1]
            if time_to_ms(seg["end"]) > time_to_ms(prev["end"]):
                prev["end"] = seg["end"]
                prev["end_source"] = seg.get("end_source", prev.get("end_source"))
            joined = (prev["text"].replace("\n", " ").strip() + " "
                      + seg["text"].replace("\n", " ").strip()).strip()
            prev["text"] = balance_block_enhanced(joined)
            merged += 1
        else:
            out.append(dict(seg))
    if merged:
        print(f"[ok] Merged {merged} sub-frame cue(s) into neighbours")
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
