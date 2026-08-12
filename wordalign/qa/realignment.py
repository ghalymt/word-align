"""Local re-alignment: edit text → surgically repair timestamps.

When the user edits transcript text in the review editor, this module:
1. Identifies the edited span (by comparing word IDs / text)
2. Extends the audio region slightly for context
3. Keeps reliable neighboring timestamps as anchors
4. Re-runs local alignment on just the affected words
5. Replaces only the affected timestamps
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..core.types import WordResult


@dataclass
class EditRegion:
    """A span of words that was edited and needs re-alignment."""
    start_idx: int               # index in the word list
    end_idx: int                 # exclusive
    anchor_before: Optional[float]  # timestamp of the word before the edit
    anchor_after: Optional[float]   # timestamp of the word after the edit
    old_text: List[str]          # original word texts
    new_text: List[str]          # edited word texts


def detect_edits(original: List[WordResult],
                 edited: List[WordResult]) -> List[EditRegion]:
    """Compare original and edited word lists to find edit regions.

    Uses a simple diff: find spans where word texts differ.
    """
    regions = []
    i = 0
    while i < min(len(original), len(edited)):
        if original[i].text != edited[i].text:
            # Find the end of the edit span
            j = i
            while (j < len(original) and j < len(edited)
                   and original[j].text != edited[j].text):
                j += 1
            # Extend by 1 word on each side for context if possible
            start = max(0, i - 1)
            end = min(len(original), j + 1)

            anchor_before = original[start].start if start > 0 else None
            anchor_after = original[end].start if end < len(original) else None

            old_text = [w.text for w in original[i:j]]
            new_text = [w.text for w in edited[i:j]]

            regions.append(EditRegion(
                start_idx=i, end_idx=j,
                anchor_before=anchor_before,
                anchor_after=anchor_after,
                old_text=old_text, new_text=new_text,
            ))
            i = j
        else:
            i += 1
    return regions


def compute_local_timestamps(region: EditRegion,
                             words_per_second: float = 2.5) -> List[Tuple[float, float]]:
    """Estimate timestamps for edited words using linear interpolation.

    Falls back to a words-per-second heuristic when anchors are missing.

    Parameters
    ----------
    region : EditRegion
        The edit region with anchor timestamps.
    words_per_second : float
        Fallback speaking rate when interpolation isn't possible.

    Returns
    -------
    list of (start, end) tuples, one per edited word.
    """
    n = len(region.new_text)
    if n == 0:
        return []

    start_time = region.anchor_before or 0.0
    end_time = region.anchor_after or (start_time + n / words_per_second)
    duration = end_time - start_time
    per_word = duration / n if n > 0 else 0.3

    return [(start_time + i * per_word, start_time + (i + 1) * per_word)
            for i in range(n)]


def apply_realignment(words: List[WordResult],
                      regions: List[EditRegion]) -> List[WordResult]:
    """Apply re-aligned timestamps to the word list.

    Returns a new list with updated timestamps for edited words.
    Non-edited words are untouched.
    """
    result = list(words)
    for region in regions:
        timestamps = compute_local_timestamps(region)
        for i, (start, end) in enumerate(timestamps):
            idx = region.start_idx + i
            if idx < len(result):
                result[idx] = WordResult(
                    id=result[idx].id,
                    text=region.new_text[i] if i < len(region.new_text) else result[idx].text,
                    normalized_text=region.new_text[i].lower().strip(".,!?") if i < len(region.new_text) else result[idx].normalized_text,
                    start=start,
                    end=end,
                    confidence=0.5,  # lower confidence for re-aligned words
                    timing_source="realignment",
                    matched=True,
                    engine_id=result[idx].engine_id,
                    model_id=result[idx].model_id,
                )
    return result
