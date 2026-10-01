"""Conversion helpers between typed WordResult/SegmentResult and legacy dicts.

The existing algorithm files (align.py, ensemble.py, segment.py) operate on
``list[dict]`` with keys like ``word``, ``start``, ``end``, ``conf``, ``source``,
``matched``. These converters bridge the new typed architecture to those
algorithms without modifying them.
"""
from __future__ import annotations

from typing import Any

from .types import SegmentResult, WordResult


def words_to_dicts(words: list[WordResult]) -> list[dict[str, Any]]:
    """Convert typed WordResult list to legacy dict format for align.py/segment.py."""
    return [
        {
            "word": w.text,
            "start": w.start,
            "end": w.end,
            "conf": w.confidence,
            "source": w.timing_source,
            "matched": w.matched,
        }
        for w in words
    ]


def dicts_to_words(dicts: list[dict[str, Any]],
                   start_id: int = 0) -> list[WordResult]:
    """Convert legacy dict list to typed WordResult list."""
    from ..utils import normalize_word

    result = []
    for i, d in enumerate(dicts):
        wid = f"w{start_id + i:06d}"
        text = d.get("word", "")
        result.append(WordResult(
            id=wid,
            text=text,
            normalized_text=normalize_word(text),
            start=d.get("start"),
            end=d.get("end"),
            confidence=float(d.get("conf", 1.0)),
            timing_source=d.get("source"),
            matched=d.get("matched", False),
            engine_id=d.get("engine_id"),
            model_id=d.get("model_id"),
        ))
    return result


def segments_to_dicts(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pass-through for segment dicts (segment.py already uses dicts).

    This exists for type consistency in the pipeline runner.
    """
    return segments


def dict_segment_to_typed(d: dict[str, Any]) -> SegmentResult:
    """Convert a segment dict to SegmentResult."""
    from ..utils import time_to_ms

    return SegmentResult(
        index=d.get("index", 0),
        start=time_to_ms(d.get("start", "00:00:00,000")) / 1000.0,
        end=time_to_ms(d.get("end", "00:00:00,000")) / 1000.0,
        text=d.get("text", ""),
        start_source=d.get("start_source"),
        end_source=d.get("end_source"),
        speaker=d.get("speaker"),
    )
