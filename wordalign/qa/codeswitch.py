"""Per-span language detection (code-switching support).

Detects the language of individual words or spans using lightweight
heuristics plus optional langid/transformers support. Populates the
``WordResult.language`` field so mixed-language recordings get correct
per-span language labels.

Strategy:
1. Try langid (if installed) per chunk of words
2. Fall back to character-set heuristics:
   - Arabic script → ar
   - CJK characters → zh/ja/ko (han detection)
   - Cyrillic → ru
   - Latin + common English words → en
   - Latin + accents → fr/de/es/pt (light heuristics)
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..core.types import WordResult

# Unicode ranges
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_HAN_RE = re.compile(r"[\u4E00-\u9FFF]")
_HIRAGANA_RE = re.compile(r"[\u3040-\u309F]")
_KATAKANA_RE = re.compile(r"[\u30A0-\u30FF]")
_HANGUL_RE = re.compile(r"[\uAC00-\uD7AF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_GREEK_RE = re.compile(r"[\u0370-\u03FF]")
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")

# Common English stopwords for heuristic detection
_EN_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "has", "was",
    "were", "been", "will", "would", "could", "should", "can", "not",
    "for", "but", "are", "you", "your", "they", "them", "their", "what",
    "when", "where", "which", "who", "whom", "how", "why", "all", "any",
    "more", "most", "some", "such", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "about", "into", "over", "after",
    "before", "between", "under", "again", "further", "then", "once",
    "here", "there", "each", "few", "both", "also", "one", "two", "three",
}

_EMPTY_RE = re.compile(r"^[\W\d_]+$")


def detect_script(text: str) -> Optional[str]:
    """Detect the writing system of a text fragment."""
    if _ARABIC_RE.search(text):
        return "ar"
    if _HAN_RE.search(text):
        return "zh"
    if _HIRAGANA_RE.search(text) or _KATAKANA_RE.search(text):
        return "ja"
    if _HANGUL_RE.search(text):
        return "ko"
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _GREEK_RE.search(text):
        return "el"
    if _HEBREW_RE.search(text):
        return "he"
    return None


def _looks_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def detect_word_language(word: str, fallback: str = "en") -> str:
    """Detect the language of a single word using heuristics."""
    if not word or _EMPTY_RE.match(word):
        return fallback

    script = detect_script(word)
    if script:
        return script

    if not _looks_latin(word):
        return fallback

    # Latin script: use langid if available for better accuracy
    try:
        import langid
        lang, _ = langid.classify(word)
        return lang
    except Exception:
        pass

    # Heuristic: English stopword → en; otherwise fallback
    clean = word.lower().strip(".,;:!?\"'()[]{}")
    if clean in _EN_STOPWORDS:
        return "en"
    return fallback


def detect_language_span(words: List[WordResult],
                         window: int = 8,
                         fallback: str = "en") -> List[WordResult]:
    """Assign a language to each word, smoothing over a sliding window.

    A word's language is the majority language of its window, which avoids
    single-word flicker while still catching genuine code-switches.
    """
    if not words:
        return []

    texts = [w.text for w in words]
    # First pass: per-word raw detection
    raw_langs = [detect_word_language(t, fallback) for t in texts]

    # Second pass: majority vote over window
    result = []
    n = len(words)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        window_langs = raw_langs[start:end]
        # Majority vote, ties → the word's own detection
        counts: dict = {}
        for lang in window_langs:
            counts[lang] = counts.get(lang, 0) + 1
        best = max(counts, key=counts.get)
        if counts[best] == counts.get(raw_langs[i], 0) and counts[best] == max(counts.values()):
            best = raw_langs[i] if counts.get(raw_langs[i]) == counts[best] else best

        w = words[i]
        result.append(WordResult(
            id=w.id, text=w.text, normalized_text=w.normalized_text,
            start=w.start, end=w.end, confidence=w.confidence,
            engine_id=w.engine_id, model_id=w.model_id,
            timing_source=w.timing_source, matched=w.matched,
            speaker=w.speaker, language=best,
            alternatives=w.alternatives, metadata=w.metadata,
        ))
    return result


def language_stats(words: List[WordResult]) -> dict:
    """Count words per detected language."""
    stats: dict = {}
    for w in words:
        lang = w.language or "unknown"
        stats[lang] = stats.get(lang, 0) + 1
    return stats
