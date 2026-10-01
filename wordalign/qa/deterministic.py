"""Deterministic QA signals: ensemble disagreement, repetition, timing anomalies.

These signals require no GPU and no LLM — they analyze the existing
per-word confidence and timing data from the pipeline.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from ..core.types import WordResult
from .issues import RiskScore, TranscriptIssue

# Thresholds
LOW_CONFIDENCE_THRESHOLD = 0.5
HIGH_RATE_WPS = 5.0       # words per second
LOW_RATE_WPS = 0.3
MAX_INTERPOLATED_RUN = 5  # consecutive interpolated words
REPETITION_MIN_N = 3
REPETITION_WINDOW_SECONDS = 30.0


def detect_agreement_issues(consensus: List[WordResult],
                            threshold: float = LOW_CONFIDENCE_THRESHOLD
                            ) -> List[TranscriptIssue]:
    """Flag words with low ensemble agreement.

    Uses the ``confidence`` field on WordResult, which holds the fused
    agreement score from ``build_consensus()``.
    """
    issues = []
    for i, word in enumerate(consensus):
        if word.confidence < threshold:
            word_ids = [word.id]
            issues.append(TranscriptIssue(
                id=f"issue_agree_{i:04d}",
                category="agreement",
                severity="high" if word.confidence < 0.3 else "medium",
                confidence=word.confidence,
                start=word.start or 0.0,
                end=word.end or 0.0,
                word_ids=word_ids,
                original_text=word.text,
                explanation=f"Low ensemble agreement: {word.confidence:.0%}. "
                            f"Engines disagree on this word.",
                sources=["ensemble_disagreement"],
                risk=RiskScore(agreement_risk=1.0 - word.confidence,
                                confidence_risk=1.0 - word.confidence),
            ))
    return issues


def detect_repetition(consensus: List[WordResult],
                      min_n: int = REPETITION_MIN_N,
                      max_n: int = 8,
                      window_seconds: float = REPETITION_WINDOW_SECONDS
                      ) -> List[TranscriptIssue]:
    """Detect repeated n-grams that may indicate hallucination.

    Looks for n-grams (n=min_n..max_n) that appear more than once within
    a ``window_seconds`` time window.
    """
    issues = []
    words = [(w.text.lower(), w.start or 0.0, w.id) for w in consensus
             if w.start is not None]

    for n in range(min_n, min(max_n + 1, len(words))):
        # Build n-gram -> positions map
        ngram_positions: Dict[tuple, list] = {}
        for i in range(len(words) - n + 1):
            ngram = tuple(w[0] for w in words[i:i + n])
            ngram_positions.setdefault(ngram, []).append(i)

        for ngram, positions in ngram_positions.items():
            if len(positions) < 2:
                continue
            # Check if any pair is within the window
            for j in range(len(positions) - 1):
                p1 = positions[j]
                p2 = positions[j + 1]
                t1 = words[p1][1]
                t2 = words[p2][1]
                if t2 - t1 > window_seconds:
                    continue
                # Found a repetition
                word_ids = [words[p2 + k][2] for k in range(n)]
                start = words[p2][1]
                end = words[p2 + n - 1][1] if p2 + n - 1 < len(words) else start
                issues.append(TranscriptIssue(
                    id=f"issue_rep_{p2:04d}_{n}",
                    category="repetition",
                    severity="high",
                    confidence=0.85,
                    start=start,
                    end=end,
                    word_ids=word_ids,
                    original_text=" ".join(words[p2 + k][0] for k in range(n)),
                    explanation=f"Repeated phrase: "
                                f"'{' '.join(ngram)}' (also at {t1:.1f}s)",
                    sources=["repetition_detection"],
                    risk=RiskScore(repetition_risk=0.9),
                ))
                break  # Only flag first repetition of this n-gram

    return issues


def detect_timing_anomalies(consensus: List[WordResult]) -> List[TranscriptIssue]:
    """Flag timing problems: extreme speaking rates, interpolation runs,
    impossible overlaps."""
    issues = []
    timed = [w for w in consensus if w.start is not None and w.end is not None]

    # Speaking rate anomalies (every word, including the last one)
    for i in range(len(timed)):
        w = timed[i]
        w_next = timed[i + 1] if i + 1 < len(timed) else None
        gap = w_next.start - w.end if w_next is not None else 0.0
        duration = w.end - w.start

        # Extremely fast speech
        if duration > 0 and duration < 0.1 and w.text and len(w.text) > 2:
            issues.append(TranscriptIssue(
                id=f"issue_timing_fast_{i:04d}",
                category="timing",
                severity="medium",
                confidence=0.7,
                start=w.start,
                end=w.end,
                word_ids=[w.id],
                original_text=w.text,
                explanation=f"Unusually short duration: {duration*1000:.0f}ms",
                sources=["timing_anomaly"],
                risk=RiskScore(timing_risk=0.7),
            ))

        # Impossible overlap
        if gap < -0.05:
            issues.append(TranscriptIssue(
                id=f"issue_timing_overlap_{i:04d}",
                category="timing",
                severity="medium",
                confidence=0.8,
                start=w.start,
                end=w_next.end or w_next.start,
                word_ids=[w.id, w_next.id],
                original_text=f"{w.text} {w_next.text}",
                explanation=f"Overlapping timestamps: gap={gap*1000:.0f}ms",
                sources=["timing_anomaly"],
                risk=RiskScore(timing_risk=0.8),
            ))

    # Long runs of interpolated words. The trailing None flushes a run that
    # reaches the end of the file -- the most common place for one (the
    # engines stopped recognising speech before the transcript ended).
    interpolated_run = []
    for i, word in enumerate([*consensus, None]):
        if word is not None and word.timing_source == "Interpolated":
            interpolated_run.append((i, word))
        else:
            if len(interpolated_run) >= MAX_INTERPOLATED_RUN:
                first = interpolated_run[0][1]
                last = interpolated_run[-1][1]
                word_ids = [w.id for _, w in interpolated_run]
                issues.append(TranscriptIssue(
                    id=f"issue_timing_interp_{interpolated_run[0][0]:04d}",
                    category="timing",
                    severity="medium",
                    confidence=0.6,
                    start=first.start or 0.0,
                    end=last.end or 0.0,
                    word_ids=word_ids,
                    original_text=" ".join(w.text for _, w in interpolated_run),
                    explanation=f"{len(interpolated_run)} consecutive "
                                f"interpolated words (no reliable timestamps)",
                    sources=["timing_anomaly"],
                    risk=RiskScore(timing_risk=0.6),
                ))
            interpolated_run = []

    return issues


def run_deterministic_qa(consensus: List[WordResult]) -> List[TranscriptIssue]:
    """Run all deterministic QA signals and return combined issue list."""
    issues = []
    issues.extend(detect_agreement_issues(consensus))
    issues.extend(detect_repetition(consensus))
    issues.extend(detect_timing_anomalies(consensus))

    # Deduplicate by word_id overlap
    seen_word_ids = set()
    deduped = []
    for issue in sorted(issues, key=lambda x: x.start):
        if any(wid in seen_word_ids for wid in issue.word_ids):
            # Merge into existing issue rather than skip
            for existing in deduped:
                if any(wid in existing.word_ids for wid in issue.word_ids):
                    existing.sources = list(set(existing.sources + issue.sources))
                    if existing.risk and issue.risk:
                        existing.risk.agreement_risk = max(
                            existing.risk.agreement_risk, issue.risk.agreement_risk)
                        existing.risk.timing_risk = max(
                            existing.risk.timing_risk, issue.risk.timing_risk)
                        existing.risk.repetition_risk = max(
                            existing.risk.repetition_risk, issue.risk.repetition_risk)
                    break
        else:
            deduped.append(issue)
            seen_word_ids.update(issue.word_ids)

    return deduped
