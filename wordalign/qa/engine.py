"""QA engine — orchestrates all QA signals and produces TranscriptIssues."""
from __future__ import annotations

from typing import List, Optional

from ..core.types import WordResult
from .deterministic import run_deterministic_qa
from .issues import TranscriptIssue


class QAEngine:
    """Orchestrates transcript quality analysis.

    Runs deterministic signals first (no GPU/LLM needed), then optionally
    adds LLM-based semantic review.
    """

    def __init__(self, enable_llm: bool = False,
                 llm_provider=None,
                 confidence_threshold: float = 0.5):
        self.enable_llm = enable_llm
        self.llm_provider = llm_provider
        self.confidence_threshold = confidence_threshold

    def review(self, consensus: List[WordResult]) -> List[TranscriptIssue]:
        """Run all enabled QA signals and return the combined issue list."""
        issues = run_deterministic_qa(consensus)

        if self.enable_llm and self.llm_provider:
            llm_issues = self._run_llm_review(consensus)
            issues.extend(llm_issues)

        # Sort by start time
        issues.sort(key=lambda x: x.start)
        return issues

    def _run_llm_review(self, consensus: List[WordResult]) -> List[TranscriptIssue]:
        """Run LLM-based semantic review on transcript chunks."""
        # Chunk the consensus into 60-120s segments with overlap
        chunks = self._chunk_transcript(consensus,
                                        target_seconds=90,
                                        context_seconds=20)
        issues = []
        seen_word_ids = set()

        for chunk in chunks:
            try:
                findings = self.llm_provider.review_chunk(
                    chunk["center"], chunk["before"], chunk["after"])
                for finding in findings:
                    word_ids = finding.get("word_ids", [])
                    # Deduplicate across overlapping chunks
                    new_ids = [wid for wid in word_ids if wid not in seen_word_ids]
                    if not new_ids:
                        continue
                    seen_word_ids.update(new_ids)

                    word_objs = [w for w in consensus if w.id in new_ids]
                    if not word_objs:
                        continue
                    issues.append(TranscriptIssue(
                        id=f"issue_llm_{len(issues):04d}",
                        category="semantic",
                        severity="high" if finding.get("confidence", 0) > 0.7 else "medium",
                        confidence=finding.get("confidence", 0.5),
                        start=word_objs[0].start or 0.0,
                        end=word_objs[-1].end or 0.0,
                        word_ids=new_ids,
                        original_text=" ".join(w.text for w in word_objs),
                        suggested_text=finding.get("suggested_text"),
                        explanation=finding.get("reason", "LLM flagged as suspicious"),
                        sources=["llm_review"],
                    ))
            except Exception:
                pass  # LLM failures shouldn't block deterministic QA

        return issues

    def _chunk_transcript(self, consensus: List[WordResult],
                          target_seconds: float = 90,
                          context_seconds: float = 20) -> list[dict]:
        """Split transcript into overlapping chunks for LLM review."""
        if not consensus:
            return []

        chunks = []
        timed = [w for w in consensus if w.start is not None]
        if not timed:
            return []

        total_duration = (timed[-1].end or timed[-1].start or 0) - timed[0].start
        if total_duration <= target_seconds:
            return [{"center": consensus, "before": [], "after": []}]

        chunk_start = timed[0].start
        while chunk_start < (timed[-1].end or timed[-1].start or 0):
            chunk_end = chunk_start + target_seconds
            center = [w for w in consensus
                      if w.start is not None
                      and chunk_start <= w.start < chunk_end]
            before = [w for w in consensus
                      if w.start is not None
                      and chunk_start - context_seconds <= w.start < chunk_start]
            after = [w for w in consensus
                     if w.start is not None
                     and chunk_end <= w.start < chunk_end + context_seconds]
            if center:
                chunks.append({"center": center, "before": before, "after": after})
            chunk_start = chunk_end

        return chunks
