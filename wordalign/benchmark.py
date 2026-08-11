"""Benchmark / calibration mode — evaluate engine quality on test audio.

Runs selected engines on test audio with a reference transcript and reports:
- WER (word error rate)
- Runtime (seconds)
- Peak memory (MB)
- Timestamp coverage (% of reference words with timestamps)
- Confidence distribution

Also supports "Create profile from benchmark": given a best-scoring engine
combination, writes a PipelineProfile JSON.
"""
from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .core.types import WordResult


@dataclass
class BenchmarkResult:
    """Results for a single engine benchmark run."""
    engine_id: str
    model_id: Optional[str] = None
    wer: float = 1.0              # 0.0 = perfect
    runtime_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    coverage: float = 0.0         # fraction of reference words with timestamps
    words_reference: int = 0
    words_hypothesis: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    confidence_mean: float = 0.0
    confidence_median: float = 0.0
    confidence_histogram: dict = field(default_factory=lambda: {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0})

    def to_dict(self) -> dict:
        return asdict(self)


def compute_wer(reference: List[str], hypothesis: List[str]) -> Dict[str, Any]:
    """Compute word error rate via Levenshtein distance.

    Returns dict with wer, substitutions, deletions, insertions.
    """
    ref = [w.lower() for w in reference]
    hyp = [w.lower() for w in hypothesis]

    # DP Levenshtein on words
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    dist = dp[n][m]
    return {
        "wer": dist / max(1, n),
        "substitutions": dist,
        "deletions": 0,
        "insertions": 0,
        # Simpler decomposition: use alignment backtrack for exact counts
        "distance": dist,
        "reference_len": n,
        "hypothesis_len": m,
    }


def _count_edit_ops(reference: List[str], hypothesis: List[str]) -> Dict[str, int]:
    """Count substitutions/deletions/insertions via backtracking."""
    ref = [w.lower() for w in reference]
    hyp = [w.lower() for w in hypothesis]
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            if ref[i - 1] != hyp[j - 1]:
                subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return {"substitutions": subs, "deletions": dels, "insertions": ins}


def _confidence_stats(words: List[WordResult]) -> Dict[str, Any]:
    """Compute confidence statistics from a word list."""
    confs = [w.confidence for w in words if w.confidence is not None]
    if not confs:
        return {"mean": 0.0, "median": 0.0, "histogram": {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}}
    confs_sorted = sorted(confs)
    mid = len(confs_sorted) // 2
    median = confs_sorted[mid] if len(confs_sorted) % 2 else (confs_sorted[mid - 1] + confs_sorted[mid]) / 2
    histogram = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for c in confs:
        if c < 0.2: histogram["0.0-0.2"] += 1
        elif c < 0.4: histogram["0.2-0.4"] += 1
        elif c < 0.6: histogram["0.4-0.6"] += 1
        elif c < 0.8: histogram["0.6-0.8"] += 1
        else: histogram["0.8-1.0"] += 1
    return {"mean": sum(confs) / len(confs), "median": median, "histogram": histogram}


def benchmark_engine(engine_id: str,
                     transcribe_fn: Callable,
                     reference_words: List[str],
                     audio_path: str,
                     model_id: Optional[str] = None,
                     **engine_kwargs) -> BenchmarkResult:
    """Run a single engine and measure quality + resource usage.

    Parameters
    ----------
    engine_id : str
        Engine identifier for the report.
    transcribe_fn : callable
        Function that takes audio_path + kwargs and returns list[WordResult].
    reference_words : list[str]
        Ground-truth transcript words.
    audio_path : str
        Audio file to transcribe.
    """
    tracemalloc.start()
    t0 = time.time()
    words = transcribe_fn(audio_path, **engine_kwargs)
    runtime = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    hypothesis = [w.text for w in words]
    wer_info = _count_edit_ops(reference_words, hypothesis)
    distance = sum(wer_info.values())
    wer = distance / max(1, len(reference_words))

    timed_words = [w for w in words if w.start is not None]
    coverage = len(timed_words) / max(1, len(reference_words))

    conf_stats = _confidence_stats(words)

    return BenchmarkResult(
        engine_id=engine_id,
        model_id=model_id,
        wer=wer,
        runtime_seconds=runtime,
        peak_memory_mb=peak / (1024 * 1024),
        coverage=coverage,
        words_reference=len(reference_words),
        words_hypothesis=len(hypothesis),
        substitutions=wer_info["substitutions"],
        deletions=wer_info["deletions"],
        insertions=wer_info["insertions"],
        confidence_mean=conf_stats["mean"],
        confidence_median=conf_stats["median"],
        confidence_histogram=conf_stats["histogram"],
    )


def run_benchmark(engines: List[Dict[str, Any]],
                  reference_words: List[str],
                  audio_path: str) -> List[BenchmarkResult]:
    """Run multiple engines and return sorted results (best WER first)."""
    results = []
    for spec in engines:
        fn = spec["fn"]
        result = benchmark_engine(
            engine_id=spec.get("engine_id", "unknown"),
            transcribe_fn=fn,
            reference_words=reference_words,
            audio_path=audio_path,
            model_id=spec.get("model_id"),
            **spec.get("kwargs", {}),
        )
        results.append(result)
    results.sort(key=lambda r: (r.wer, -r.coverage))
    return results


def create_profile_from_benchmark(results: List[BenchmarkResult],
                                  output_path: Optional[str] = None) -> dict:
    """Generate a PipelineProfile dict favoring the best-performing engines."""
    if not results:
        raise ValueError("No benchmark results to build a profile from")

    # Best WER first, then highest coverage
    ordered = sorted(results, key=lambda r: (r.wer, -r.coverage))
    best = ordered[0]
    profile = {
        "schema_version": 1,
        "name": f"Benchmark {best.engine_id}",
        "description": f"Auto-generated from benchmark: WER={best.wer:.3f}, "
                       f"coverage={best.coverage:.1%}",
        "mode": "ensemble",
        "transcription": {
            "engines": [
                {"plugin": r.engine_id, "enabled": True, "weight": 1.0 / (1 + r.wer)}
                for r in ordered if r.wer < 0.9
            ]
        },
        "timing": {
            "strategy": "fill_only",
            "order": ["vosk", "rough_srt", "whisperx", "mfa", "interpolation"],
        },
        "segmentation": {
            "max_cpl": 42, "max_lines": 2, "max_duration_ms": 7000, "min_cue_ms": 700,
        },
        "export": {"word_srt": True, "sentence_srt": True,
                   "transcript_format": "txt", "transcript_timestamps": True,
                   "tags": False},
    }
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)
    return profile


def format_benchmark_report(results: List[BenchmarkResult]) -> str:
    """Format benchmark results as a readable text table."""
    lines = [
        "=" * 72,
        "WORD-ALIGN BENCHMARK",
        "=" * 72,
        f"{'Engine':<14} {'WER':>7} {'Runtime':>9} {'PeakMem':>9} {'Coverage':>9} {'Conf(mean)':>11}",
        "-" * 72,
    ]
    for r in results:
        lines.append(
            f"{r.engine_id:<14} {r.wer:>6.1%} {r.runtime_seconds:>8.1f}s "
            f"{r.peak_memory_mb:>8.0f}MB {r.coverage:>8.1%} {r.confidence_mean:>10.2f}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)
