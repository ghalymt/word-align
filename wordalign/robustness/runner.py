"""Run pipeline variants over a suite and report per-condition numbers.

Variants are named sets of CLI flags, so anything the CLI can do can be
compared. The defaults compare v2.1's flow against v2.2's:

* ``v2.1``   -- the old flow: WhisperX-first transcript, fill-only timing
  waterfall, no VAD, no hallucination filter (``--timing fill_only``);
* ``v2.2``   -- the new defaults: VAD analysis, hallucination filter,
  WhisperX + Qwen + Vosk voting, anchors + forced alignment, triage;
* ``robust`` -- the robust profile: gated cleanup, Cohere + Qwen +
  parakeet + WhisperX voting, overlap handling.

Each variant runs in both modes: ``reference`` (the suite's transcript is
given, so only timing is measured) and ``free`` (transcript-free: text and
timing). Results go to ``<out>/report.json`` and ``<out>/report.md``.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import metrics

VARIANTS: Dict[str, List[str]] = {
    "v2.1": ["--timing", "fill_only", "--no-vad", "--no-hallucination-filter",
             "--no-triage"],
    "v2.2": [],
    "robust": ["--profile", "robust"],
}
MODES = ("reference", "free")


def _words_in(words: Sequence[Dict], start: float, end: float) -> List[Dict]:
    out = []
    for w in words:
        s, e = w.get("start"), w.get("end")
        if s is None or e is None:
            continue
        mid = (float(s) + float(e)) / 2
        if start <= mid < end:
            out.append(w)
    return out


def evaluate(truth: Sequence[Dict], blocks: Sequence[Dict], hyp: Sequence[Dict],
             mode: str, review: Optional[Dict] = None) -> Dict:
    """Per-block and overall metrics for one run."""
    per_block = []
    nonspeech_s, invented = 0.0, 0
    for b in blocks:
        bt = [w for w in truth if w.get("block") == b["id"]]
        # Hypothesis words up to 0.5 s outside a block still belong to it
        # (the 1 s gaps between blocks are silence).
        bh = _words_in(hyp, b["start"] - 0.5, b["end"] + 0.5)
        row = {"block": b["id"], "condition": b["condition"], "kind": b["kind"]}
        if b["kind"] == "nonspeech":
            nonspeech_s += b["end"] - b["start"]
            invented += len(bh)
            row["invented_words"] = len(bh)
        else:
            if mode == "free":
                row.update({k: v for k, v in metrics.wer(
                    " ".join(w["word"] for w in bt),
                    " ".join(w["word"] for w in bh)).items()
                    if k in ("wer", "substitutions", "deletions", "insertions", "reference")})
            row.update(metrics.timing_errors(bt, bh))
        per_block.append(row)
    speech_rows = [r for r in per_block if r["kind"] == "speech"]
    overall: Dict = {}
    if mode == "free":
        ref_n = sum(r.get("reference", 0) for r in speech_rows)
        err = sum(r.get("substitutions", 0) + r.get("deletions", 0) + r.get("insertions", 0)
                  for r in speech_rows)
        overall["wer"] = round(err / max(1, ref_n), 4)
    matched = sum(r.get("matched") or 0 for r in speech_rows)
    if matched:
        overall["mean_shift_ms"] = round(sum((r.get("mean_ms") or 0) * (r.get("matched") or 0)
                                             for r in speech_rows) / matched, 1)
        overall["within_100ms"] = round(sum((r.get("within_100ms") or 0) * (r.get("matched") or 0)
                                            for r in speech_rows) / matched, 3)
    if nonspeech_s:
        overall["invented_per_10min"] = round(invented / nonspeech_s * 600.0, 1)
    if review:
        overall["triage"] = metrics.triage_scores(review, truth, hyp, timed_truth=True)
    return {"overall": overall, "blocks": per_block}


def _hyp_words(result) -> List[Dict]:
    return [{"word": w.text, "start": w.start, "end": w.end}
            for w in result.aligned_words if w.start is not None]


def run_variant(suite_dir: str, variant: str, flags: Sequence[str], mode: str,
                out_dir: str, language: str = "en", fresh_cache: bool = False,
                log: Callable[[str], None] = print) -> Dict:
    from ..cli_v2 import _load_profile, _parse_args
    from ..core.events import CollectSink
    from ..core.pipeline import PipelineRunner
    suite = Path(suite_dir)
    target = Path(out_dir) / variant / mode
    target.mkdir(parents=True, exist_ok=True)
    argv = [str(suite / "bench.wav"), "-o", str(target), "-l", language, *flags]
    if mode == "reference":
        argv += ["-t", str(suite / "bench.txt")]
    cfg = _parse_args(argv)
    cache_ctx = contextlib.nullcontext()
    tmp_cache = None
    if fresh_cache:
        tmp_cache = tempfile.mkdtemp(prefix="wa_bench_cache_")
        cache_ctx = _env(WORDALIGN_CACHE=tmp_cache)
    sink = CollectSink()
    log_path = target / "run.log"
    t0 = time.time()
    with cache_ctx, open(log_path, "w", encoding="utf-8") as logf, \
            contextlib.redirect_stdout(logf):
        result = PipelineRunner(cfg, profile=_load_profile(cfg.profile_name), sink=sink).run()
    seconds = time.time() - t0
    if tmp_cache:
        shutil.rmtree(tmp_cache, ignore_errors=True)
    with open(log_path, "a", encoding="utf-8") as logf:
        for e in sink.events:
            msg = getattr(e, "message", None)
            if msg:
                logf.write(f"[{getattr(e, 'stage', '')}] {msg}\n")
    review = None
    if getattr(result, "review_json_path", None):
        review = json.loads(Path(result.review_json_path).read_text(encoding="utf-8"))
    truth = json.loads((suite / "bench.words.json").read_text(encoding="utf-8"))
    blocks = json.loads((suite / "bench.blocks.json").read_text(encoding="utf-8"))
    scores = evaluate(truth, blocks, _hyp_words(result), mode, review)
    scores["seconds"] = round(seconds, 1)
    scores["error"] = result.error
    scores["stage_seconds"] = {k: round(v, 1) for k, v in
                               sorted(getattr(result, "statistics", {}).get("stage_seconds", {}).items())}
    log(f"  {variant:<8} {mode:<9} {seconds:6.0f}s  " + ", ".join(
        f"{k} {v}" for k, v in scores["overall"].items() if k != "triage"))
    return scores


@contextlib.contextmanager
def _env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_suite(suite_dir: str, out_dir: str, variants: Optional[Dict[str, List[str]]] = None,
              modes: Sequence[str] = MODES, language: str = "en",
              fresh_cache: bool = False, log: Callable[[str], None] = print) -> Dict:
    variants = variants or VARIANTS
    report: Dict = {"suite": os.path.abspath(suite_dir), "variants": {}, "created": time.time()}
    for name, flags in variants.items():
        report["variants"][name] = {"flags": list(flags)}
        for mode in modes:
            try:
                report["variants"][name][mode] = run_variant(
                    suite_dir, name, flags, mode, out_dir, language, fresh_cache, log)
            except Exception as exc:     # one broken variant must not lose the rest
                log(f"  {name} {mode} failed: {exc}")
                report["variants"][name][mode] = {"error": str(exc)}
            write_report(report, out_dir)
    return report


def write_report(report: Dict, out_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return str(out / "report.md")


def _fmt(v, pct=False):
    if v is None:
        return "–"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def render_markdown(report: Dict) -> str:
    variants = report.get("variants", {})
    names = list(variants)
    lines = ["# word-align bad-audio benchmark", "", f"Suite: `{report.get('suite')}`", ""]
    lines += ["## Overall", "",
              "| Variant | Mode | WER | Mean shift (ms) | Within 100 ms | Invented words / 10 min "
              "| Triage recall | Accept cues with errors | Run time (s) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        for mode in MODES:
            r = variants[name].get(mode)
            if not r or "overall" not in r:
                if r and r.get("error"):
                    lines.append(f"| {name} | {mode} | error: {r['error'][:60]} |||||||")
                continue
            o = r["overall"]
            tri = o.get("triage") or {}
            lines.append(
                f"| {name} | {mode} | {_fmt(o.get('wer'), True)} | {_fmt(o.get('mean_shift_ms'))} "
                f"| {_fmt(o.get('within_100ms'), True)} | {_fmt(o.get('invented_per_10min'))} "
                f"| {_fmt(tri.get('recall'), True)} | {_fmt(tri.get('accept_error_rate'), True)} "
                f"| {_fmt(r.get('seconds'))} |")
    for mode, label, key, pct in (("free", "WER by condition (transcript-free)", "wer", True),
                                  ("reference", "Mean word-start shift by condition, ms (reference mode)", "mean_ms", False),
                                  ("free", "Mean word-start shift by condition, ms (transcript-free)", "mean_ms", False)):
        rows: Dict[str, Dict[str, object]] = {}
        for name in names:
            r = variants[name].get(mode) or {}
            for b in r.get("blocks", []):
                if b["kind"] != "speech":
                    continue
                rows.setdefault(b["condition"], {})[name] = b.get(key)
        if not rows:
            continue
        lines += ["", f"## {label}", "", "| Condition | " + " | ".join(names) + " |",
                  "|---|" + "---|" * len(names)]
        for cond, vals in rows.items():
            lines.append(f"| {cond} | " + " | ".join(_fmt(vals.get(n), pct) for n in names) + " |")
    rows = {}
    for name in names:
        r = variants[name].get("free") or {}
        for b in r.get("blocks", []):
            if b["kind"] == "nonspeech":
                rows.setdefault(b["condition"], {})[name] = b.get("invented_words")
    if rows:
        lines += ["", "## Invented words on non-speech (transcript-free)", "",
                  "| Block | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for cond, vals in rows.items():
            lines.append(f"| {cond} | " + " | ".join(_fmt(vals.get(n)) for n in names) + " |")
    return "\n".join(lines) + "\n"
