"""``wordalign-bench`` -- the bad-audio benchmark from the command line.

    wordalign-bench sources  --dest D:/wa-bench/sources
    wordalign-bench build    --sources D:/wa-bench/sources --out D:/wa-bench/suite [--quick]
    wordalign-bench run      --suite D:/wa-bench/suite --out D:/wa-bench/results
    wordalign-bench report   --out D:/wa-bench/results
    wordalign-bench fit-triage --jobs D:/reviewed-jobs

See docs/benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _variants(spec: str, extra):
    from .runner import VARIANTS
    names = [v.strip() for v in spec.split(",") if v.strip()] if spec else list(VARIANTS)
    out = {}
    for n in names:
        if n not in VARIANTS:
            raise SystemExit(f"unknown variant '{n}' (known: {', '.join(VARIANTS)}); "
                             "add your own with --variant NAME=\"--flags ...\"")
        out[n] = list(VARIANTS[n])
    for item in extra or []:
        name, _, flags = item.partition("=")
        out[name.strip()] = flags.split()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wordalign-bench", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sources", help="download the public speech, noise and music")
    s.add_argument("--dest", required=True)

    b = sub.add_parser("build", help="build a suite of condition blocks")
    b.add_argument("--sources", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--quick", action="store_true", help="9 key conditions only")
    b.add_argument("--conditions", default=None, help="comma-separated condition ids")
    b.add_argument("--block-seconds", type=float, default=30.0)
    b.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("run", help="run pipeline variants over a suite")
    r.add_argument("--suite", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--variants", default="", help="comma-separated: v2.1,v2.2,robust")
    r.add_argument("--variant", action="append", metavar='NAME="--flags"',
                   help="an extra variant made of CLI flags")
    r.add_argument("--modes", default="reference,free")
    r.add_argument("-l", "--language", default="en")
    r.add_argument("--fresh-cache", action="store_true",
                   help="no stage-cache reuse between variants (honest run times)")

    rp = sub.add_parser("report", help="re-render report.md from report.json")
    rp.add_argument("--out", required=True)

    ft = sub.add_parser("fit-triage", help="re-fit triage weights on corrected jobs")
    ft.add_argument("--jobs", required=True,
                    help="folder with <name>_review.json beside a corrected <name>.srt")
    ft.add_argument("--dry-run", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "sources":
        from .sources import ensure_sources
        src = ensure_sources(a.dest)
        print(f"speech utterances: {len(src['speech'])}, noise clips: {len(src['noise'])}, "
              f"music tracks: {len(src['music'])}")
        return 0
    if a.cmd == "build":
        from .suite import QUICK_CONDITIONS, build_suite
        conds = None
        if a.conditions:
            conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
        elif a.quick:
            conds = QUICK_CONDITIONS
        build_suite(a.out, a.sources, conditions=conds, block_seconds=a.block_seconds,
                    seed=a.seed)
        return 0
    if a.cmd == "run":
        from .runner import run_suite, write_report
        modes = [m.strip() for m in a.modes.split(",") if m.strip()]
        report = run_suite(a.suite, a.out, _variants(a.variants, a.variant), modes,
                           a.language, a.fresh_cache)
        print(f"Report: {write_report(report, a.out)}")
        return 0
    if a.cmd == "report":
        from .runner import write_report
        report = json.loads((Path(a.out) / "report.json").read_text(encoding="utf-8"))
        print(write_report(report, a.out))
        return 0
    if a.cmd == "fit-triage":
        from ..qa.triage_calibration import fit_from_jobs
        return fit_from_jobs(a.jobs, dry_run=a.dry_run)
    return 1


if __name__ == "__main__":
    sys.exit(main())
