"""Command line: ``whyline explain`` and ``whyline demo``."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import synth
from .explain import explain
from .metrics import Ratio, Sum


def _read(path: str) -> pd.DataFrame:
    if path == "-":
        return pd.read_csv(sys.stdin)
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if path.endswith(".json"):
        return pd.read_json(path)
    return pd.read_csv(path)


def _metric(args) -> Sum | Ratio:
    if args.ratio:
        num, sep, den = args.ratio.partition("/")
        if not sep:
            raise SystemExit("--ratio must look like numerator/denominator")
        return Ratio(num.strip(), den.strip(), name=args.name, percent=args.percent)
    return Sum(args.sum, name=args.name)


def _emit(result, args) -> None:
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=float))
    else:
        print(result.to_text())
    if args.html:
        result.to_html(args.html)
        print(f"\nHTML report written to {args.html}", file=sys.stderr)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-causes", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--html", metavar="PATH", help="also write an HTML report")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="whyline", description="Explain why a metric moved.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("explain", help="explain a change in your data")
    ex.add_argument("file", help="CSV / Parquet / JSON file, or - for stdin")
    g = ex.add_mutually_exclusive_group(required=True)
    g.add_argument("--sum", metavar="COL", help="additive metric, e.g. revenue")
    g.add_argument("--ratio", metavar="NUM/DEN", help="ratio metric, e.g. conversions/sessions")
    ex.add_argument("--by", required=True, help="comma-separated dimensions")
    ex.add_argument("--time", required=True, help="period column")
    ex.add_argument("--baseline", help="baseline period (default: second to last)")
    ex.add_argument("--current", help="current period (default: last)")
    ex.add_argument("--name", help="display name for the metric")
    ex.add_argument("--percent", action="store_true", help="format a ratio as a percentage")
    _common(ex)

    demo = sub.add_parser("demo", help="run on synthetic data with planted incidents")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--save-data", metavar="PATH", help="write the synthetic CSV")
    _common(demo)

    args = ap.parse_args(argv)
    if args.cmd == "explain":
        df = _read(args.file)
        dims = [d.strip() for d in args.by.split(",") if d.strip()]
        result = explain(df, _metric(args), dims, args.time, args.baseline, args.current,
                         max_causes=args.max_causes, max_depth=args.max_depth)
        _emit(result, args)
    else:
        ds = synth.demo(args.seed)
        if args.save_data:
            ds.frame.to_csv(args.save_data, index=False)
        metric = Ratio("conversions", "sessions", name="conversion rate", percent=True)
        result = explain(ds.frame, metric, ds.dimensions, "week",
                         max_causes=args.max_causes, max_depth=args.max_depth)
        _emit(result, args)
        if not args.json:
            print("\nWhat was actually planted:")
            for inc in ds.incidents:
                seg = " · ".join(f"{k}={v}" for k, v in inc.segment.items())
                print(f"  - {inc.mechanism:<4}  {seg:<34} {inc.label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
