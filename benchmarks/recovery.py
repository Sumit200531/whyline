"""Planted-incident recovery benchmark.

Each trial generates ten weeks of funnel data and plants 0–3 random incidents
(rate shocks or traffic surges/drops on random 1- or 2-dimension segments) in
the last week. We then check what each method reports.

Methods
-------
whyline   : this library, default settings.
top-movers: the usual dashboard approach — rank every segment of depth ≤ 2 by
            its contribution to the change and report the top k, where k is
            *told* the true number of incidents (an advantage whyline lacks).

Usage: python benchmarks/recovery.py [trials] [seed]
"""

from __future__ import annotations

import sys
import time
from itertools import combinations

import numpy as np

import whyline as wl
from whyline.cube import Cube
from whyline.synth import DIMENSIONS, Incident, make_funnel

CR = wl.Ratio("conversions", "sessions", percent=True)
DIMS = list(DIMENSIONS)


def random_incidents(rng, k):
    out, used = [], set()
    while len(out) < k:
        depth = rng.choice([1, 2], p=[0.5, 0.5])
        dims = sorted(rng.choice(DIMS, depth, replace=False), key=DIMS.index)
        seg = {d: str(rng.choice(DIMENSIONS[d][0])) for d in dims}
        # keep incidents on disjoint dimensions-values so ground truth is unambiguous
        if any(seg.get(d) == v for s in used for d, v in s):
            continue
        if rng.random() < 0.5:
            mult = rng.uniform(0.45, 0.7) if rng.random() < 0.6 else rng.uniform(1.35, 1.7)
            inc = Incident(seg, rate=mult)
        else:
            mult = rng.uniform(2.0, 3.5) if rng.random() < 0.6 else rng.uniform(0.3, 0.5)
            inc = Incident(seg, volume=mult)
        used.add(tuple(seg.items()))
        out.append(inc)
    return out


def top_movers(df, k):
    cube = Cube(df, CR, DIMS, "week")
    n0, d0 = cube.state(len(cube.periods) - 2)
    n1, d1 = cube.state(len(cube.periods) - 1)
    rows = []
    for depth in (1, 2):
        for dims in combinations(DIMS, depth):
            for vals in {tuple(cube.codes[d][i] for d in dims) for i in range(len(cube.segments))}:
                seg = dict(zip(dims, vals))
                comps = CR.components(n0, d0, n1, d1, cube.mask(seg))
                mech = max(comps, key=lambda m: abs(comps[m]))
                rows.append((abs(sum(comps.values())), seg, mech))
    rows.sort(key=lambda r: -r[0])
    return [(r[1], r[2]) for r in rows[:k]]


def key(seg, mech):
    return tuple(sorted(seg.items())), mech


def canonical(seg, mech):
    """In a two-valued dimension, 'A's share rose' is the same event as 'B's share fell'."""
    if mech == "mix" and len(seg) == 1:
        (d, v), = seg.items()
        values = DIMENSIONS[d][0]
        if len(values) == 2:
            seg = {d: min(values)}
    return key(seg, mech)


def impacts(df, incidents):
    """True counterfactual impact of each planted incident, in units of the
    metric's normal week-to-week standard deviation."""
    from whyline.explain import _Noise, _effect
    cube = Cube(df, CR, DIMS, "week")
    ci = len(cube.periods) - 1
    n0, d0 = cube.state(ci - 1)
    n1, d1 = cube.state(ci)
    sd = _Noise(cube, ci - 1, ci, 8).total_sd()
    return [abs(_effect(CR, n0, d0, n1, d1, cube.mask(i.segment), i.mechanism)) / sd for i in incidents]


def main(trials=200, seed=0, detectable=1.0, quiet=False, **options):
    """An incident counts as a cause to find when its true impact on the metric
    exceeds `detectable` × the normal week-to-week standard deviation; quieter
    incidents are not counted against recall, nor is finding them a false positive."""
    rng = np.random.default_rng(seed)
    stats = {m: dict(tp=0, fp=0, planted=0, quiet=0, quiet_fa=0)
             for m in ("whyline", "top-movers")}
    t_total, n_silent = 0.0, 0
    for _ in range(trials):
        k = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.3, 0.3, 0.25]))
        incidents = random_incidents(rng, k)
        ds = make_funnel(seed=int(rng.integers(1 << 31)), incidents=incidents)
        size = impacts(ds.frame, incidents)
        truth = {canonical(i.segment, i.mechanism) for i, sz in zip(incidents, size) if sz >= detectable}
        silent = {canonical(i.segment, i.mechanism) for i in incidents} - truth
        n_silent += len(silent)

        t0 = time.perf_counter()
        r = wl.explain(ds.frame, CR, DIMS, "week", **options)
        t_total += time.perf_counter() - t0
        reports = {
            "whyline": [(c.segment, c.mechanism) for c in r.causes],
            "top-movers": top_movers(ds.frame, len(truth)) if truth else [],
        }
        for name, found in reports.items():
            st = stats[name]
            got = {canonical(*f) for f in found} - silent
            st["planted"] += len(truth)
            st["tp"] += len(got & truth)
            st["fp"] += len(got - truth)
            if not truth:
                st["quiet"] += 1
                st["quiet_fa"] += bool(got)

    w = stats["whyline"]
    if quiet:
        return stats
    print(f"{trials} trials · {w['planted']} material incidents "
          f"(+{n_silent} planted but too small to move the metric) · "
          f"whyline mean runtime {t_total / trials * 1000:.0f} ms\n")
    print(f"{'method':<12} {'recall':>8} {'precision':>10} {'F1':>6} {'false alarms on quiet weeks':>29}")
    for name, st in stats.items():
        rec = st["tp"] / max(st["planted"], 1)
        prec = st["tp"] / max(st["tp"] + st["fp"], 1)
        f1 = 2 * rec * prec / max(rec + prec, 1e-9)
        fa = f"{st['quiet_fa']}/{st['quiet']}" if name == "whyline" else "n/a (told k)"
        print(f"{name:<12} {rec:>8.0%} {prec:>10.0%} {f1:>6.0%} {fa:>29}")
    return stats


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(*args)
