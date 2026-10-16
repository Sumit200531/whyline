"""Root-cause discovery and attribution.

The algorithm, in four moves:

1. **Discover** (greedy, counterfactual).  For every segment and mechanism
   (rate / mix / change) ask: how much of the change disappears if this
   segment had kept its baseline rate (or baseline share)?  Take the largest
   such effect, drill into a sub-segment while one clearly more intense child
   carries most of it (``country=IN`` → ``country=IN · device=android``), then
   *neutralize* it and search what is left.

2. **Gate** (noise + materiality).  Every candidate is compared with how much
   the same segment and mechanism moved in earlier periods.  If the overall
   change is itself unusual, material causes need |z| ≥ ``noise_z``;
   otherwise candidates must clear a Bonferroni-corrected Student-t threshold
   over the whole search.  Causes must also move the headline metric by a
   noticeable fraction of its normal period-to-period movement.

3. **Attribute** (Shapley).  Greedy order is arbitrary and causes overlap, so
   each cause's impact is its exact Shapley value in the game
   ``v(S) = metric after neutralizing causes S``.  Impacts plus the residual
   sum *exactly* to the observed change.

4. **Explain away**.  A cause that no longer matters once all the others are
   neutralized (typically a composition side-effect of another cause, i.e.
   Simpson's paradox) is dropped, weakest first, and attribution recomputed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .cube import Cube
from .metrics import Ratio, Sum


@dataclass
class Cause:
    segment: dict[str, str]
    mechanism: str                 # "rate" | "mix" | "change"
    impact: float                  # Shapley-attributed change in the metric
    share: float                   # impact / total change
    values: dict[str, float]       # segment-level base/current values
    components: dict[str, float]   # raw mix/rate split of the segment
    z: float | None = None
    z_method: str | None = None

    @property
    def name(self) -> str:
        return " · ".join(f"{k}={v}" for k, v in self.segment.items())

    @property
    def verdict(self) -> str:
        if self.z is None or not math.isfinite(self.z):
            return "untested"
        a = abs(self.z)
        return "strong" if a >= 4 else "likely real" if a >= 2 else "within normal variation"


@dataclass
class Explanation:
    metric: Sum | Ratio
    baseline: str
    current: str
    base_value: float
    current_value: float
    causes: list[Cause]
    residual: float
    dimensions: list[str]
    history_periods: int = 0
    total_z: float | None = None

    @property
    def delta(self) -> float:
        return self.current_value - self.base_value

    @property
    def explained_share(self) -> float:
        if self.delta == 0:
            return float("nan")
        return sum(c.impact for c in self.causes) / self.delta

    def headline(self) -> str:
        from .report import headline
        return headline(self)

    def to_text(self) -> str:
        from .report import render_text
        return render_text(self)

    def to_html(self, path: str | None = None) -> str:
        from .report import render_html
        html = render_html(self)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(html)
        return html

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for i, c in enumerate(self.causes, 1):
            rows.append({"rank": i, "segment": c.name, "mechanism": c.mechanism,
                         "impact": c.impact, "share": c.share, "z": c.z,
                         "verdict": c.verdict, **{f"segment_{k}": v for k, v in c.values.items()}})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {
            "metric": self.metric.label, "baseline": self.baseline, "current": self.current,
            "base_value": self.base_value, "current_value": self.current_value,
            "delta": self.delta, "residual": self.residual,
            "causes": [{"segment": c.segment, "mechanism": c.mechanism, "impact": c.impact,
                        "share": c.share, "z": c.z, "verdict": c.verdict, "values": c.values}
                       for c in self.causes],
        }

    def __str__(self) -> str:
        return self.to_text()

    def _repr_html_(self) -> str:  # Jupyter
        return self.to_html()


# ----------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------

def explain(df: pd.DataFrame, metric: Sum | Ratio, dimensions: Sequence[str], time: str,
            baseline=None, current=None, *, max_causes: int = 6, max_depth: int = 3,
            drill: float = 0.7, min_share: float = 0.04, noise_z: float = 3.0,
            history: int = 8, materiality: float = 0.75, search_depth: int = 1) -> Explanation:
    """Explain why ``metric`` changed between two periods.

    Parameters
    ----------
    df : long-format data, one row per (period, dimension values...) or finer.
    metric : :class:`Sum` or :class:`Ratio`.
    dimensions : columns to slice by.
    time : the period column.
    baseline, current : periods to compare (default: the last two).
    max_causes : stop after this many causes.
    max_depth : maximum number of dimensions in one cause's segment.
    drill : refine a segment into a child when the child carries at least this
        fraction of the parent's effect.
    min_share : ignore candidate causes smaller than this share of the change.
    noise_z : when history is available, stop at candidates whose |z| is
        below this (they look like normal fluctuation).
    history : number of earlier period-pairs used to estimate noise.
    search_depth : segments up to this many dimensions are scored directly
        (deeper ones are reached by drilling down).
    materiality : ignore causes that move the metric by less than this many
        standard deviations of its normal period-to-period movement.
    """
    cube = Cube(df, metric, dimensions, time)
    if len(cube.periods) < 2:
        raise ValueError("need at least two periods to compare")
    ci = cube.index_of(current) if current is not None else len(cube.periods) - 1
    bi = cube.index_of(baseline) if baseline is not None else ci - 1
    if bi < 0 or bi == ci:
        raise ValueError("baseline and current must be two different periods")

    n0, d0 = cube.state(bi)
    n1, d1 = cube.state(ci)
    base_value = metric.value(n0, d0)
    current_value = metric.value(n1, d1)
    total = current_value - base_value

    noise = _Noise(cube, bi, ci, history)
    found = _discover(cube, (n0, d0, n1, d1), total, noise, max_causes=max_causes,
                      max_depth=max_depth, drill=drill, min_share=min_share, noise_z=noise_z,
                      materiality=materiality, search_depth=search_depth)
    state = (n0, d0, n1, d1)
    threshold = max(min_share * abs(total), materiality * noise.total_sd(), 1e-300)
    while True:
        impacts, counterfactual, loo = _shapley(cube, state, [(s, mc) for s, mc, _ in found])
        # Explaining away. A cause must (a) keep a fair share pointing the same
        # way as the effect that got it discovered, and (b) still matter once
        # every other cause is neutralized; composition side-effects of another
        # cause fail (b). Remove only the weakest offender per pass, because a
        # parent and child describing one incident each look redundant.
        bad = [i for i, (_, _, ev) in enumerate(found)
               if np.sign(impacts[i]) != np.sign(ev[2]) or abs(impacts[i]) < threshold
               or np.sign(loo[i]) != np.sign(ev[2]) or abs(loo[i]) < threshold]
        if not bad:
            break
        worst = min(bad, key=lambda i: abs(loo[i]) if np.sign(loo[i]) == np.sign(found[i][2][2]) else -1)
        found = [f for i, f in enumerate(found) if i != worst]

    causes = []
    for (segment, mech, (z, how, _)), impact in zip(found, impacts):
        m = cube.mask(segment)
        comps = metric.components(n0, d0, n1, d1, m)
        causes.append(Cause(segment=dict(segment), mechanism=mech, impact=impact,
                            share=impact / total if total else float("nan"),
                            values=metric.segment_values(n0, d0, n1, d1, m),
                            components=comps, z=z, z_method=how))
    causes.sort(key=lambda c: -abs(c.impact))

    return Explanation(metric=metric, baseline=str(cube.periods[bi]),
                       current=str(cube.periods[ci]), base_value=base_value,
                       current_value=current_value, causes=causes,
                       residual=counterfactual - base_value, dimensions=list(dimensions),
                       history_periods=noise.pairs, total_z=noise.total_z())


# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------

def _apply(metric, state, causes, cube):
    n0, d0, n1, d1 = state
    for segment, mech in causes:
        n1, d1 = metric.neutralize(n0, d0, n1, d1, cube.mask(segment), mech)
    return n1, d1


def _critical(tests: int, df: int | None, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni critical value; Student-t (Fisher expansion) for small samples."""
    z = NormalDist().inv_cdf(1 - alpha / (2 * max(tests, 1)))
    if not df:
        return z
    v = float(df)
    return (z + (z ** 3 + z) / (4 * v) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * v ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * v ** 3))


def _intensity(metric, mech, n0, d0, n1, d1, m) -> float:
    """How strongly a segment moved, independent of its size (|log ratio|)."""
    def lr(a, b):
        if a <= 0 or b <= 0:
            return math.inf if a != b else 0.0
        return abs(math.log(b / a))
    if mech == "change":
        return lr(abs(n0[m].sum()), abs(n1[m].sum()))
    dm0, dm1 = d0[m].sum(), d1[m].sum()
    if mech == "mix":
        return lr(dm0 / d0.sum(), dm1 / d1.sum())
    if dm0 == 0 or dm1 == 0:
        return 0.0
    return lr(n0[m].sum() / dm0, n1[m].sum() / dm1)


def _effect(metric, n0, d0, n1, d1, m, mech) -> float:
    """Counterfactual effect: how much of the current value disappears if this
    segment had kept its baseline rate (or share). Unlike the Shapley mix term,
    this credits a traffic drop in an average segment for the composition
    shift it causes everywhere else."""
    nn, dd = metric.neutralize(n0, d0, n1, d1, m, mech)
    return metric.value(n1, d1) - metric.value(nn, dd)


def _dominant(metric, n0, d0, n1, d1, m):
    comps = metric.components(n0, d0, n1, d1, m)
    mech = max(comps, key=lambda k: abs(comps[k]))
    return mech, comps[mech]


def _drill(metric, cube, state, segment, m, mech, c, max_depth, drill, intensity_gain):
    """Refine a segment while a single, more intense child carries most of its effect."""
    n0, d0, n1, d1 = state
    while len(segment) < max_depth:
        child = None
        here = _intensity(metric, mech, n0, d0, n1, d1, m)
        for dim in cube.dimensions:
            if dim in segment:
                continue
            for val in pd.unique(cube.codes[dim][m]):
                cm = m & (cube.codes[dim] == val)
                if cm.sum() == m.sum():
                    continue
                cc = _effect(metric, n0, d0, n1, d1, cm, mech)
                # Most of the effect AND clearly more intense: a campaign that
                # tripled a whole channel must not be blamed on its largest slice.
                if (np.sign(cc) == np.sign(c) and abs(cc) >= drill * abs(c)
                        and _intensity(metric, mech, n0, d0, n1, d1, cm) >= intensity_gain * here):
                    if child is None or abs(cc) > abs(child[2]):
                        child = ({**segment, dim: val}, cm, cc)
        if child is None:
            break
        segment, m, c = child
    return segment, m, c


def _discover(cube: Cube, state, total, noise, *, max_causes, max_depth, drill,
              min_share, noise_z, materiality, search_depth=2, intensity_gain=1.25):
    metric = cube.metric
    n0, d0, _, _ = state
    # A cause must move the headline metric noticeably, not merely be detectable.
    floor = materiality * noise.total_sd()
    scale = abs(total)
    if scale == 0:
        # Offsetting changes: fall back to the gross movement across top segments.
        n1, d1 = state[2], state[3]
        scale = max((abs(_dominant(metric, n0, d0, n1, d1, cube.codes[d] == v)[1])
                     for d in cube.dimensions for v in cube.values[d]), default=0.0)
    if scale == 0:
        return []

    # Two-tier evidence gate. If the overall change is itself unusual, we are
    # explaining a real event and material causes only need |z| >= noise_z.
    # Otherwise (or for small effects) every candidate must survive a
    # search-wide multiple-comparison correction.
    z_total = noise.total_z()
    unusual = z_total is None or abs(z_total) >= noise_z
    tests = None  # set once the candidate pool is known

    # Candidate segments: every value of every dimension, plus every observed
    # combination of two dimensions (so a surge in `ios · new` is found
    # directly rather than leaking composition effects through its parents).
    pool = []
    for depth in range(1, min(search_depth, len(cube.dimensions)) + 1):
        for dims in combinations(cube.dimensions, depth):
            combos = cube.segments[list(dims)].drop_duplicates().itertuples(index=False, name=None)
            for vals in combos:
                m = cube.mask(dict(zip(dims, vals)))
                if not m.all():
                    pool.append((dict(zip(dims, vals)), m))

    tests = len(pool) * len(metric.mechanisms)
    strict = max(noise_z, _critical(tests, noise.pairs - 1 if noise.pairs >= 3 else None))

    causes: list[tuple[dict, str]] = []
    rejected: set = set()
    evidence: dict = {}

    def nested(segment, mech):
        # a refinement of something we already explained is the same incident
        return any(mech == cm and all(segment.get(k) == v for k, v in cs.items())
                   for cs, cm in causes)

    while len(causes) < max_causes:
        n1, d1 = _apply(metric, state, causes, cube)

        candidates = []
        for segment, m in pool:
            for mech in metric.mechanisms:
                c = _effect(metric, n0, d0, n1, d1, m, mech)
                if abs(c) >= max(min_share * scale, floor):
                    candidates.append((dict(segment), m, mech, c))
        # Biggest effect first; among (near-)equal effects, e.g. the two halves
        # of a binary dimension, prefer the more intense movement.
        candidates.sort(key=lambda t: (-round(abs(t[3]) / scale, 3),
                                       -_intensity(metric, t[2], n0, d0, n1, d1, t[1])))
        if not causes and candidates:
            scale = max(scale, abs(candidates[0][3]))

        chosen = None
        for segment, m, mech, c in candidates:
            segment, m, c = _drill(metric, cube, (n0, d0, n1, d1), segment, m, mech, c,
                                   max_depth, drill, intensity_gain)
            key = (tuple(sorted(segment.items())), mech)
            if key in rejected or nested(segment, mech):
                continue
            z, how = noise.score(m, mech, observed=c)
            # Dozens of segments are searched, so small effects need stronger evidence.
            needed = noise_z if unusual and abs(c) >= 0.15 * scale else strict
            if z is not None and abs(z) < needed:
                rejected.add(key)
                continue
            chosen = ({d: segment[d] for d in cube.dimensions if d in segment}, mech)
            evidence[key] = (z, how, c)
            break
        if chosen is None:
            break
        causes.append(chosen)
    return [(c, m, evidence[(tuple(sorted(c.items())), m)]) for c, m in causes]


# ----------------------------------------------------------------------------
# attribution
# ----------------------------------------------------------------------------

def _shapley(cube: Cube, state, causes):
    """Exact Shapley values of the neutralization game (k ≤ ~10 is instant)."""
    metric = cube.metric
    k = len(causes)
    n0, d0, n1, d1 = state
    current = metric.value(n1, d1)
    if k == 0:
        return [], current, []

    cache: dict[tuple, float] = {}

    def v(subset: tuple) -> float:
        if subset not in cache:
            nn, dd = _apply(metric, state, [causes[i] for i in subset], cube)
            cache[subset] = metric.value(nn, dd)
        return cache[subset]

    phi = [0.0] * k
    for i in range(k):
        others = [j for j in range(k) if j != i]
        for size in range(k):
            weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for s in combinations(others, size):
                with_i = tuple(sorted((*s, i)))
                phi[i] += weight * (v(s) - v(with_i))
    everything = tuple(range(k))
    # leave-one-out: what cause i still explains once all the others are removed
    loo = [v(tuple(j for j in everything if j != i)) - v(everything) for i in range(k)]
    return phi, v(everything), loo


# ----------------------------------------------------------------------------
# noise
# ----------------------------------------------------------------------------

class _Noise:
    def __init__(self, cube: Cube, bi: int, ci: int, history: int):
        self.cube = cube
        self.bi, self.ci = bi, ci
        lo = max(0, bi - history)
        self.history_pairs = [(i, i + 1) for i in range(lo, bi)]
        self.pairs = len(self.history_pairs)
        self.counts = cube.is_count_data()

    def _moves(self) -> list[float]:
        metric = self.cube.metric
        return [metric.value(*self.cube.state(b)) - metric.value(*self.cube.state(a))
                for a, b in self.history_pairs]

    def total_sd(self) -> float:
        """Typical period-to-period movement of the headline metric."""
        return float(np.std(self._moves(), ddof=1)) if self.pairs >= 3 else 0.0

    def total_z(self):
        """How unusual the headline change is compared with earlier periods."""
        metric = self.cube.metric
        observed = metric.value(*self.cube.state(self.ci)) - metric.value(*self.cube.state(self.bi))
        if self.pairs >= 3:
            moves = self._moves()
            sd = np.std(moves, ddof=1)
            if sd > 0:
                return float((observed - np.mean(moves)) / sd)
        everything = np.ones(len(self.cube.segments), dtype=bool)
        return self.score(everything, metric.mechanisms[0])[0]

    def score(self, m: np.ndarray, mech: str, observed: float | None = None):
        cube, metric = self.cube, self.cube.metric
        n0, d0 = cube.state(self.bi)
        n1, d1 = cube.state(self.ci)
        if observed is None:
            observed = _effect(metric, n0, d0, n1, d1, m, mech)

        if self.pairs >= 3:
            hist = []
            for a, b in self.history_pairs:
                na, da = cube.state(a)
                nb, db = cube.state(b)
                hist.append(_effect(metric, na, da, nb, db, m, mech))
            hist = np.asarray(hist)
            sd = hist.std(ddof=1)
            if sd > 0:
                return float((observed - hist.mean()) / sd), f"vs {self.pairs} prior periods"

        if not self.counts:
            return None, None
        if isinstance(metric, Sum):
            a, b = n0[m].sum(), n1[m].sum()
            return (float((b - a) / math.sqrt(a + b)) if a + b > 0 else None), "Poisson test"
        dm0, dm1, nm0, nm1 = d0[m].sum(), d1[m].sum(), n0[m].sum(), n1[m].sum()
        if mech == "rate":
            if dm0 == 0 or dm1 == 0 or np.any(n0 > d0) or np.any(n1 > d1):
                return None, None
            p = (nm0 + nm1) / (dm0 + dm1)
            se = math.sqrt(p * (1 - p) * (1 / dm0 + 1 / dm1)) if 0 < p < 1 else 0
            return (float((nm1 / dm1 - nm0 / dm0) / se) if se else None), "two-proportion test"
        D0, D1 = d0.sum(), d1.sum()
        w0, w1 = dm0 / D0, dm1 / D1
        p = (dm0 + dm1) / (D0 + D1)
        se = math.sqrt(p * (1 - p) * (1 / D0 + 1 / D1)) if 0 < p < 1 else 0
        # sign follows the effect on the metric, not on the share
        sign = np.sign(observed) * np.sign(w1 - w0) if observed else 1.0
        return (float(sign * (w1 - w0) / se) if se else None), "two-proportion test (share)"
