"""Metric definitions.

A metric knows three things about itself:

* how to compute its value from segment-level arrays,
* how to split one segment's share of a change into *mix* and *rate* effects,
* how to *neutralize* a segment, i.e. build the counterfactual in which that
  segment behaved in the current period exactly as it did in the baseline.

Everything else in whyline is metric-agnostic and talks to these methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def _fmt_number(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if ax >= 1e4:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.1f}"
    if ax >= 1:
        return f"{x:,.2f}"
    return f"{x:.4g}"


@dataclass(frozen=True)
class Sum:
    """An additive metric such as revenue, orders or signups.

    Mechanisms: a segment can only *change* (there is no mix for sums).
    """

    column: str
    name: str | None = None

    kind = "sum"

    @property
    def label(self) -> str:
        return self.name or self.column

    @property
    def columns(self) -> list[str]:
        return [self.column]

    mechanisms = ("change",)

    # -- core maths ---------------------------------------------------------
    def value(self, n: Array, d: Array | None) -> float:
        return float(n.sum())

    def components(self, n0, d0, n1, d1, m) -> dict[str, float]:
        return {"change": float(n1[m].sum() - n0[m].sum())}

    def neutralize(self, n0, d0, n1, d1, m, mechanism):
        """Every finest-grain cell in the segment returns to its baseline value."""
        n1 = n1.copy()
        n1[m] = n0[m]
        return n1, d1

    def segment_values(self, n0, d0, n1, d1, m) -> dict[str, float]:
        return {"base": float(n0[m].sum()), "current": float(n1[m].sum())}

    # -- formatting ---------------------------------------------------------
    def fmt(self, x: float) -> str:
        return _fmt_number(x)

    def fmt_delta(self, x: float) -> str:
        return ("+" if x >= 0 else "−") + _fmt_number(abs(x))


@dataclass(frozen=True)
class Ratio:
    """A ratio of two additive columns: conversion rate, ARPU, CTR, AOV...

    ``R = Σ n_i / Σ d_i = Σ w_i · r_i`` with ``w_i = d_i / D`` (the segment's
    share of the denominator) and ``r_i = n_i / d_i`` (its own rate).

    A segment moves R in two ways:

    * **mix**  – its share ``w`` changed (who showed up),
    * **rate** – its own ratio ``r`` changed (how they behaved).

    The split is the exact two-player Shapley value of the product ``w·r``,
    centred on the overall average so that a segment growing only matters if
    it is better or worse than average. Contributions are exactly additive:
    over any partition of the data they sum to ``R₁ − R₀``.
    """

    numerator: str
    denominator: str
    name: str | None = None
    percent: bool = False

    kind = "ratio"
    mechanisms = ("rate", "mix")

    @property
    def label(self) -> str:
        return self.name or f"{self.numerator} / {self.denominator}"

    @property
    def columns(self) -> list[str]:
        return [self.numerator, self.denominator]

    # -- core maths ---------------------------------------------------------
    def value(self, n: Array, d: Array) -> float:
        D = d.sum()
        return float(n.sum() / D) if D else float("nan")

    def _parts(self, n0, d0, n1, d1, m):
        D0, D1 = d0.sum(), d1.sum()
        dm0, dm1 = d0[m].sum(), d1[m].sum()
        nm0, nm1 = n0[m].sum(), n1[m].sum()
        w0, w1 = dm0 / D0, dm1 / D1
        # A segment that is new (or gone) has no rate in one period; treat
        # its rate as unchanged so that its whole effect is mix.
        r1 = nm1 / dm1 if dm1 > 0 else (nm0 / dm0 if dm0 > 0 else 0.0)
        r0 = nm0 / dm0 if dm0 > 0 else r1
        R0, R1 = n0.sum() / D0, n1.sum() / D1
        return w0, w1, r0, r1, R0, R1

    def components(self, n0, d0, n1, d1, m) -> dict[str, float]:
        if d0[m].sum() == 0 and d1[m].sum() == 0:
            return {"rate": 0.0, "mix": 0.0}
        w0, w1, r0, r1, R0, R1 = self._parts(n0, d0, n1, d1, m)
        rbar = (R0 + R1) / 2
        mix = (w1 - w0) * ((r0 + r1) / 2 - rbar)
        rate = (r1 - r0) * (w0 + w1) / 2
        return {"rate": float(rate), "mix": float(mix)}

    def neutralize(self, n0, d0, n1, d1, m, mechanism):
        """Counterfactual applied cell by cell at the finest grain.

        * ``rate``: every cell keeps its current volume but gets back its own
          baseline rate. Composition shifts inside the segment are therefore
          *not* mistaken for behaviour change (no Simpson's paradox).
        * ``mix``: the segment's volume returns to its baseline share and
          baseline internal composition; each cell keeps its current rate.

        The two commute, and overlapping causes do not double-count.
        """
        n1, d1 = n1.copy(), d1.copy()
        if mechanism == "rate":
            rows = m & (d0 > 0) & (d1 > 0)
            n1[rows] = d1[rows] * (n0[rows] / d0[rows])
            return n1, d1

        dm0 = d0[m].sum()
        w0 = dm0 / d0.sum()
        rest = d1.sum() - d1[m].sum()
        if dm0 == 0:                       # segment is new: remove it
            n1[m] = 0.0
            d1[m] = 0.0
            return n1, d1
        if rest <= 0 or w0 >= 1:           # segment is everything: nothing to shift
            return n1, d1
        k = w0 * rest / (1 - w0) / dm0
        cur, base = d1[m] > 0, d0[m] > 0
        rate = np.where(cur, n1[m] / np.where(cur, d1[m], 1.0),
                        np.where(base, n0[m] / np.where(base, d0[m], 1.0), 0.0))
        d1[m] = d0[m] * k
        n1[m] = d1[m] * rate
        return n1, d1

    def segment_values(self, n0, d0, n1, d1, m) -> dict[str, float]:
        w0, w1, r0, r1, R0, R1 = self._parts(n0, d0, n1, d1, m)
        return {
            "base": float(r0), "current": float(r1),
            "base_share": float(w0), "current_share": float(w1),
            "overall_base": float(R0), "overall_current": float(R1),
        }

    # -- formatting ---------------------------------------------------------
    def fmt(self, x: float) -> str:
        return f"{x * 100:.2f}%" if self.percent else _fmt_number(x)

    def fmt_delta(self, x: float) -> str:
        sign = "+" if x >= 0 else "−"
        if self.percent:
            return f"{sign}{abs(x) * 100:.2f} pp"
        return sign + _fmt_number(abs(x))
