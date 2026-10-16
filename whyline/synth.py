"""Synthetic e-commerce funnel data with planted incidents.

Because the ground truth is known, this doubles as the demo dataset and the
benchmark harness: plant causes, run whyline, check what it recovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

# name: (values, traffic weights, conversion multipliers, avg order value)
DIMENSIONS: dict[str, tuple[list[str], list[float], list[float]]] = {
    "country": (["IN", "US", "BR", "DE", "ID", "GB"],
                [0.32, 0.22, 0.16, 0.10, 0.12, 0.08],
                [0.85, 1.25, 0.90, 1.15, 0.80, 1.20]),
    "device": (["android", "ios", "desktop"],
               [0.50, 0.25, 0.25],
               [0.80, 1.15, 1.30]),
    "channel": (["organic", "paid_search", "paid_social", "email", "referral"],
                [0.38, 0.22, 0.12, 0.13, 0.15],
                [1.00, 1.15, 0.45, 1.60, 1.10]),
    "user_type": (["new", "returning"],
                  [0.60, 0.40],
                  [0.70, 1.50]),
}
AOV = {"IN": 18.0, "US": 74.0, "BR": 31.0, "DE": 66.0, "ID": 15.0, "GB": 61.0}


@dataclass
class Incident:
    """Something that happened in the current period."""
    segment: dict[str, str]
    rate: float = 1.0      # multiplier on the segment's conversion rate
    volume: float = 1.0    # multiplier on the segment's traffic
    label: str = ""

    @property
    def mechanism(self) -> str:
        return "rate" if abs(np.log(self.rate)) >= abs(np.log(self.volume)) else "mix"


@dataclass
class Dataset:
    frame: pd.DataFrame
    incidents: list[Incident] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=lambda: list(DIMENSIONS))


def make_funnel(weeks: int = 10, sessions_per_week: float = 400_000, base_rate: float = 0.03,
                incidents: list[Incident] | None = None, noise: float = 0.03,
                seed: int | None = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    names = list(DIMENSIONS)
    combos = list(product(*(DIMENSIONS[d][0] for d in names)))

    weight = np.array([np.prod([DIMENSIONS[d][1][DIMENSIONS[d][0].index(v)]
                                for d, v in zip(names, c)]) for c in combos])
    rate = np.array([base_rate * np.prod([DIMENSIONS[d][2][DIMENSIONS[d][0].index(v)]
                                          for d, v in zip(names, c)]) for c in combos])
    weight *= rng.lognormal(0, 0.15, len(combos))
    weight /= weight.sum()
    rate *= rng.lognormal(0, 0.10, len(combos))
    aov = np.array([AOV[c[0]] for c in combos]) * rng.lognormal(0, 0.1, len(combos))

    incidents = incidents or []
    frames = []
    for w in range(weeks):
        week = f"2026-W{30 + w:02d}"
        trend = 1 + 0.01 * w
        vol = weight * trend * rng.lognormal(0, noise, len(combos))
        r = rate * rng.lognormal(0, noise / 2, len(combos)) * rng.lognormal(0, noise / 3)
        if w == weeks - 1:
            for inc in incidents:
                hit = np.array([all(dict(zip(names, c)).get(k) == v for k, v in inc.segment.items())
                                for c in combos])
                vol = np.where(hit, vol * inc.volume, vol)
                r = np.where(hit, r * inc.rate, r)
        sessions = rng.poisson(vol * sessions_per_week)
        conversions = rng.binomial(sessions, np.clip(r, 0, 1))
        revenue = np.round(conversions * aov * rng.lognormal(0, 0.05, len(combos)), 2)
        frames.append(pd.DataFrame({"week": week, **{d: [c[i] for c in combos] for i, d in enumerate(names)},
                                    "sessions": sessions, "conversions": conversions,
                                    "revenue": revenue}))
    return Dataset(pd.concat(frames, ignore_index=True), incidents, names)


def demo(seed: int | None = 7) -> Dataset:
    """The story used in the README: one bug, one campaign, one promotion."""
    return make_funnel(seed=seed, incidents=[
        Incident({"country": "IN", "device": "android"}, rate=0.45,
                 label="payment-gateway outage for Android users in India"),
        Incident({"channel": "paid_social"}, volume=2.6,
                 label="new paid-social campaign brought low-intent traffic"),
        Incident({"country": "US", "user_type": "returning"}, rate=1.25,
                 label="loyalty promotion for returning US customers"),
    ])
