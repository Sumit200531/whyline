"""A tiny columnar cube: one row per finest-grain segment, one column per period.

All analysis runs on boolean masks over these arrays, so a segment such as
``{device: android, country: IN}`` is just ``code[device] == a & code[country] == b``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .metrics import Ratio, Sum


class Cube:
    def __init__(self, df: pd.DataFrame, metric: Sum | Ratio,
                 dimensions: Sequence[str], time: str):
        missing = [c for c in [time, *dimensions, *metric.columns] if c not in df.columns]
        if missing:
            raise KeyError(f"columns not found in data: {missing}")
        if not dimensions:
            raise ValueError("at least one dimension is required")

        self.metric = metric
        self.dimensions = list(dimensions)
        self.time = time

        data = df[[time, *self.dimensions, *metric.columns]].copy()
        for dim in self.dimensions:
            data[dim] = data[dim].astype("string").fillna("(missing)")
        for col in metric.columns:
            data[col] = pd.to_numeric(data[col], errors="raise").astype(float)

        self.periods = sorted(data[time].dropna().unique().tolist())
        grouped = data.groupby([*self.dimensions, time], observed=True)[metric.columns].sum()
        wide = grouped.unstack(time, fill_value=0.0)

        self.segments = wide.index.to_frame(index=False)
        self.codes = {d: self.segments[d].to_numpy(dtype=object) for d in self.dimensions}
        self.values = {d: sorted(pd.unique(self.codes[d])) for d in self.dimensions}

        def matrix(col: str) -> np.ndarray:
            return wide[col].reindex(columns=self.periods, fill_value=0.0).to_numpy(float).T.copy()

        if isinstance(metric, Ratio):
            self.N, self.D = matrix(metric.numerator), matrix(metric.denominator)
        else:
            self.N, self.D = matrix(metric.column), None

    # ------------------------------------------------------------------
    def index_of(self, period) -> int:
        for i, p in enumerate(self.periods):
            if p == period or str(p) == str(period):
                return i
        raise KeyError(f"period {period!r} not found; available: {[str(p) for p in self.periods]}")

    def state(self, i: int):
        return self.N[i], (self.D[i] if self.D is not None else None)

    def mask(self, segment: Mapping[str, str]) -> np.ndarray:
        m = np.ones(len(self.segments), dtype=bool)
        for dim, val in segment.items():
            m &= self.codes[dim] == val
        return m

    def is_count_data(self) -> bool:
        arrays = [self.N] + ([self.D] if self.D is not None else [])
        return all(np.all(a >= 0) and np.allclose(a, np.round(a)) for a in arrays)
