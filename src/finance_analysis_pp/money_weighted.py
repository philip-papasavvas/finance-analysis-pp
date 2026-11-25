"""Money-weighted returns: what the investor earned, given when they paid in."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd


def xirr(flows: pd.Series, *, period_days: float = 365.25) -> float | None:
    """Internal rate of return of dated cash flows, per ``period_days``.

    Flows are signed from the investor's side: money paid in is negative, and money
    received, including the final valuation, is positive.

    Some flows have more than one rate that sets their present value to zero. No
    single answer is right for those, so the result is ``None``, as it is when no
    rate between -99.9% and 1,000,000 (100,000,000%) does.

    Pass the span of the flows as ``period_days`` for the rate over the whole period.
    """
    if not isinstance(flows.index, pd.DatetimeIndex):
        raise TypeError("flows must be indexed by date")
    if not np.isfinite(flows.to_numpy(dtype=float)).all() or flows.index.hasnans:
        raise ValueError("flows and their dates must be finite")
    flows = flows.groupby(level=0).sum().sort_index()
    # A leading zero would make every present value vanish at very high rates,
    # which looks like a root. Dropping zeros does not move a real one.
    flows = flows[flows != 0]
    values = flows.to_numpy(dtype=float)
    if len(values) < 2 or period_days <= 0 or not (values.min() < 0 < values.max()):
        return None

    dates = pd.DatetimeIndex(flows.index)
    years = (dates - dates[0]).days.to_numpy() / period_days
    values = values / np.abs(values).max()
    grid = np.expm1(np.linspace(np.log(0.001), np.log(1_000_001), 500))
    npvs = values @ np.exp(-years[:, None] * np.log1p(grid)[None, :])
    on_grid = np.flatnonzero(np.abs(npvs) < 1e-12)
    crossings = np.flatnonzero(npvs[:-1] * npvs[1:] < 0)
    crossings = np.array([i for i in crossings if i not in on_grid and i + 1 not in on_grid], int)
    if len(on_grid) + len(crossings) != 1:
        return None
    if len(on_grid):
        return float(grid[on_grid[0]])

    low, high = float(grid[crossings[0]]), float(grid[crossings[0] + 1])
    npv_low = float(npvs[crossings[0]])
    for _ in range(100):
        middle = (low + high) / 2
        if middle in (low, high):
            break
        npv_middle = float(np.sum(values / np.power(1 + middle, years)))
        if npv_low * npv_middle <= 0:
            high = middle
        else:
            low, npv_low = middle, npv_middle
    return (low + high) / 2


def modified_dietz(
    opening: float,
    closing: float,
    flows: Iterable[tuple[date, float]],
    start: date,
    end: date,
    *,
    minimum_base: float = 0.0,
) -> float | None:
    """Modified Dietz return over ``start`` to ``end``, not annualised.

    Here money paid in is positive. That is the opposite of :func:`xirr`, because
    flows are read from the portfolio's side.

    The result is ``None`` when the average capital is not positive or is below
    ``minimum_base``: a percentage of almost nothing is not a return.
    """
    span = (end - start).days
    if span <= 0:
        return None
    flows = list(flows)
    if any(not start <= when <= end for when, _ in flows):
        raise ValueError("every flow must fall between start and end")
    net = sum(amount for _, amount in flows)
    base = opening + sum(amount * (end - when).days / span for when, amount in flows)
    if base <= 0 or base < minimum_base:
        return None
    return (closing - opening - net) / base
