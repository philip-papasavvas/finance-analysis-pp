"""Time-weighted returns: what the portfolio earned, whenever money arrived."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _by_date(amounts: pd.Series, values: pd.Series, name: str) -> pd.Series:
    if not np.isfinite(amounts.to_numpy(dtype=float)).all():
        raise ValueError(f"every {name} must be finite")
    missing = amounts.index.difference(values.index)
    if len(missing):
        raise ValueError(f"no valuation on the date of a {name}: {missing[0]}")
    return amounts.groupby(level=0).sum().reindex(values.index, fill_value=0.0)


def period_returns(
    values: pd.Series, flows: pd.Series, income: pd.Series | None = None
) -> pd.Series:
    """Return for each date in ``values`` after the first.

    ``values`` are end-of-day valuations. ``flows`` are money paid in (positive) or
    taken out (negative), already included in that day's valuation, so a flow earns
    nothing until the next day. A flow on a date with no valuation is an error
    rather than silently counted as a gain.

    ``income`` is cash the holding paid out, such as the dividend of a distributing fund,
    and is not in the valuation. Without it, a distributing fund looks worse than an
    accumulating fund with the same assets.
    """
    values = values.sort_index()
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("every date needs a valuation")
    if income is None:
        income = pd.Series(dtype=float, index=values.index[:0])
    if (values < 0).any() or values.index.has_duplicates:
        raise ValueError("valuations must be non-negative, one per date")
    moved = _by_date(flows, values, "flow")
    earned = _by_date(income, values, "payment of income")
    before = values.shift(1).iloc[1:]
    grown = (values - moved + earned).iloc[1:]
    empty = before == 0
    if (grown[empty].abs() > 1e-9 * values.max()).any():
        raise ValueError("a gain appeared on a portfolio that held nothing")
    return (grown / before.where(~empty) - 1).fillna(0.0)


def time_weighted_return(
    values: pd.Series, flows: pd.Series, income: pd.Series | None = None
) -> float:
    """Chain-linked return over the whole series, not annualised."""
    growth = (1 + period_returns(values, flows, income)).to_numpy(dtype=float)
    return float(np.prod(growth)) - 1


def return_index(returns: pd.Series, *, base: float = 100.0) -> pd.Series:
    """Cumulative value of ``base`` invested before the first of ``returns``."""
    return (1 + returns).cumprod() * base
