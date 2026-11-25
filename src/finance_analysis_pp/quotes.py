"""Convert quoted prices to pounds.

London-listed shares and funds may be quoted in pence (GBX) or pounds (GBP), and a
price does not say which. Guessing from its size misvalues a holding by 100 times, so
every function here needs the quote currency to be stated.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def _factor(currency: str, rates: Mapping[str, float]) -> float:
    if currency == "GBP":
        return 1.0
    if currency == "GBX":
        return 0.01
    if currency not in rates:
        raise ValueError(f"no rate to convert {currency} to GBP")
    return rates[currency]


def to_gbp(price: float, currency: str, rates: Mapping[str, float] | None = None) -> float:
    """``price`` in pounds. ``rates`` holds pounds per unit of each foreign currency."""
    return price * _factor(currency, rates or {})


def panel_to_gbp(
    prices: pd.DataFrame,
    currencies: Mapping[str, str],
    rates: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """A panel of prices, one column per ticker, in pounds."""
    unknown = [str(ticker) for ticker in prices.columns if ticker not in currencies]
    if unknown:
        raise ValueError(f"no quote currency for {', '.join(unknown)}")
    factors = {ticker: _factor(currencies[ticker], rates or {}) for ticker in prices.columns}
    return prices * pd.Series(factors, dtype=float)
