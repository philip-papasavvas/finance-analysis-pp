"""Put traded units on the same basis as split-adjusted prices.

Price feeds restate history after a split; a trade confirmation does not. Splits are
declared here from the issuer's announcement and never inferred from prices. A trade
that is exactly 100 times the stored price is more often pounds read as pence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Split:
    """``ratio`` new units for each old one, on and after ``effective``."""

    effective: date
    ratio: float

    def __post_init__(self) -> None:
        if self.ratio <= 0:
            raise ValueError("a split ratio must be positive")


def adjusted_units(units: float, traded: date, splits: Iterable[Split]) -> float:
    """``units`` as traded, restated for every split that came afterwards.

    A trade on the effective date is already on the new basis.
    """
    for split in splits:
        if traded < split.effective:
            units *= split.ratio
    return units
