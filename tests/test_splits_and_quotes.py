from datetime import date

import pandas as pd
import pytest

from finance_analysis_pp import Split, adjusted_units, panel_to_gbp, to_gbp

TEN_FOR_ONE = Split(date(2022, 6, 1), 10)
HALVING = Split(date(2024, 6, 1), 0.5)


@pytest.mark.parametrize(
    ("traded", "expected"),
    [
        (date(2022, 5, 31), 50),
        (date(2022, 6, 1), 5),
        (date(2024, 5, 31), 5),
        (date(2024, 6, 1), 10),
    ],
)
def test_only_later_splits_restate_a_trade(traded, expected):
    assert adjusted_units(10, traded, [HALVING, TEN_FOR_ONE]) == pytest.approx(expected)


def test_no_splits_leaves_units_alone():
    assert adjusted_units(10, date(2020, 1, 1), []) == 10


@pytest.mark.parametrize("ratio", [0, -1])
def test_a_ratio_must_be_positive(ratio):
    with pytest.raises(ValueError):
        Split(date(2024, 1, 1), ratio)


def test_pence_pounds_and_foreign_prices():
    assert to_gbp(850, "GBX") == pytest.approx(8.5)
    assert to_gbp(850, "GBP") == 850
    assert to_gbp(200, "USD", {"USD": 0.75}) == pytest.approx(150)


def test_a_missing_rate_is_never_treated_as_pounds():
    with pytest.raises(ValueError):
        to_gbp(200, "USD")


def test_panel_converts_each_column_by_its_own_currency():
    prices = pd.DataFrame({"PENCE": [850.0, 900.0], "POUNDS": [106.0, 107.0]})
    result = panel_to_gbp(prices, {"PENCE": "GBX", "POUNDS": "GBP"})
    assert result["PENCE"].tolist() == pytest.approx([8.5, 9.0])
    assert result["POUNDS"].tolist() == [106.0, 107.0]


def test_panel_refuses_a_ticker_with_no_stated_currency():
    with pytest.raises(ValueError, match="MYSTERY"):
        panel_to_gbp(pd.DataFrame({"MYSTERY": [1.0]}), {})
