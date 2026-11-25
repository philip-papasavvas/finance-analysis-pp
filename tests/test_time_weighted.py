import numpy as np
import pandas as pd
import pytest

from finance_analysis_pp import period_returns, return_index, time_weighted_return

DAYS = pd.date_range("2024-01-01", periods=4, freq="D")


def series(*values: float) -> pd.Series:
    return pd.Series(values, index=DAYS[: len(values)], dtype=float)


NOTHING = pd.Series(dtype=float, index=pd.DatetimeIndex([]))


def test_without_flows_it_is_the_price_return():
    assert time_weighted_return(series(100, 110, 99), NOTHING) == pytest.approx(-0.01)


def test_paying_in_does_not_change_the_return():
    """The asset gains 10% then 10%; 1,000 arrives after the first gain."""
    values = series(100, 1110, 1221)
    assert time_weighted_return(values, series(0, 1000, 0)) == pytest.approx(0.21)


def test_taking_out_does_not_change_the_return():
    assert time_weighted_return(series(100, 60, 66), series(0, -50, 0)) == pytest.approx(0.21)


def test_selling_everything_and_returning_later():
    values, flows = series(100, 0, 50, 55), series(0, -110, 50, 0)
    assert time_weighted_return(values, flows) == pytest.approx(0.21)


def test_a_total_loss_is_minus_one():
    assert time_weighted_return(series(100, 0), NOTHING) == pytest.approx(-1)


def test_several_flows_on_one_day_are_added():
    flows = pd.Series([400.0, 600.0], index=[DAYS[1], DAYS[1]])
    assert time_weighted_return(series(100, 1110, 1221), flows) == pytest.approx(0.21)


@pytest.mark.parametrize(
    ("values", "flows"),
    [
        (series(100, np.nan, 110), NOTHING),
        (series(100, -5), NOTHING),
        (series(100, 110), series(0, np.nan)),
        (series(0, 10), NOTHING),
        (pd.Series([100.0, 110.0], index=[DAYS[0], DAYS[0]]), NOTHING),
        (series(100, 110), pd.Series([5.0], index=[pd.Timestamp("2024-02-01")])),
    ],
)
def test_input_that_would_give_a_wrong_number_raises(values, flows):
    with pytest.raises(ValueError):
        time_weighted_return(values, flows)


def test_index_starts_from_the_base():
    index = return_index(period_returns(series(100, 110, 99), NOTHING))
    assert index.tolist() == pytest.approx([110, 99])


def test_distributing_and_accumulating_funds_with_the_same_assets_match():
    """Both earn 10% then 10%; one pays half of the first gain out."""
    accumulating = time_weighted_return(series(100, 110, 121), NOTHING)
    distributing = time_weighted_return(series(100, 105, 115.5), NOTHING, series(0, 5, 0))
    assert distributing == pytest.approx(accumulating)


def test_ignoring_the_dividend_understates_the_return():
    values, dividend = series(100, 105, 115.5), series(0, 5, 0)
    assert time_weighted_return(values, NOTHING) < time_weighted_return(values, NOTHING, dividend)


def test_reinvesting_the_dividend_gives_the_same_return():
    """The 5 is paid out, then paid straight back in."""
    values, dividend = series(100, 110, 121), series(0, 5, 0)
    result = time_weighted_return(values, series(0, 5, 0), dividend)
    assert result == pytest.approx(0.21)


def test_income_on_a_day_without_a_valuation_raises():
    with pytest.raises(ValueError, match="income"):
        late = pd.Series([5.0], index=[pd.Timestamp("2024-02-01")])
        time_weighted_return(series(100, 110), NOTHING, late)
