from datetime import date

import numpy as np
import pandas as pd
import pytest

from finance_analysis_pp import modified_dietz, xirr


def flows(**amounts: float) -> pd.Series:
    """`d20260101=-100` style cash flows."""
    index = pd.to_datetime([key[1:] for key in amounts], format="%Y%m%d")
    return pd.Series(list(amounts.values()), index=index, dtype=float)


def test_one_year_of_growth():
    result = xirr(flows(d20260101=-100, d20270101=110))
    assert result == pytest.approx(1.1 ** (365.25 / 365) - 1)


def test_the_rate_sets_present_value_to_zero():
    series = flows(d20200110=-500, d20200601=-250, d20211115=120, d20230303=900)
    rate = xirr(series)
    years = (series.index - series.index[0]).days / 365.25
    assert (series / (1 + rate) ** years).sum() == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("scale", [1e-14, 1e-6, 1, 1e9])
def test_the_rate_does_not_depend_on_the_currency_unit(scale):
    series = flows(d20200110=-500, d20200601=-250, d20230303=900)
    assert xirr(series * scale) == pytest.approx(xirr(series))


def test_order_duplicates_and_zero_flows_do_not_matter():
    plain = flows(d20200110=-500, d20230303=900)
    messy = pd.concat(
        [flows(d20230303=400), flows(d20100101=0), flows(d20200110=-500), flows(d20230303=500)]
    )
    assert xirr(messy) == pytest.approx(xirr(plain))


def test_two_valid_rates_are_refused():
    """10% and 20% both solve these flows."""
    assert xirr(flows(d20240101=-100, d20250101=230, d20260101=-132)) is None


@pytest.mark.parametrize(
    "series", [flows(d20240101=-100), flows(d20240101=-1, d20250101=-1), flows()]
)
def test_no_rate_exists(series):
    assert xirr(series) is None


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_unusable_amounts_raise(bad):
    with pytest.raises(ValueError):
        xirr(flows(d20240101=-100, d20250101=bad))


@pytest.mark.parametrize("period", [0, -10])
def test_a_non_positive_period_is_refused(period):
    assert xirr(flows(d20240101=-1, d20240110=2), period_days=period) is None


def test_a_short_sharp_loss_has_a_rate_over_its_own_period():
    series = flows(d20240101=-100, d20240115=70)
    assert xirr(series) is None
    assert xirr(series, period_days=14) == pytest.approx(-0.30)


START, END = date(2021, 1, 1), date(2022, 1, 1)


def test_dietz_matches_a_hand_calculation():
    """Weighted: 3650 x 265/365 - 1825 x 90/365 = 2200. Gain: 13000 - 10000 - 1825."""
    paid = [(date(2021, 4, 11), 3650.0), (date(2021, 10, 3), -1825.0)]
    assert modified_dietz(10_000, 13_000, paid, START, END) == pytest.approx(1175 / 12200)


def test_dietz_without_flows_is_the_simple_return():
    assert modified_dietz(100, 110, [], START, END) == pytest.approx(0.10)


def test_the_same_gain_on_later_money_is_a_higher_return():
    early = modified_dietz(10_000, 16_000, [(date(2021, 1, 2), 5000.0)], START, END)
    late = modified_dietz(10_000, 16_000, [(date(2021, 12, 30), 5000.0)], START, END)
    assert late > early


def test_dietz_refuses_a_base_that_is_empty_negative_or_too_small():
    assert modified_dietz(0, 10, [], START, END) is None
    assert modified_dietz(100, 10, [(START, -500.0)], START, END) is None
    assert modified_dietz(300, 600, [], START, END, minimum_base=1000) is None


def test_dietz_refuses_an_empty_period_and_a_flow_outside_it():
    assert modified_dietz(1, 2, [], START, START) is None
    with pytest.raises(ValueError):
        modified_dietz(100, 110, [(date(2020, 12, 31), 5.0)], START, END)


def test_the_two_methods_agree_on_smooth_contributions():
    months = pd.date_range("2021-01-01", "2021-12-01", freq="MS")
    daily = 0.06 / 365
    closing = sum(100 * (1 + daily * (pd.Timestamp(END) - m).days) for m in months)
    series = pd.Series(-100.0, index=months)
    series[pd.Timestamp(END)] = closing
    paid = [(m.date(), 100.0) for m in months]
    annual = xirr(series, period_days=365)
    assert modified_dietz(0, closing, paid, START, END) == pytest.approx(annual, abs=0.002)


def test_they_part_company_on_one_badly_timed_lump_sum():
    """Doubles by mid-year, the lump sum arrives at the top, then halves."""
    opening, lump = 100.0, 1000.0
    closing = (opening * 2 + lump) / 2
    paid = [(date(2021, 7, 2), lump)]
    series = pd.Series(
        [-opening, -lump, closing], index=pd.to_datetime([START, date(2021, 7, 2), END])
    )
    dietz = modified_dietz(opening, closing, paid, START, END)
    assert abs(dietz - xirr(series, period_days=365)) > 0.05
