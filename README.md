# finance-analysis-pp

Small, tested functions for measuring investment returns, written in Python.

| Function | What it answers |
|---|---|
| `xirr` | What did I earn a year, given when I paid in? |
| `modified_dietz` | The same question over one period, without solving for a rate |
| `time_weighted_return` | What did the portfolio earn, whatever I paid in? |
| `adjusted_units` | How many units is an old trade worth after a share split? |
| `to_gbp`, `panel_to_gbp` | What is a price in pounds, when some are quoted in pence? |

## Why two kinds of return

A fund can make money while its investor loses it. Here a price doubles and then
falls back to where it started, and most of the money arrives at the top:

```python
import pandas as pd
from finance_analysis_pp import time_weighted_return, xirr

days = pd.to_datetime(["2024-01-01", "2024-07-01", "2024-12-31"])
values = pd.Series([100.0, 1200.0, 600.0], index=days)
paid_in = pd.Series([0.0, 1000.0, 0.0], index=days)

time_weighted_return(values, paid_in)  # 0.0
xirr(pd.Series([-100.0, -1000.0, 600.0], index=days))  # about -0.68
```

A distributing fund such as VUSA pays its dividends out, so its price alone understates
the return. Pass them as `income`, and as money received in `xirr`.

## Design

- **It refuses rather than guesses.** `xirr` returns `None` when more than one rate
  solves the flows. A price with no stated quote currency is an error, not a guess
  from its size.
- **Pure functions.** No database, files or network.
- **Tests state what should be true**, rather than what the code happens to do.

All figures in the tests and examples are invented.

## Development

```bash
uv sync
uv run pytest
```
