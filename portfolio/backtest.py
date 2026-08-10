"""
backtest.py

The other files in this program help us choose the optimal weights
given my and Sigma. This file helps us backtest the optimal weights
based on data the optimiser never uses.

The backtest finds mu and Sigma from one window (training), then 
applies the optimal weights to a later window (test) that the optimiser
has not accessed. This allows us to check whether previously "optimal" 
weights still work going forward.
"""

import numpy as np

from portfolio.stats import compute_returns, expected_returns, covariance_matrix
from portfolio.optimiser import min_variance_portfolio


def backtest(prices, train_days=250, test_days=60):
    """
    Fits Sigma on a training window, computes the global min-variance
    weights from it , then applies those FIXED weights to a later, held-out 
    test window and compares realised performance against a naive equal-weight 
    portfolio.

    prices: DataFrame of prices (rows = dates, columns = assets),
        must have at least train_days + test_days + 1 rows (the +1
        is because compute_returns drops the first day).
    train_days, test_days: length of the fit window and the
        held-out evaluation window, in trading days.

    Returns a dict of realised daily mean return and volatility
    (standard deviation) for both the optimised and naive portfolios
    over the test window, plus the fitted weights themselves.
    """
    returns = compute_returns(prices)

    if len(returns) < train_days + test_days:
        raise ValueError(
            f"Not enough data: need {train_days + test_days} days of "
            f"returns, got {len(returns)}."
        )

    train = returns.iloc[:train_days]
    test = returns.iloc[train_days:train_days + test_days]

    cov = covariance_matrix(train, annualize=False)
    w_opt = min_variance_portfolio(cov)

    n = cov.shape[0]
    w_naive = np.ones(n) / n  # naive equal-weight baseline, no fitting at all

    # Apply the FIXED weights (from train) to each day of the test
    # window's actual realised returns: test.values is (test_days,
    # n_assets), so this matrix-vector product gives one realised
    # portfolio return per test day, for both naive and "optimal"
    # strategies.
    opt_daily = test.values @ w_opt
    naive_daily = test.values @ w_naive

    return {
        "weights_optimised": w_opt,
        "weights_naive": w_naive,
        "optimised_mean_return": opt_daily.mean(),
        "optimised_volatility": opt_daily.std(),
        "naive_mean_return": naive_daily.mean(),
        "naive_volatility": naive_daily.std(),
    }


def rolling_backtest(prices, train_days=250, test_days=60, step_days=None):
    """
    Repeats backtest()'s train/test idea many times, sliding both
    windows forward through the whole price history.

    step_days: how far to slide forward between windows. Defaults to
        test_days, which gives non-overlapping test windows -- each
        day of realised return is used in exactly one window's
        evaluation, never double-counted.

    Returns a list of per-window dicts, each with the window's start
    index and its realised daily return series for both strategies
    (test_days-long arrays) -- summarise_rolling_backtest() turns
    this into aggregate numbers and cumulative growth curves.
    """
    step_days = step_days or test_days
    returns = compute_returns(prices)
    n_total = len(returns)

    windows = []
    start = 0
    while start + train_days + test_days <= n_total:
        train = returns.iloc[start:start + train_days]
        test = returns.iloc[start + train_days:start + train_days + test_days]

        cov = covariance_matrix(train, annualize=False)
        w_opt = min_variance_portfolio(cov)
        n = cov.shape[0]
        w_naive = np.ones(n) / n

        windows.append({
            "start": start,
            "opt_daily": test.values @ w_opt,
            "naive_daily": test.values @ w_naive,
        })
        start += step_days

    if not windows:
        raise ValueError(
            "Not enough data for even one window: need at least "
            f"{train_days + test_days} days of returns, got {n_total}."
        )
    return windows


def summarise_rolling_backtest(windows):
    """
    Aggregates rolling_backtest()'s per-window results into overall
    statistics and cumulative growth curves.

    optimised_win_rate: fraction of windows where the optimised
        portfolio's mean daily return beat the naive one's

    opt_cumulative / naive_cumulative: growth of $1 invested at the
        very start, stepping through every test-window day in order
        (concatenated across all windows) -- this is what a growth
        chart over the whole out-of-sample period would plot.
    """
    opt_all = np.concatenate([w["opt_daily"] for w in windows])
    naive_all = np.concatenate([w["naive_daily"] for w in windows])

    win_rate = np.mean([
        w["opt_daily"].mean() > w["naive_daily"].mean() for w in windows
    ])

    return {
        "n_windows": len(windows),
        "optimised_mean_return": opt_all.mean(),
        "optimised_volatility": opt_all.std(),
        "naive_mean_return": naive_all.mean(),
        "naive_volatility": naive_all.std(),
        "optimised_win_rate": win_rate,
        "opt_cumulative": np.cumprod(1 + opt_all),
        "naive_cumulative": np.cumprod(1 + naive_all),
    }
