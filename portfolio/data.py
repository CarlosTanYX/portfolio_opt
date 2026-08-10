"""

data.py

Retrieves price data with two ways:
- load_prices(): retrieves historical ticker data from yfinance
  (requires yfinance library and internet connection).
- simulate_prices(): simulates assets with known and controllable
  correlations. Used for testing the math in stats.py and optimiser.py
  (since we control the covariance structure).
"""

import numpy as np
import pandas as pd


def load_prices(tickers: list, start: str, end: str):
    """
    Historical closing prices for a list of tickers
    Requires 'pip install yfinance' and internet access.

    Returns a DataFrame: rows = dates, columns = tickers.
    """
    import yfinance as yf
    data = yf.download(tickers, start=start, end=end)['Close']
    return data.dropna()


def simulate_prices(n_assets=4, n_days=500, seed=60):
    """
    Generates a controlled set of daily prices for n_assets.
    Correlation structure is built in. To be used as a check 
    for the optimiser.

    Assets 0 and 1 are positively correlated (shared common_driver);
    Asset 2 is independent; Asset 3 is negatively correlated with assets
    0 and 1 (negative common_driver). According to Modern Portfolio Theory, 
    negative correlation should be preferred for diversification.

    Similarly to load_prices() returns a DataFrame: rows = dates, columns = tickers.
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(
        end=pd.Timestamp.today().normalize(), periods=n_days)
    n = len(dates)  # number of days is based off the number of business days

    common_driver = rng.normal(0, 0.01, n)
    independent_driver = rng.normal(0, 0.01, (n, 2))

    returns = np.zeros((n, n_assets))
    returns[:, 0] = common_driver + \
        rng.normal(0, 0.005, n)                     # correlated with 1
    returns[:, 1] = common_driver + \
        rng.normal(0, 0.005, n)                     # correlated with 0
    returns[:, 2] = independent_driver[:, 0]        # independent
    returns[:, 3] = -common_driver + rng.normal(0, 0.005, n) \
        if n_assets > 3 else independent_driver[:, 1]  # negatively correlated with 0 and 1

    closing_prices = 100 * np.exp(np.cumsum(returns, axis=0))
    columns = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(closing_prices, index=dates, columns=columns)
