"""
stats.py

Processes raw price data into daily returns, mean expected return
and the covariance matrix between assets. Functions in optimiser.py
only requires these three values and not the raw price data.
"""

import numpy as np


def compute_returns(prices):
    """
    Daily percentage returns from a price DataFrame.
    prices: DataFrame, rows = dates, columns = assets.
    Returns: DataFrame of the same shape with one less row (first 
    day has no prior day to compute a return against).
    """
    returns = prices.pct_change().dropna()
    return returns


def expected_returns(returns, annualize=True, trading_days=252):
    """
    Mean daily return per asset as a vector, with one number per asset.
    annualize = True scales it up to a yearly figure, convetional for
    financial data.
    """
    mu = returns.mean()         # returns mean of each column as a vector
    if annualize:
        mu = mu * trading_days
    return mu.values            # numpy array, shape (n_assets,)


def covariance_matrix(returns, annualize=True, trading_days=252):
    """
    The covariance matrix Sigma is an (n_assets, n_assets) matrix.
    Sigma[i, i] is asset i's own variance; Sigma[i, j] captures how
    assets i and j move together. Important for diversification: two 
    assets with a large negative Sigma[i, j] reduces portfolio variance.
    """
    cov = returns.cov()     # Sigma[i, j] = sum_t[(R_it - mu_i)(R_jt - mu_j)]
    if annualize:
        cov = cov * trading_days
    return cov.values       # plain numpy array, shape (n_assets, n_assets)
