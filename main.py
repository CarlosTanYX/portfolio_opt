"""
main.py

Entry point. Run with: python3 main.py

Uses simulated price data by default (no internet/yfinance needed to
try this out). Swap in load_prices(...) to test with real tickers
(refer to commented-out line below).
"""
import numpy as np
import matplotlib.pyplot as plt

from portfolio.data import simulate_prices
from portfolio.data import load_prices
from portfolio.stats import compute_returns, expected_returns, covariance_matrix
from portfolio.optimiser import (
    min_variance_portfolio,
    efficient_frontier,
    random_portfolios,
    portfolio_return,
    portfolio_variance,
)

# ---- 1. get price data ----
prices = simulate_prices(n_assets=4, n_days=500)
# prices = load_prices(["COP", "BJRI", "EVH", "PLMR"], start="2020-01-01", end="2026-07-01")

# ---- 2. process raw data ----
returns = compute_returns(prices)
mu = expected_returns(returns)
cov = covariance_matrix(returns)

print("Assets:", list(prices.columns))
print("Expected annual returns:", np.round(mu, 4))
print()

# ---- 3. run the optimiser ----
w_minvar = min_variance_portfolio(cov)
print("Minimum-variance portfolio weights:", np.round(w_minvar, 4))
print("  -> expected return:", round(portfolio_return(w_minvar, mu), 4))
print("  -> variance:       ", round(portfolio_variance(w_minvar, cov), 6))
print()

frontier_returns, frontier_variances, frontier_weights = efficient_frontier(
    mu, cov, n_points=50)
random_returns, random_variances = random_portfolios(
    mu, cov, n_portfolios=3000)

# ---- 4. plot results ----
plt.figure(figsize=(9, 6))
plt.scatter(np.sqrt(random_variances), random_returns,
            s=4, alpha=0.3, label="Random portfolios")
plt.plot(np.sqrt(frontier_variances), frontier_returns,
         color="red", linewidth=2, label="Efficient frontier")
plt.scatter(
    [np.sqrt(portfolio_variance(w_minvar, cov))],
    [portfolio_return(w_minvar, mu)],
    color="black", marker="*", s=200, zorder=5, label="Min-variance portfolio",
)
plt.xlabel("Risk (standard deviation)")
plt.ylabel("Expected return")
plt.title("Efficient Frontier vs. Random Portfolios")
plt.legend()
plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150)
print("Saved plot to efficient_frontier.png")
