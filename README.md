# Markowitz Portfolio Optimization

Closed-form mean-variance portfolio optimization, following Harry
Markowitz's 1952 "Portfolio Selection" -- solved via matrix algebra
(Lagrange multipliers), not an iterative solver.

## Structure

- `portfolio/data.py` -- load real prices (`yfinance`) or generate
  synthetic prices with a known correlation structure, for testing
  the math without needing network access
- `portfolio/stats.py` -- turn prices into returns, an expected-return
  vector, and a covariance matrix
- `portfolio/optimizer.py` -- the actual Markowitz math: the global
  minimum-variance portfolio, the efficient frontier (swept via the
  two-fund separation closed form), and random portfolios for
  comparison
- `main.py` -- entry point: loads data, runs the optimizer, plots the
  efficient frontier against a cloud of random portfolios

## Setup

```
pip install -r requirements.txt
```

## Run

```
python3 main.py
```

By default this uses simulated price data (`simulate_prices` in
`data.py`), so it works immediately with no internet access. To use
real data, uncomment the `load_prices(...)` line in `main.py` and
supply real tickers -- requires `yfinance` and an internet connection.

## What to expect

`efficient_frontier.png` shows the classic Markowitz picture: a cloud
of randomly-weighted portfolios, and a curved red line (the efficient
frontier) tracing its upper-left boundary -- the best possible
risk/return trade-off achievable from this set of assets. The black
star marks the global minimum-variance portfolio.

## Validated against an independent solver

The closed-form weights in `optimizer.py` were checked against
`scipy.optimize.minimize`'s numerical constrained solver on a known
test case and matched to 4 decimal places -- see the project's
development notes for the verification script.
