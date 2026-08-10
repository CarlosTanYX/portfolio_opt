# Markowitz Portfolio Optimisation

Closed-form mean-variance portfolio optimisation, following Harry
Markowitz's 1952 "Portfolio Selection", solved with Lagrange multipliers.

## Structure

- `portfolio/data.py` -- load real prices (`yfinance`) or generate
  synthetic prices with predetermined correlation, for testing
  the optimiser without network access
- `portfolio/stats.py` -- processed prices data into returns, an
  expected-return vector, and a covariance matrix
- `portfolio/optimizer.py` -- Markowitz math: the global minimum-
  variance portfolio, the efficient frontier (swept via the
  two-fund separation closed form), and random portfolios for
  comparison
- `main.py` -- entry point: loads data, runs the optimiser, plots the
  efficient frontier against a cloud of random portfolios

## Setup

```
pip install -r requirements.txt
```

## Run

```
python3 main.py
```

By default, this uses simulated price data (`simulate_prices` in
`data.py`), so it works immediately with no internet access. To use
real data, uncomment the `load_prices(...)` line in `main.py` and
supply real tickers -- requires `yfinance` and an internet connection.

## What to expect


`efficient_frontier.png` shows the classic Markowitz picture: a cloud
of randomly-weighted portfolios, and a curved red line (the efficient
frontier) tracing its upper-left boundary -- the best possible
risk/return trade-off achievable from this set of assets. The black
star marks the global minimum-variance portfolio.

