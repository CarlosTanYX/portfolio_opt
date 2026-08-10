"""
optimiser.py

The actual Markowitz mean-variance math -- solved in closed form via
matrix algebra (Lagrange multipliers). Solve for optimum by inverting the matrix.

Problem being solved:
    minimise   w^T Sigma w                  (portfolio variance)
    subject to w^T mu = target_return       (hit a specific expected return)
               w^T 1 = 1                    (weights sum to 1 -- fully invested)
"""
import numpy as np


def portfolio_return(w, mu):
    return w @ mu           # w^T . mu


def portfolio_variance(w, cov):
    return w @ cov @ w      # w^T . Sigma w


def min_variance_portfolio(cov):
    """
    The GLOBAL minimum-variance portfolio -- ignores expected return
    entirely, just finds the weights that minimise risk as long as 
    portfolio is fully invested (w^T 1 = 1). 
    Closed form:       w = (Sigma^-1 1) / (1^T Sigma^-1 1)
    from solving the Lagrangian: 
                       L(w, gam) = w^T . Sigma . w - gam(w^T . 1 - 1)
    """
    n = cov.shape[0]
    ones = np.ones(n)
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ ones
    w = w / (ones @ inv_cov @ ones)
    return w


def efficient_frontier_weights(mu, cov, target_return):
    """
    Weights for the minimum-variance portfolio that achieves EXACTLY
    target_return. Sweeping this over many target_return values traces
    out the whole efficient frontier. Consequence of the two-fund separation
    formula: every efficient portfolio is a linear combination of the
    same two "fund" vectors (Sigma^-1 @ ones and Sigma^-1 @ mu).

    Lagrangian:
        L(w, gam, lam) = w^T . Sigma . w - gam(w^T . mu - r) - lam(w^T . 1 - 1)

    Closed-form solution in terms of four scalars derived from Sigma^-1:
        A = 1^T Sigma^-1 1
        B = 1^T Sigma^-1 mu
        C = mu^T Sigma^-1 mu
        D = A*C - B^2

    For p = lam / 2; q = gam / 2:
        C * p + B * q = target_return
        B * p + A * q = 1
    Apply Cramer's Rule to solve.

    From the Lagrangian: 
        w = (gam / 2) * (inv_cov . mu) + (lam / 2) * (inv_cov . ones)
    """
    n = cov.shape[0]
    ones = np.ones(n)
    inv_cov = np.linalg.inv(cov)

    A = ones @ inv_cov @ ones
    B = ones @ inv_cov @ mu
    C = mu @ inv_cov @ mu
    D = A * C - B ** 2

    lam = [(C - B * target_return) / D] * 2
    gam = [(A * target_return - B) / D] * 2

    w = 2 * lam * (inv_cov @ ones) + 2 * gam * (inv_cov @ mu)
    return w


def efficient_frontier(mu, cov, n_points=50):
    """
    Sweeps target_return across a sensible range (from the min-variance
    portfolio's own return up to the single best-returning asset) and
    solves for the optimal weights at each point via the function 
    efficient_frontier_weights().

    Returns three arrays of length n_points: returns, variances (risk),
    and the weight vectors themselves.
    """
    w_minvar = min_variance_portfolio(cov)
    min_ret = portfolio_return(w_minvar, mu)
    max_ret = mu.max()

    target_returns = np.linspace(min_ret, max_ret, n_points)
    variances = []
    weights_list = []
    for r in target_returns:
        w = efficient_frontier_weights(mu, cov, r)
        variances.append(portfolio_variance(w, cov))
        weights_list.append(w)

    return target_returns, np.array(variances), np.array(weights_list)


def random_portfolios(mu, cov, n_portfolios=2000, seed=42):
    """
    Randomly sampled (not optimised) portfolio weights, used purely for
    comparison. Plotting these alongside the efficient frontier makes 
    its curved shape visually obvious: random portfolios form 
    a cloud, and the frontier traces its upper-left boundary (least risk 
    for a given return).
    """
    rng = np.random.default_rng(seed)
    n = cov.shape[0]
    returns, variances = [], []
    for _ in range(n_portfolios):
        w = rng.random(n)
        w = w / w.sum()
        returns.append(portfolio_return(w, mu))
        variances.append(portfolio_variance(w, cov))
    return np.array(returns), np.array(variances)
