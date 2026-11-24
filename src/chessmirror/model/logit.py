"""Conditional-logit move model:  P(move | position) = softmax_j(beta . x_j).

Fitted by MAP with a Gaussian prior ``beta ~ N(mu, sigma^2)``.  For the
personal model ``mu`` is the population (Elo-matched opponents) coefficient
vector, so with little personal data the model *is* the population model and it
moves away only as far as the player's own moves justify.  ``sigma`` is chosen
by game-grouped cross-validation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..analysis.features import FEATURES, N_FEATURES
from .dataset import Stacked


def _segment_softmax(u: np.ndarray, S: Stacked) -> tuple[np.ndarray, np.ndarray]:
    starts = S.offsets[:-1]
    m = np.maximum.reduceat(u, starts)
    e = np.exp(u - m[S.seg])
    z = np.add.reduceat(e, starts)
    p = e / z[S.seg]
    lse = m + np.log(z)
    return p, lse


def nll_grad(beta: np.ndarray, S: Stacked, mu: np.ndarray, prec: np.ndarray):
    u = S.X @ beta
    p, lse = _segment_softmax(u, S)
    nll = float(np.sum(lse) - np.sum(u[S.y]))
    grad = S.X.T @ p - S.X[S.y].sum(axis=0)
    d = beta - mu
    nll += 0.5 * float(np.sum(prec * d * d))
    grad = grad + prec * d
    return nll, grad


def hessian(beta: np.ndarray, S: Stacked, prec: np.ndarray) -> np.ndarray:
    u = S.X @ beta
    p, _ = _segment_softmax(u, S)
    Xp = S.X * p[:, None]
    xbar = np.add.reduceat(Xp, S.offsets[:-1])
    H = Xp.T @ S.X - xbar.T @ xbar
    return H + np.diag(prec)


def fit_logit(S: Stacked, mu: np.ndarray | None = None, sigma: float | np.ndarray = 3.0,
              beta0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (beta, standard_errors)."""
    mu = np.zeros(N_FEATURES) if mu is None else np.asarray(mu, dtype=np.float64)
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (N_FEATURES,))
    prec = 1.0 / (sig ** 2)
    x0 = mu.copy() if beta0 is None else beta0
    if S.n == 0:
        return mu.copy(), sig.copy()
    res = minimize(nll_grad, x0, args=(S, mu, prec), jac=True, method="L-BFGS-B",
                   options={"maxiter": 500, "gtol": 1e-6})
    beta = res.x
    try:
        cov = np.linalg.inv(hessian(beta, S, prec))
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(N_FEATURES, np.nan)
    return beta, se


def log_lik(beta: np.ndarray, S: Stacked) -> float:
    u = S.X @ beta
    _, lse = _segment_softmax(u, S)
    return float(np.sum(u[S.y]) - np.sum(lse))


def proba(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    u = X.astype(np.float64) @ beta
    u -= u.max()
    e = np.exp(u)
    return e / e.sum()


def default_prior(elo: float = 1500.0) -> np.ndarray:
    """Starting point before ANY data exists: Elo-scaled engine quality plus a few
    generic human habits.  A rough, uncalibrated guess -- replaced by the population
    fit (opponents) and by the player's own games as soon as they exist.
    """
    beta = np.zeros(N_FEATURES)
    k = max(2.0, 6.0 + (elo - 1500.0) / 150.0)
    habits = {
        # engine quality: stronger players avoid losing moves more sharply
        "loss_sqrt": -k, "is_best": 0.3 + max(0.0, (elo - 1500.0) / 1000.0), "shallow_best": 0.3,
        # generic club-player habits (so a brand-new Mirror doesn't open 1.a4); replaced by data
        "develop": 0.7, "center": 0.7, "castle": 1.0, "recapture": 0.8, "escape": 0.6,
        "hangs": -1.0, "early_queen": -0.5, "king_walk": -1.0,
    }
    for name, v in habits.items():
        beta[FEATURES.index(name)] = v
    return beta


def group_folds(game_ids: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    return [np.isin(game_ids, games[i::k]) for i in range(k)]
