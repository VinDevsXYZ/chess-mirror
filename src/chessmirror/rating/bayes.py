"""Bayesian rating estimation on a grid.

The posterior over the player's rating R is kept on a 5-Elo grid, so it is
exact (no Gaussian approximation).  Between games R performs a random walk
(``drift_per_day``) so old games count less and improvement is tracked -- the
same idea as Glicko's rating deviation growth, computed exactly.

Likelihood of a game with score s against an opponent of rating r_o:
    E = 1 / (1 + 10^((r_o - R)/400)),   L = E^s (1-E)^(1-s)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

GRID = np.arange(0.0, 3605.0, 5.0)


def expected(r: np.ndarray | float, r_opp: float) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** ((r_opp - np.asarray(r)) / 400.0))


@dataclass
class Summary:
    mean: float
    sd: float
    lo: float
    hi: float
    n_games: int

    def text(self) -> str:
        return f"{self.mean:.0f} (95% interval {self.lo:.0f}-{self.hi:.0f}, {self.n_games} games)"

    def to_dict(self) -> dict:
        return {"mean": round(self.mean), "sd": round(self.sd), "lo": round(self.lo), "hi": round(self.hi),
                "n_games": self.n_games}


class RatingFilter:
    def __init__(self, prior_mean: float = 1500.0, prior_sd: float = 400.0, drift_per_day: float = 3.0):
        self.logp = -0.5 * ((GRID - prior_mean) / prior_sd) ** 2
        self.drift = drift_per_day
        self.last: dt.datetime | None = None
        self.n = 0

    def _diffuse(self, when: dt.datetime | None) -> None:
        if when is None or self.last is None or self.drift <= 0:
            self.last = when or self.last
            return
        days = max(0.0, (when - self.last).total_seconds() / 86400.0)
        self.last = when
        sd = self.drift * np.sqrt(days)
        if sd < 1.0:
            return
        p = self.prob()
        p = gaussian_filter1d(p, sigma=sd / 5.0, mode="nearest")
        self.logp = np.log(np.clip(p, 1e-300, None))

    def update(self, r_opp: float, score: float, when: dt.datetime | None = None) -> None:
        self._diffuse(when)
        e = np.clip(expected(GRID, r_opp), 1e-12, 1 - 1e-12)
        self.logp = self.logp + score * np.log(e) + (1 - score) * np.log(1 - e)
        self.logp -= self.logp.max()
        self.n += 1

    def prob(self) -> np.ndarray:
        p = np.exp(self.logp - self.logp.max())
        return p / p.sum()

    def summary(self, when: dt.datetime | None = None) -> Summary:
        if when is not None:
            self._diffuse(when)
        p = self.prob()
        mean = float((GRID * p).sum())
        sd = float(np.sqrt(((GRID - mean) ** 2 * p).sum()))
        cdf = np.cumsum(p)
        lo = float(GRID[np.searchsorted(cdf, 0.025)])
        hi = float(GRID[min(len(GRID) - 1, np.searchsorted(cdf, 0.975))])
        return Summary(mean, sd, lo, hi, self.n)

    def sample(self, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        return float(rng.choice(GRID, p=self.prob()))


def parse_when(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M"):
        try:
            return dt.datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def estimate(games: list[dict], prior_mean: float = 1500.0, prior_sd: float = 400.0,
             drift_per_day: float = 3.0) -> tuple[Summary, list[dict]]:
    """games: [{opp, score, when}] in chronological order. Returns summary + history trace."""
    f = RatingFilter(prior_mean, prior_sd, drift_per_day)
    trace = []
    for g in games:
        f.update(g["opp"], g["score"], g.get("when"))
        s = f.summary()
        trace.append({"when": g.get("when").isoformat() if g.get("when") else None, **s.to_dict()})
    return f.summary(), trace


def performance_rating(games: list[dict], prior_sd: float = 1e9) -> Summary:
    """Static (no drift) performance rating over a set of games, flat prior."""
    mean = float(np.mean([g["opp"] for g in games])) if games else 1500.0
    s, _ = estimate(games, prior_mean=mean, prior_sd=min(prior_sd, 2000.0), drift_per_day=0.0)
    return s
