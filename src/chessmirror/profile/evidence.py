"""Honest uncertainty: every claim carries its sample size and an evidence tier.

Tiers (as required by the design):
  observed fact          raw counts from the games ("7 of 12")
  statistical inference  P(player differs from baseline) >= 1 - 0.05/m (Bonferroni over the m parallel
                         comparisons in the same family) with >= 30 opportunities
  weak hypothesis        P >= 0.75, or fewer observations
  insufficient evidence  otherwise -- not presented as a tendency
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist

FACT, INFERENCE, HYPOTHESIS, INSUFFICIENT = ("observed fact", "statistical inference", "weak hypothesis",
                                            "insufficient evidence")


def beta_ci(k: float, n: float, a: float = 1.0, b: float = 1.0, level: float = 0.95) -> tuple[float, float, float]:
    """Posterior mean and credible interval of a rate (Beta(a,b) prior)."""
    lo = beta_dist.ppf((1 - level) / 2, a + k, b + n - k)
    hi = beta_dist.ppf(1 - (1 - level) / 2, a + k, b + n - k)
    return (a + k) / (a + b + n), float(lo), float(hi)


def p_greater(k1: float, n1: float, k2: float, n2: float, draws: int = 20000, seed: int = 1) -> float:
    """P(rate1 > rate2) under independent Beta(1,1) posteriors."""
    rng = np.random.default_rng(seed)
    r1 = rng.beta(1 + k1, 1 + n1 - k1, draws)
    r2 = rng.beta(1 + k2, 1 + n2 - k2, draws)
    return float((r1 > r2).mean())


def tier(p_diff: float, n: int, min_n: int = 30, family: int = 1) -> str:
    """Evidence tier for a directional claim with posterior probability p_diff.

    ``family`` is the number of parallel comparisons the claim belongs to (e.g. 21
    position types); the inference threshold is Bonferroni-corrected so scanning
    many patterns does not manufacture "findings" out of noise.
    """
    if p_diff >= 1 - 0.05 / max(1, family) and n >= min_n:
        return INFERENCE
    if p_diff >= 0.75 and n >= 5:
        return HYPOTHESIS
    return INSUFFICIENT


def rate_vs_baseline(k: int, n: int, kb: int, nb: int, higher_is: str = "worse", family: int = 1) -> dict:
    """Compare a player rate with a baseline (population) rate."""
    mean, lo, hi = beta_ci(k, n)
    bmean, blo, bhi = beta_ci(kb, nb)
    p_hi = p_greater(k, n, kb, nb) if n and nb else 0.5
    direction = "higher" if p_hi >= 0.5 else "lower"
    p_dir = max(p_hi, 1 - p_hi)
    t = tier(p_dir, n, family=family)
    return {"k": k, "n": n, "rate": mean, "ci": [lo, hi], "baseline_k": kb, "baseline_n": nb,
            "baseline_rate": bmean, "baseline_ci": [blo, bhi], "p_higher": p_hi, "direction": direction,
            "confidence": p_dir, "tier": t, "family": family,
            "verdict": ("no clear difference" if t == INSUFFICIENT
                        else direction if higher_is == "neutral"  # style: higher/lower, not good/bad
                        else "worse" if (direction == "higher") == (higher_is == "worse") else "better")}


def bootstrap_mean(values_by_game: dict, iters: int = 1000, seed: int = 3) -> tuple[float, float, float]:
    """Mean of per-move values with a 95% CI from resampling whole games."""
    games = [np.asarray(v, dtype=float) for v in values_by_game.values() if len(v)]
    if not games:
        return float("nan"), float("nan"), float("nan")
    allv = np.concatenate(games)
    rng = np.random.default_rng(seed)
    sums = np.array([g.sum() for g in games])
    cnts = np.array([len(g) for g in games])
    boots = []
    for _ in range(iters):
        i = rng.integers(0, len(games), len(games))
        boots.append(sums[i].sum() / cnts[i].sum())
    return float(allv.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def describe_rate(label: str, r: dict, unit: str = "") -> str:
    txt = (f"{label}: {r['k']}/{r['n']}{unit} ({r['rate'] * 100:.0f}%, 95% CI {r['ci'][0] * 100:.0f}-"
           f"{r['ci'][1] * 100:.0f}%) vs similar-rated opponents {r['baseline_rate'] * 100:.0f}%")
    if r["tier"] == INSUFFICIENT:
        return txt + f" -> no clear difference ({INSUFFICIENT})"
    return txt + f" -> {r['verdict']} than baseline, confidence {r['confidence'] * 100:.0f}% [{r['tier']}]"


def pct(p: float) -> str:
    """Percent with enough precision that 99.7% is not shown as a misleading 100%."""
    return f"{p * 100:.1f}%" if 0.99 <= p < 1 else f"{p * 100:.0f}%"
