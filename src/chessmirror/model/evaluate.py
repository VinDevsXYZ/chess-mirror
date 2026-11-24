"""Held-out evaluation of move models.

Metrics answer "is the clone becoming more like me?" -- top-1 / top-3 accuracy,
log loss (cross-entropy), calibration, and *behavioural* similarity: moves are
sampled from each model and their engine-loss distribution / style rates are
compared with the player's actual moves.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import wasserstein_distance

from ..analysis.features import F_INDEX
from ..analysis.winprob import eval_band
from .base import MovePredictor
from .dataset import Decision

STYLE_RATES = ["capture", "check", "sac", "trade", "queen_trade", "pawn_push", "pawn_storm", "castle", "retreat",
               "fork"]


def topk_credit(p: np.ndarray, y: int, k: int) -> float:
    """P(played move is in the model's top-k) with ties broken uniformly at random."""
    greater = int((p > p[y] + 1e-12).sum())
    equal = int((np.abs(p - p[y]) <= 1e-12).sum())
    return float(min(1.0, max(0.0, (k - greater) / equal)))


def _style(X_row: np.ndarray) -> dict:
    return {s: float(X_row[F_INDEX[s]] > 0) for s in STYLE_RATES}


def evaluate(model: MovePredictor, decisions: list[Decision], samples: int = 3, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    if not decisions:
        return {"n": 0}
    ll, top1, top3, conf = [], [], [], []
    groups = defaultdict(lambda: {"n": 0, "top1": 0, "ll": 0.0})
    act_loss, sim_loss = [], []
    act_style, sim_style = defaultdict(float), defaultdict(float)
    for d in decisions:
        p = model.predict(d)
        py = float(max(p[d.y], 1e-9))
        t1, t3 = topk_credit(p, d.y, 1), topk_credit(p, d.y, 3)
        ll.append(-np.log(py))
        top1.append(t1)
        top3.append(t3)
        conf.append((float(p.max()), t1))
        keys = [f"phase:{d.phase}", f"eval:{eval_band(d.wp_before or 0.5)}",
                f"clock:{'pressure' if d.ctx.time_pressure else 'normal'}",
                "opening_plies" if (d.ply or 99) < 20 else "after_ply_20"]
        for k in keys:
            g = groups[k]
            g["n"] += 1
            g["top1"] += t1
            g["ll"] += -np.log(py)
        act_loss.append(float(d.losses[d.y]))
        for s, v in _style(d.X[d.y]).items():
            act_style[s] += v
        for _ in range(samples):
            j = int(rng.choice(len(p), p=p / p.sum()))
            sim_loss.append(float(d.losses[j]))
            for s, v in _style(d.X[j]).items():
                sim_style[s] += v / samples

    n = len(decisions)
    # calibration: expected calibration error of the top-1 probability
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    c = np.array(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (c[:, 0] >= lo) & (c[:, 0] < hi)
        if m.any():
            ece += m.mean() * abs(c[m, 0].mean() - c[m, 1].mean())
    act_loss_a, sim_loss_a = np.array(act_loss), np.array(sim_loss)
    return {
        "n": n,
        "top1": float(np.mean(top1)),
        "top3": float(np.mean(top3)),
        "log_loss": float(np.mean(ll)),
        "perplexity": float(np.exp(np.mean(ll))),
        "ece": float(ece),
        "groups": {k: {"n": g["n"], "top1": g["top1"] / g["n"], "log_loss": g["ll"] / g["n"]}
                   for k, g in sorted(groups.items())},
        "behaviour": {
            "actual_mean_loss": float(act_loss_a.mean()),
            "model_mean_loss": float(sim_loss_a.mean()),
            "actual_blunder_rate": float((act_loss_a >= 0.15).mean()),
            "model_blunder_rate": float((sim_loss_a >= 0.15).mean()),
            "loss_wasserstein": float(wasserstein_distance(act_loss_a, sim_loss_a)),
            "style_actual": {s: act_style[s] / n for s in STYLE_RATES},
            "style_model": {s: sim_style[s] / n for s in STYLE_RATES},
            "style_abs_error": float(np.mean([abs(act_style[s] - sim_style[s]) / n for s in STYLE_RATES])),
        },
        "_per_decision_ll": ll,
    }


def paired_bootstrap(ll_a: list[float], ll_b: list[float], game_ids: list[int], iters: int = 2000,
                     seed: int = 11) -> dict:
    """Difference in mean log loss (a - b) with a 95% CI, resampling whole games."""
    a, b, g = np.array(ll_a), np.array(ll_b), np.array(game_ids)
    games = np.unique(g)
    idx = {gg: np.where(g == gg)[0] for gg in games}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iters):
        pick = rng.choice(games, size=len(games), replace=True)
        rows = np.concatenate([idx[x] for x in pick])
        diffs.append(a[rows].mean() - b[rows].mean())
    diffs = np.array(diffs)
    return {"mean_diff": float(a.mean() - b.mean()), "ci95": [float(np.percentile(diffs, 2.5)),
                                                             float(np.percentile(diffs, 97.5))],
            "p_a_better": float((diffs < 0).mean())}
