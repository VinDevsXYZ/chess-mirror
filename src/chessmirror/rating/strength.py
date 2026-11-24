"""Playing-strength estimates from several independent signals.

1. Results vs rated opponents, per rating pool (platform + time class), with a
   drifting Bayesian filter (``bayes.estimate``).
2. Move quality, calibrated on the player's own opponents: fit
   ``quality = c + d * rating`` across opponent games (the metric is the noisy
   dependent variable, so the slope is not attenuated), then invert for the
   player's average quality.  Uncertainty by bootstrap over games.
3. Elo-test games vs Stockfish UCI_LimitStrength (its own, separate scale).

Signals on the same scale are combined by inverse-variance weighting; scales
are never mixed silently.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..db import DB
from .bayes import Summary, estimate, parse_when, performance_rating

MIN_POOL_GAMES = 5
MIN_CALIBRATION_GAMES = 30


def _pool(row) -> str:
    return f"{row['platform']}:{row['tc_class']}"


def results_estimates(db: DB) -> dict:
    rows = db.q("""SELECT id, platform, tc_class, opponent_elo, player_score, utc_datetime, date, player_elo
                   FROM games WHERE player_color IS NOT NULL AND opponent_elo IS NOT NULL
                   AND player_score IS NOT NULL AND opponent_is_engine = 0 AND platform != 'mirror-engine'
                   ORDER BY COALESCE(utc_datetime, date), id""")
    pools = defaultdict(list)
    reported = {}
    for r in rows:
        pools[_pool(r)].append({"opp": r["opponent_elo"], "score": r["player_score"],
                                "when": parse_when(r["utc_datetime"] or r["date"])})
        if r["player_elo"]:
            reported[_pool(r)] = r["player_elo"]
    out = {}
    for pool, games in pools.items():
        if len(games) < MIN_POOL_GAMES:
            continue
        prior = float(np.mean([g["opp"] for g in games]))
        cur, trace = estimate(games, prior_mean=prior, prior_sd=400.0)
        recent = performance_rating(games[-30:])
        out[pool] = {"current": cur.to_dict(), "recent_performance_30": recent.to_dict(),
                     "platform_reported_latest": reported.get(pool), "trace": trace[-200:],
                     "n_games": len(games)}
    return out


def _game_quality(db: DB, player_side: bool) -> list[dict]:
    """Mean win-probability loss per game (x100), undecided positions only."""
    rows = db.q(f"""
        SELECT g.id AS gid, g.platform, g.tc_class, g.opponent_elo, g.player_elo,
               AVG(p.wp_loss) * 100 AS q, COUNT(*) AS n
        FROM positions p JOIN games g ON g.id = p.game_id
        WHERE p.is_player = {1 if player_side else 0} AND p.wp_loss IS NOT NULL
          AND p.ply >= 8 AND p.wp_before BETWEEN 0.03 AND 0.97 AND p.classification != 'forced'
          AND g.player_color IS NOT NULL AND g.opponent_is_engine = 0
        GROUP BY g.id HAVING n >= 8""")
    return [dict(r) for r in rows]


def quality_estimates(db: DB, iters: int = 1000, seed: int = 5) -> dict:
    opp = _game_quality(db, player_side=False)
    me = _game_quality(db, player_side=True)
    out = {}
    pools = {(r["platform"], r["tc_class"]) for r in opp}
    rng = np.random.default_rng(seed)
    for pool in pools:
        o = [r for r in opp if (r["platform"], r["tc_class"]) == pool and r["opponent_elo"]]
        m = [r for r in me if (r["platform"], r["tc_class"]) == pool]
        key = f"{pool[0]}:{pool[1]}"
        if len(o) < MIN_CALIBRATION_GAMES or len(m) < 5:
            continue
        R = np.array([r["opponent_elo"] for r in o], dtype=float)
        Q = np.array([r["q"] for r in o], dtype=float)
        if R.std() < 60:
            out[key] = {"status": "insufficient rating spread among opponents to calibrate",
                        "opponent_rating_sd": float(R.std())}
            continue
        Qm = np.array([r["q"] for r in m], dtype=float)
        ests = []
        for _ in range(iters):
            i = rng.integers(0, len(R), len(R))
            d, c = np.polyfit(R[i], Q[i], 1)
            if d >= 0:
                continue  # stronger players must lose less; skip degenerate resamples
            j = rng.integers(0, len(Qm), len(Qm))
            ests.append((Qm[j].mean() - c) / d)
        if len(ests) < iters * 0.5:
            out[key] = {"status": "calibration slope not reliably negative (quality does not track rating here)"}
            continue
        ests = np.clip(np.array(ests), 0, 3600)
        d, c = np.polyfit(R, Q, 1)
        out[key] = {"status": "ok", "mean": float(np.median(ests)), "sd": float(ests.std()),
                    "lo": float(np.percentile(ests, 2.5)), "hi": float(np.percentile(ests, 97.5)),
                    "calibration": {"slope_per_100elo": float(d * 100), "intercept": float(c),
                                    "opponent_games": len(o), "player_games": len(m),
                                    "player_mean_loss_x100": float(Qm.mean())}}
    return out


def elotest_estimate(db: DB) -> dict | None:
    rows = db.q("""SELECT opponent_rating, result, played_at FROM elo_games
                   WHERE scale = 'stockfish-uci' AND status = 'final' AND result IS NOT NULL
                   ORDER BY played_at""")
    if not rows:
        return None
    games = [{"opp": r["opponent_rating"], "score": r["result"], "when": parse_when(r["played_at"])} for r in rows]
    s, trace = estimate(games, prior_mean=1500.0, prior_sd=500.0, drift_per_day=2.0)
    return {"scale": "Stockfish UCI_Elo (CCRL-anchored engine scale; not identical to Lichess/FIDE)",
            "current": s.to_dict(), "trace": trace}


def combine(a: dict | None, b: dict | None) -> dict | None:
    parts = [x for x in (a, b) if x and x.get("sd")]
    if not parts:
        return None
    w = np.array([1.0 / x["sd"] ** 2 for x in parts])
    mean = float(np.sum(w * np.array([x["mean"] for x in parts])) / w.sum())
    sd = float(np.sqrt(1.0 / w.sum()))
    return {"mean": round(mean), "sd": round(sd), "lo": round(mean - 1.96 * sd), "hi": round(mean + 1.96 * sd),
            "note": "inverse-variance combination of independent estimates on the same scale (approximate)"}


def strength_report(db: DB) -> dict:
    res = results_estimates(db)
    qual = quality_estimates(db)
    combined = {}
    for pool in set(res) | set(qual):
        r = res.get(pool, {}).get("current")
        q = qual.get(pool) if qual.get(pool, {}).get("status") == "ok" else None
        combined[pool] = combine(r, q)
    primary = max(res, key=lambda k: res[k]["n_games"]) if res else (next(iter(qual)) if qual else None)
    return {"results": res, "quality": qual, "combined": combined, "elotest": elotest_estimate(db),
            "primary_pool": primary}
