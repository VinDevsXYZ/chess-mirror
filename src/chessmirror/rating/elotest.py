"""ELO TEST mode: adaptive games against a calibrated opponent.

The opponent is Stockfish with ``UCI_LimitStrength`` (valid 1320-3190; Stockfish
calibrates this against CCRL-anchored engine ratings -- a different scale from
Lichess/Chess.com/FIDE, so the estimate is reported on that scale and kept
separate from the platform-rating estimates).

Opponent selection is Thompson sampling from the current posterior: each game is
played against a rating drawn from where the player's strength probably is,
which concentrates games where they are most informative (expected score ~50%)
while still exploring when the estimate is uncertain.
"""

from __future__ import annotations

import json

import numpy as np

from ..config import Config
from ..db import DB
from .bayes import RatingFilter, parse_when

MIN_ELO, MAX_ELO = 1320, 3190


def _filter(db: DB) -> RatingFilter:
    rows = db.q("""SELECT opponent_rating, result, played_at FROM elo_games
                   WHERE scale='stockfish-uci' AND status='final' AND result IS NOT NULL ORDER BY played_at""")
    # start from the platform-based estimate if we have one (only as a vague prior)
    prior = db.scalar("SELECT AVG(player_elo) FROM games WHERE player_elo IS NOT NULL AND platform != 'mirror-engine'")
    f = RatingFilter(prior_mean=float(prior or 1500), prior_sd=450.0, drift_per_day=2.0)
    for r in rows:
        f.update(r["opponent_rating"], r["result"], parse_when(r["played_at"]))
    return f


def prepare_next(db: DB, cfg: Config, forced: int | None = None) -> str:
    f = _filter(db)
    s = f.summary()
    if forced:
        opp = int(forced)
    elif f.n < 3:
        opp = int(round(s.mean / 10.0) * 10)  # first games: play at the best guess
    else:
        # Thompson sampling, kept within one posterior sd so every game stays informative
        draw = min(max(f.sample(), s.mean - s.sd), s.mean + s.sd)
        opp = int(round(draw / 10.0) * 10)
    opp = max(MIN_ELO, min(MAX_ELO, opp))
    state = cfg.state_dir / "elotest.json"
    state.write_text(json.dumps({"opponent_elo": opp}))
    msg = (f"Next Elo-test opponent: Stockfish UCI_Elo {opp}. "
           f"Current Elo-test estimate: {s.text()} [Stockfish UCI scale]. "
           "Set the engine's Mode option to EloTest and start a game in En Croissant's Play tab.")
    if s.mean < MIN_ELO + 50 and f.n >= 3:
        msg += (" Note: your estimate is near the lowest calibrated opponent (1320); games still add information, "
                "but less per game.")
    return msg


def current_opponent(cfg: Config) -> float:
    state = cfg.state_dir / "elotest.json"
    if state.exists():
        try:
            return float(json.loads(state.read_text())["opponent_elo"])
        except Exception:
            pass
    return 1500.0


def pending_text(db: DB) -> str:
    from ..uci.recorder import pending_text as pt
    from ..config import load_config
    return pt(load_config())


def record_result(db: DB, cfg: Config, gid, result: str) -> str:
    from ..uci.recorder import finalize_pending
    result = {"win": None, "loss": None, "draw": "1/2-1/2"}.get(result, result)
    if result is None:
        return "Use 1-0 / 0-1 / 1/2-1/2 (from White's point of view)."
    return finalize_pending(db, cfg, str(gid), result)


def elo_text(rep: dict, db: DB) -> str:
    lines = ["STRENGTH ESTIMATES (each on its own rating scale)"]
    if not rep["results"] and not rep["quality"] and not rep["elotest"]:
        return "No rated games yet. Import rated games (with opponent ratings) or play Elo-test games."
    for pool, r in sorted(rep["results"].items()):
        c = r["current"]
        lines.append(f"\n[{pool}]  {r['n_games']} rated games")
        lines.append(f"  From results (Bayesian, drift-aware): {c['mean']}  95% interval {c['lo']}-{c['hi']}")
        rp = r["recent_performance_30"]
        lines.append(f"  Performance, last 30 games:           {rp['mean']}  95% interval {rp['lo']}-{rp['hi']}")
        if r.get("platform_reported_latest"):
            lines.append(f"  Platform-reported rating (latest):     {r['platform_reported_latest']}  (observed fact)")
        q = rep["quality"].get(pool)
        if q and q.get("status") == "ok":
            lines.append(f"  From move quality (calibrated on {q['calibration']['opponent_games']} opponent games): "
                         f"{q['mean']:.0f}  95% interval {q['lo']:.0f}-{q['hi']:.0f}")
        elif q:
            lines.append(f"  From move quality: not estimated ({q['status']})")
        cb = rep["combined"].get(pool)
        if cb:
            lines.append(f"  Combined estimate:                     {cb['mean']}  95% interval {cb['lo']}-{cb['hi']}")
    if rep["elotest"]:
        e = rep["elotest"]["current"]
        lines.append(f"\n[Elo test vs Stockfish]  {e['n_games']} games")
        lines.append(f"  {e['mean']}  95% interval {e['lo']}-{e['hi']}   ({rep['elotest']['scale']})")
    lines.append("\nImitation quality is reported separately (`mirror models`): strength and similarity are different things.")
    return "\n".join(lines)
