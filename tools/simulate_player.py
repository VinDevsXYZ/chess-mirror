"""Generate SYNTHETIC games from a simulated player with a known style.

Used only to validate the pipeline end to end ("parameter recovery"): if the
system is sound, the profile and personal model should rediscover the style we
planted here, and the personal model should beat the population baseline.

Games are clearly marked (Event "SYNTHETIC ...", Site "synthetic://") and should
be imported into a separate data dir:  MIRROR_DATA_DIR=data-demo
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chessmirror.analysis.features import FEATURES, Context, candidate_features  # noqa: E402
from chessmirror.config import find_stockfish  # noqa: E402
from chessmirror.engine.stockfish import AnalysisSettings, analyse_core  # noqa: E402
from chessmirror.model.logit import proba  # noqa: E402

SIM = AnalysisSettings(deep_depth=8, deep_multipv=5, broad_depth=5)

# Planted personality of the synthetic player (deltas vs. the population).
AHMED_STYLE = {
    "check": 0.7, "capture": 0.4, "pawn_storm": 0.8, "kz_attack": 0.5, "sac": 0.6,
    "queen_trade": -1.3, "trade_x_winning": -0.7, "fork": -1.2, "allows_fork": 0.9,
    "retreat": -0.5, "early_queen": 0.6,
}
AHMED_ELO = 1520
REPERTOIRE_WHITE = {"e2e4": 0.7, "d2d4": 0.3}
REPERTOIRE_BLACK = {"e2e4": {"c7c5": 0.60, "c7c6": 0.25, "e7e6": 0.15},
                    "d2d4": {"d7d5": 0.5, "g8f6": 0.5}}
OPP_FIRST = {"e2e4": 0.55, "d2d4": 0.35, "g1f3": 0.05, "c2c4": 0.05}


def population_beta(elo: float) -> np.ndarray:
    b = np.zeros(len(FEATURES))
    k = 6.0 + (elo - 1500) / 150.0

    def s(name, v):
        b[FEATURES.index(name)] = v
    s("loss_sqrt", -k)
    s("is_best", 0.3)
    s("shallow_best", 0.5)
    s("hidden", -1.0)
    s("trap", 0.8)
    s("loss_x_tp", 4.0)
    s("recapture", 0.8)
    s("develop", 0.6)
    s("castle", 0.8)
    s("hangs", -1.5)
    s("escape", 0.8)
    return b


def ahmed_beta() -> np.ndarray:
    b = population_beta(AHMED_ELO)
    for k, v in AHMED_STYLE.items():
        b[FEATURES.index(k)] += v
    return b


_ENG = None


def _init(sf):
    global _ENG
    _ENG = chess.engine.SimpleEngine.popen_uci(sf)
    _ENG.configure({"Threads": 1, "Hash": 32})


def _pick(dist: dict, rng) -> str:
    r, acc = rng.random(), 0.0
    for k, v in dist.items():
        acc += v
        if r <= acc:
            return k
    return next(iter(dist))


def play_game(args) -> str:
    gi, seed = args
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    ahmed_white = gi % 2 == 0
    opp_elo = int(rng.gauss(AHMED_ELO, 130))
    betas = {True: ahmed_beta(), False: population_beta(opp_elo)}  # key: is_ahmed
    base, inc = 300, 3
    clocks = {chess.WHITE: float(base), chess.BLACK: float(base)}
    board = chess.Board()
    game = chess.pgn.Game()
    node = game
    bad_streak = {chess.WHITE: 0, chess.BLACK: 0}
    result = None
    first_white_move = None
    while not board.is_game_over(claim_draw=True) and board.ply() < 160:
        mover = board.turn
        is_ahmed = (mover == chess.WHITE) == ahmed_white
        core = analyse_core(_ENG, board, SIM)
        top = core["top"]
        # resignation
        ev = top[0]["cp"] if top[0]["cp"] is not None else (10000 if top[0]["mate"] > 0 else -10000)
        bad_streak[mover] = bad_streak[mover] + 1 if ev < -700 else 0
        if bad_streak[mover] >= 3:
            result = "0-1" if mover == chess.WHITE else "1-0"
            break
        uci = None
        if board.ply() == 0:
            uci = _pick(REPERTOIRE_WHITE if is_ahmed else OPP_FIRST, rng)
        elif board.ply() == 1 and is_ahmed and first_white_move in REPERTOIRE_BLACK:
            uci = _pick(REPERTOIRE_BLACK[first_white_move], rng)
        ctx = Context(clock=clocks[mover], base=base, prev_move=board.peek().uci() if board.move_stack else None)
        ucis, X = candidate_features(board, top, core["broad"], ctx)
        p = proba(betas[is_ahmed], X)
        if uci is None:
            uci = ucis[int(nrng.choice(len(p), p=p))]
        # think time: harder positions and roomier clocks -> longer thinks
        sharp = float((X[:, 0] >= 0.10).mean())
        mean_t = (2.0 if board.ply() < 16 else 9.0 + 12 * sharp) * min(1.0, clocks[mover] / base * 1.5)
        spent = min(clocks[mover] - 0.1, max(0.3, nrng.lognormal(math.log(max(0.4, mean_t)), 0.6)))
        clocks[mover] = clocks[mover] - spent + inc
        if clocks[mover] <= 0.1:
            result = "0-1" if mover == chess.WHITE else "1-0"
            break
        move = chess.Move.from_uci(uci)
        if board.ply() == 0:
            first_white_move = uci
        board.push(move)
        node = node.add_variation(move)
        node.set_clock(round(clocks[mover]))
    if result is None:
        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
        else:
            info = _ENG.analyse(board, chess.engine.Limit(depth=10))
            cp = info["score"].white().score(mate_score=10000)
            result = "1-0" if cp > 250 else "0-1" if cp < -250 else "1/2-1/2"
    ahmed_elo_shown = int(rng.gauss(AHMED_ELO, 25))
    h = game.headers
    h["Event"] = "SYNTHETIC Rated Blitz game"
    h["Site"] = f"synthetic://game/{gi}"
    h["Date"] = h["UTCDate"] = f"2026.{1 + gi // 30 % 9:02d}.{1 + gi % 28:02d}"
    h["UTCTime"] = f"{10 + gi % 12:02d}:{gi % 60:02d}:00"
    h["White"], h["Black"] = ("Ahmed", f"Opponent{gi % 50}") if ahmed_white else (f"Opponent{gi % 50}", "Ahmed")
    h["WhiteElo"], h["BlackElo"] = (str(ahmed_elo_shown), str(opp_elo)) if ahmed_white else (str(opp_elo), str(ahmed_elo_shown))
    h["TimeControl"] = f"{base}+{inc}"
    h["Result"] = result
    return str(game)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default="data-demo/synthetic_ahmed.pgn")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf = find_stockfish()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(sf,)) as pool:
        pgns = []
        for i, pgn in enumerate(pool.imap_unordered(play_game, [(g, a.seed + g) for g in range(a.games)]), 1):
            pgns.append(pgn)
            if i % 20 == 0:
                print(f"{i}/{a.games} games", flush=True)
    out.write_text("\n\n".join(pgns) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
