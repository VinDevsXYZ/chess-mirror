"""Stockfish as teacher/evaluator.

Two passes per position:
  * deep  -- MultiPV (default 5) at ``deep_depth``: best move, top candidates, PVs
  * broad -- MultiPV = all legal moves at ``broad_depth``: a quick evaluation of
             *every* legal move ("what a fast glance sees")
plus, for statistics only, a deep evaluation of the played move when it falls
outside the deep top-N.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.engine

from ..analysis.winprob import score_to_cp


# Safety caps so a pathological position cannot stall a batch (they rarely bind at club depths).
DEEP_NODE_CAP = 3_000_000
BROAD_NODE_CAP = 600_000


@dataclass(frozen=True)
class AnalysisSettings:
    deep_depth: int = 14
    deep_multipv: int = 5
    broad_depth: int = 8

    @property
    def key(self) -> str:
        return f"d{self.deep_depth}m{self.deep_multipv}b{self.broad_depth}"


def open_engine(path: str, threads: int = 1, hash_mb: int = 64) -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(path)
    opts = {}
    if "Threads" in eng.options:
        opts["Threads"] = threads
    if "Hash" in eng.options:
        opts["Hash"] = hash_mb
    eng.configure(opts)
    return eng


def _pack(board: chess.Board, info: dict, pv_len: int = 12) -> dict | None:
    pv = info.get("pv")
    if not pv:
        return None
    score = info["score"].pov(board.turn)
    cp, mate = score.score(), score.mate()
    return {"uci": pv[0].uci(), "cp": score_to_cp(cp, mate) if mate is None else None, "mate": mate,
            "pv": [m.uci() for m in pv[:pv_len]]}


def analyse_core(eng: chess.engine.SimpleEngine, board: chess.Board, s: AnalysisSettings) -> dict:
    n_legal = board.legal_moves.count()
    if n_legal == 0:
        return {"top": [], "broad": {}}
    infos = eng.analyse(board, chess.engine.Limit(depth=s.deep_depth, nodes=DEEP_NODE_CAP),
                        multipv=min(s.deep_multipv, n_legal))
    top = [p for p in (_pack(board, i) for i in infos) if p]
    binfos = eng.analyse(board, chess.engine.Limit(depth=s.broad_depth, nodes=BROAD_NODE_CAP), multipv=n_legal)
    broad = {}
    for i in binfos:
        p = _pack(board, i, pv_len=1)
        if p:
            broad[p["uci"]] = score_to_cp(p["cp"], p["mate"])
    # guarantee every legal move has a broad score (engines occasionally drop lines)
    missing = [m for m in board.legal_moves if m.uci() not in broad]
    for m in missing:
        info = eng.analyse(board, chess.engine.Limit(depth=max(1, s.broad_depth - 2)), root_moves=[m])
        p = _pack(board, info, pv_len=1)
        broad[m.uci()] = score_to_cp(p["cp"], p["mate"]) if p else -score_to_cp(0, None)
    return {"top": top, "broad": broad}


def analyse_move(eng: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move,
                 s: AnalysisSettings) -> dict | None:
    info = eng.analyse(board, chess.engine.Limit(depth=s.deep_depth, nodes=DEEP_NODE_CAP), root_moves=[move])
    return _pack(board, info)
