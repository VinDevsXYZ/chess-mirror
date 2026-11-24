"""Batch analysis: run Stockfish over imported games and derive every datapoint field."""

from __future__ import annotations

import atexit
import datetime as dt
import json
import multiprocessing as mp
import time
from typing import Callable

import chess
import chess.engine

from ..config import Config
from ..db import DB
from ..engine.stockfish import AnalysisSettings, analyse_core, analyse_move, open_engine
from .features import Context, engine_losses, position_features
from .motifs import allows_after, best_capture_gain, line_motifs, static_motifs, threats_against
from .winprob import MISTAKE, classify, cp_to_wp, eval_band, move_accuracy, score_to_cp

_ENGINE = None
_SETTINGS: AnalysisSettings | None = None
_SF_PATH = ""
_HASH_MB = 64


def _init_worker(sf_path: str, hash_mb: int, settings: AnalysisSettings) -> None:
    global _ENGINE, _SETTINGS, _SF_PATH, _HASH_MB
    _SF_PATH, _HASH_MB = sf_path, hash_mb
    _ENGINE = open_engine(sf_path, threads=1, hash_mb=hash_mb)
    _SETTINGS = settings
    atexit.register(lambda: _ENGINE.quit())


def _clamp(cp: int) -> int:
    return max(-1000, min(1000, cp))


def derive(board: chess.Board, move: chess.Move, core: dict, played: dict | None,
           prev_move: str | None, clock: float | None, base: float | None) -> dict:
    """All derived fields for one datapoint (mover's point of view)."""
    top, broad = core["top"], core["broad"]
    n_legal = board.legal_moves.count()
    best = top[0]
    uci = move.uci()
    rank = next((i + 1 for i, t in enumerate(top) if t["uci"] == uci), None)
    p = top[rank - 1] if rank else played
    wp_before = cp_to_wp(best["cp"], best["mate"])
    wp_after = cp_to_wp(p["cp"], p["mate"]) if p else wp_before
    wp_loss = max(0.0, wp_before - wp_after)
    cp_before = _clamp(score_to_cp(best["cp"], best["mate"]))
    cp_after = _clamp(score_to_cp(p["cp"], p["mate"])) if p else cp_before
    ctx = Context(clock=clock, base=base, prev_move=prev_move)

    q = engine_losses(board, top, broad)
    second_wp = cp_to_wp(top[1]["cp"], top[1]["mate"]) if len(top) > 1 else 0.0
    b_best = max((cp_to_wp(v) for v in broad.values()), default=0.5)
    feats = position_features(board)
    feats.update({
        "eval_band": eval_band(wp_before),
        "sharp": sum(1 for v in q.values() if v["loss"] >= 0.10) / max(1, len(q)),
        "n_good": sum(1 for v in q.values() if v["loss"] <= 0.03),
        "volatility": abs(wp_before - b_best),
        "time_pressure": ctx.time_pressure,
        "only_move": (wp_before - second_wp) >= 0.15 and len(top) > 1,
        "best_gap": wp_before - second_wp,
    })

    after = board.copy(stack=False)
    after.push(move)
    threats = threats_against(board)
    now = set()
    if not after.is_game_over():
        now = allows_after(after)
        if best_capture_gain(after, after.turn)[0] >= 200:
            now.add("hanging_piece")
        if "mate_in_1" in now:
            now.add("mate_threat")
    motifs = {
        "best": sorted(line_motifs(board, best.get("pv", [uci]), mate=best["mate"], prev_move=prev_move)),
        "played": sorted(static_motifs(board, move)),
        "threats": sorted(threats),
        "threats_unparried": sorted(threats & now),
        "allowed": [],
    }
    if wp_loss >= MISTAKE and p and len(p.get("pv", [])) > 1:
        opp_mate = -p["mate"] if p["mate"] is not None and p["mate"] < 0 else None
        motifs["allowed"] = sorted(line_motifs(after, p["pv"][1:], mate=opp_mate, prev_move=uci))

    return {
        "eval_before_cp": cp_before, "mate_before": best["mate"],
        "eval_after_cp": cp_after, "mate_after": p["mate"] if p else None,
        "wp_before": wp_before, "wp_after": wp_after, "wp_loss": wp_loss,
        "cp_loss": max(0, cp_before - cp_after),
        "accuracy": move_accuracy(wp_before, wp_after),
        "best_uci": best["uci"], "best_san": board.san(chess.Move.from_uci(best["uci"])),
        "played_rank": rank,
        "top_moves": json.dumps([{**t, "san": board.san(chess.Move.from_uci(t["uci"]))} for t in top]),
        "broad_moves": json.dumps(broad),
        "played_pv": json.dumps(p.get("pv", []) if p else []),
        "classification": classify(wp_loss, rank, n_legal),
        "phase": feats["phase"],
        "features": json.dumps(feats),
        "motifs": json.dumps(motifs),
    }


def _restart_engine() -> None:
    global _ENGINE
    try:
        _ENGINE.quit()
    except Exception:
        pass
    _ENGINE = open_engine(_SF_PATH, threads=1, hash_mb=_HASH_MB)


def _work(task: dict) -> dict:
    """Analyse one game. A position the engine rejects (or an engine crash) skips that
    position only -- one bad position must never sink the whole batch."""
    s = _SETTINGS
    out_rows, cache_new, skipped = [], {}, []
    for pos in task["positions"]:
        board = chess.Board(pos["fen"])
        move = chess.Move.from_uci(pos["move_uci"])
        if board.legal_moves.count() == 0 or move not in board.legal_moves:
            continue
        try:
            core = pos.get("core")
            if core is None:
                core = analyse_core(_ENGINE, board, s)
                cache_new[pos["core_key"]] = core
            if not core["top"]:
                continue
            played = None
            if all(t["uci"] != pos["move_uci"] for t in core["top"]):
                played = pos.get("played")
                if played is None:
                    played = analyse_move(_ENGINE, board, move, s)
                    if played:
                        cache_new[pos["played_key"]] = played
            row = derive(board, move, core, played, pos["prev_move_uci"], pos["clock_before"], task["tc_base"])
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError) as e:
            skipped.append(f"ply {pos['fen']}: {e}")
            _restart_engine()
            continue
        row["id"] = pos["id"]
        out_rows.append(row)
    return {"game_id": task["game_id"], "rows": out_rows, "cache": cache_new, "skipped": skipped}


def _position_key(fen: str) -> str:
    return chess.Board(fen).epd()


def build_tasks(db: DB, cfg: Config, settings: AnalysisSettings, game_ids: list[int] | None,
                reanalyze: bool, limit: int | None = None) -> list[dict]:
    where = ["1=1"]
    params: list = []
    if game_ids:
        where.append(f"id IN ({','.join('?' * len(game_ids))})")
        params += game_ids
    if not reanalyze:
        # only games never analysed; re-analysis at new settings is always explicit (--reanalyze)
        where.append("analysis_key IS NULL")
    # most recent games first, so a partial run is immediately useful
    sql = (f"SELECT id, tc_base, player_color, opponent_is_engine FROM games WHERE {' AND '.join(where)} "
           f"ORDER BY COALESCE(utc_datetime, date) DESC, id DESC")
    if limit:
        sql += f" LIMIT {int(limit)}"
    games = db.q(sql, tuple(params))
    tasks = []
    for g in games:
        include_opp = cfg.analysis.analyze_opponents and not g["opponent_is_engine"]
        rows = db.q("SELECT id, fen, move_uci, prev_move_uci, clock_before, is_player FROM positions "
                    "WHERE game_id = ? ORDER BY ply", (g["id"],))
        positions = []
        for r in rows:
            if not r["is_player"] and not include_opp:
                continue
            key = _position_key(r["fen"])
            core_key = f"{key}|{settings.key}"
            played_key = f"{core_key}|{r['move_uci']}"
            positions.append({
                "id": r["id"], "fen": r["fen"], "move_uci": r["move_uci"], "prev_move_uci": r["prev_move_uci"],
                "clock_before": r["clock_before"], "core_key": core_key, "played_key": played_key,
                "core": db.cache_get(core_key), "played": db.cache_get(played_key),
            })
        tasks.append({"game_id": g["id"], "tc_base": g["tc_base"], "positions": positions})
    return tasks


UPDATE_COLS = ["eval_before_cp", "mate_before", "eval_after_cp", "mate_after", "wp_before", "wp_after", "wp_loss",
               "cp_loss", "accuracy", "best_uci", "best_san", "played_rank", "top_moves", "broad_moves",
               "played_pv", "classification", "phase", "features", "motifs"]


def analyze_games(db: DB, cfg: Config, game_ids: list[int] | None = None, settings: AnalysisSettings | None = None,
                  workers: int | None = None, reanalyze: bool = False, limit: int | None = None,
                  progress: Callable[[str], None] = print) -> int:
    settings = settings or AnalysisSettings(cfg.analysis.deep_depth, cfg.analysis.deep_multipv,
                                            cfg.analysis.broad_depth)
    from ..pgn_import import purge_nonstandard
    removed = purge_nonstandard(db)
    if removed:
        progress(f"Removed {removed} odds/custom-start games (not normal chess; originals kept in data/raw).")
    tasks = build_tasks(db, cfg, settings, game_ids, reanalyze, limit)
    if not tasks:
        progress("Nothing to analyze.")
        return 0
    n_pos = sum(len(t["positions"]) for t in tasks)
    workers = max(1, min(workers or cfg.analysis.workers, len(tasks)))
    progress(f"Analyzing {len(tasks)} games / {n_pos} positions with {workers} Stockfish workers "
             f"(deep d{settings.deep_depth} x{settings.deep_multipv}, broad d{settings.broad_depth})")
    ctx = mp.get_context("spawn")
    start, done_pos = time.time(), 0
    set_sql = ", ".join(f"{c} = ?" for c in UPDATE_COLS)
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg.stockfish(), cfg.analysis.hash_mb, settings)) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, tasks), 1):
            with db.tx() as conn:
                conn.executemany(f"UPDATE positions SET {set_sql} WHERE id = ?",
                                 [tuple(r[c] for c in UPDATE_COLS) + (r["id"],) for r in res["rows"]])
                for k, v in res["cache"].items():
                    db.cache_put(k, v)
                conn.execute("UPDATE games SET analyzed_at = ?, analysis_key = ? WHERE id = ?",
                             (dt.datetime.now().isoformat(timespec="seconds"), settings.key, res["game_id"]))
            done_pos += len(res["rows"])
            if res.get("skipped"):
                progress(f"  game {res['game_id']}: skipped {len(res['skipped'])} position(s) the engine rejected")
            el = time.time() - start
            eta = el / done_pos * (n_pos - done_pos) if done_pos else 0
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                progress(f"  {i}/{len(tasks)} games, {done_pos}/{n_pos} positions, "
                         f"{el:.0f}s elapsed, ~{eta:.0f}s left")
    return len(tasks)
