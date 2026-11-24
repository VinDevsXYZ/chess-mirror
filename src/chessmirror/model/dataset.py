"""Decision datasets for move models.

A *decision* is one position where someone chose a move.  Its candidate set is
every legal move, each with a feature vector (``analysis.features``).  Feature
matrices are cached (zlib-compressed) in the database keyed by position id.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import zlib
from dataclasses import dataclass, field

import chess
import numpy as np

from ..analysis.features import FEATURE_VERSION, N_FEATURES, Context, candidate_features
from ..analysis.winprob import cp_to_wp
from ..db import DB


@dataclass
class Decision:
    fen: str
    ucis: list[str]
    X: np.ndarray
    top: list[dict]
    broad: dict[str, int]
    ctx: Context
    y: int | None = None            # index of the played move (None at inference)
    position_id: int | None = None
    game_id: int | None = None
    ply: int | None = None
    phase: str | None = None
    wp_before: float | None = None
    losses: np.ndarray | None = None  # engine wp loss per candidate (for style/CPL simulation)
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return chess.Board(self.fen).epd()


def make_decision(fen: str, top: list[dict], broad: dict[str, int], ctx: Context, **kw) -> Decision:
    board = chess.Board(fen)
    wp_before = cp_to_wp(top[0].get("cp"), top[0].get("mate")) if top else 0.5
    ucis, X = candidate_features(board, top, broad, ctx, wp_before)
    return Decision(fen=fen, ucis=ucis, X=X, top=top, broad=broad, ctx=ctx, wp_before=wp_before,
                    losses=X[:, 0].copy(), **kw)


# ---------------------------------------------------------------- feature cache
_CACHE_SQL = """CREATE TABLE IF NOT EXISTS cand_features (
    position_id INTEGER PRIMARY KEY, feature_version TEXT, ucis TEXT, X BLOB, sig INTEGER)"""


def _signature(r) -> int:
    """Changes whenever the inputs of candidate_features change (e.g. re-analysis at other settings)."""
    return zlib.crc32(f"{FEATURE_VERSION}|{r['top_moves']}|{r['broad_moves']}|{r['clock_before']}|"
                      f"{r['tc_base']}|{r['prev_move_uci']}".encode())


def _compute(args):
    pid, fen, top, broad, clock, base, prev, sig = args
    board = chess.Board(fen)
    ucis, X = candidate_features(board, top, broad, Context(clock, base, prev))
    return pid, ucis, zlib.compress(X.astype(np.float32).tobytes(), 3), sig


def ensure_features(db: DB, rows: list, workers: int = 8, progress=None) -> None:
    db.conn.execute(_CACHE_SQL)
    cols = {r[1] for r in db.q("PRAGMA table_info(cand_features)")}
    if "sig" not in cols:
        db.conn.execute("ALTER TABLE cand_features ADD COLUMN sig INTEGER")
    have = {r[0]: r[1] for r in db.q("SELECT position_id, sig FROM cand_features WHERE feature_version = ?",
                                     (FEATURE_VERSION,))}
    todo = []
    for r in rows:
        sig = _signature(r)
        if have.get(r["id"]) != sig:
            todo.append((r["id"], r["fen"], json.loads(r["top_moves"]), json.loads(r["broad_moves"]),
                         r["clock_before"], r["tc_base"], r["prev_move_uci"], sig))
    if not todo:
        return
    if progress:
        progress(f"Computing candidate features for {len(todo)} positions...")
    if len(todo) < 200 or workers <= 1:
        results = map(_compute, todo)
        pool = None
    else:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(_compute, todo, chunksize=32)
    batch = []
    for pid, ucis, blob, sig in results:
        batch.append((pid, FEATURE_VERSION, json.dumps(ucis), blob, sig))
        if len(batch) >= 500:
            db.conn.executemany("INSERT OR REPLACE INTO cand_features(position_id, feature_version, ucis, X, sig) "
                                "VALUES (?,?,?,?,?)", batch)
            db.conn.commit()
            batch = []
    if batch:
        db.conn.executemany("INSERT OR REPLACE INTO cand_features(position_id, feature_version, ucis, X, sig) "
                            "VALUES (?,?,?,?,?)", batch)
        db.conn.commit()
    if pool:
        pool.close()
        pool.join()


POSITION_SQL = """
SELECT p.id, p.game_id, p.ply, p.fen, p.move_uci, p.prev_move_uci, p.clock_before, p.top_moves, p.broad_moves,
       p.phase, p.wp_before, p.wp_loss, p.is_player, p.features, p.motifs, g.tc_base, g.split, g.player_color
FROM positions p JOIN games g ON g.id = p.game_id
WHERE p.top_moves IS NOT NULL AND {where}
ORDER BY p.game_id, p.ply
"""


def load_decisions(db: DB, where: str, params: tuple = (), workers: int = 8, progress=None) -> list[Decision]:
    rows = db.q(POSITION_SQL.format(where=where), params)
    ensure_features(db, rows, workers, progress)
    feats = {}
    ids = [r["id"] for r in rows]
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        for pid, ucis, blob in db.q(
                f"SELECT position_id, ucis, X FROM cand_features WHERE feature_version = ? AND position_id IN "
                f"({','.join('?' * len(chunk))})", (FEATURE_VERSION, *chunk)):
            feats[pid] = (json.loads(ucis), blob)
    out = []
    for r in rows:
        if r["id"] not in feats:
            continue
        ucis, blob = feats[r["id"]]
        X = np.frombuffer(zlib.decompress(blob), dtype=np.float32).reshape(len(ucis), N_FEATURES)
        if r["move_uci"] not in ucis:
            continue
        top = json.loads(r["top_moves"])
        out.append(Decision(
            fen=r["fen"], ucis=ucis, X=X, top=top, broad=json.loads(r["broad_moves"]),
            ctx=Context(r["clock_before"], r["tc_base"], r["prev_move_uci"]),
            y=ucis.index(r["move_uci"]), position_id=r["id"], game_id=r["game_id"], ply=r["ply"],
            phase=r["phase"], wp_before=r["wp_before"], losses=X[:, 0].copy(),
            meta={"wp_loss": r["wp_loss"], "features": r["features"], "motifs": r["motifs"],
                  "is_player": r["is_player"]},
        ))
    return out


# ------------------------------------------------------------- stacked arrays
@dataclass
class Stacked:
    X: np.ndarray          # (rows, K)
    offsets: np.ndarray    # (n+1,) row offsets per decision
    y: np.ndarray          # (n,) global row index of the chosen move
    seg: np.ndarray        # (rows,) decision index of each row

    @property
    def n(self) -> int:
        return len(self.y)


def stack(decisions: list[Decision]) -> Stacked:
    sizes = np.array([len(d.ucis) for d in decisions], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    X = np.concatenate([d.X for d in decisions]).astype(np.float64) if decisions else np.zeros((0, N_FEATURES))
    y = offsets[:-1] + np.array([d.y for d in decisions], dtype=np.int64)
    seg = np.repeat(np.arange(len(decisions)), sizes)
    return Stacked(X, offsets, y, seg)
