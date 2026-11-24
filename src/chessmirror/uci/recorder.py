"""Record games played against the engine (Mirror / Elo test) in En Croissant.

En Croissant never tells the engine a game ended and kills the process when it
is done, so state is persisted after every command.  A sequence of positions is
treated as a *played game* only if each new position equals the previous one
plus the engine's own bestmove plus one reply -- analysis-board browsing never
looks like that, so it is not recorded.

Games whose final position is decisive are imported automatically; others wait
in ``state/pending`` until the result is given (``mirror games finalize``).
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

import chess
import chess.pgn

from ..config import Config

MIN_ENGINE_MOVES = 4


class GameRecorder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.state_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "pending").mkdir(exist_ok=True)
        self.path = self.dir / "current_game.json"
        self.cur: dict | None = None
        if self.path.exists():  # leftover from a killed process = a finished game
            try:
                self.cur = json.loads(self.path.read_text())
            except Exception:
                self.cur = None
            self.close_game()

    def _save(self) -> None:
        if self.cur:
            self.path.write_text(json.dumps(self.cur))
        elif self.path.exists():
            self.path.unlink()

    def on_position(self, root_fen: str, moves: list[str]) -> None:
        c = self.cur
        if c and c["root"] == root_fen and moves[:len(c["moves"])] == c["moves"] \
                and len(moves) == len(c["moves"]) + 2 and c.get("last_best") == moves[len(c["moves"])]:
            c["moves"] = list(moves)
            c["consistent"] += 1
        elif c and c["root"] == root_fen and moves == c["moves"]:
            pass  # same position re-sent
        else:
            self.close_game()
            self.cur = {"id": uuid.uuid4().hex[:10], "root": root_fen, "moves": list(moves), "consistent": 0,
                        "clocks": {}, "started": dt.datetime.now().isoformat(timespec="seconds"),
                        "analysis": False, "mode": None, "engine_color": None, "opponent_rating": None,
                        "model": None, "tc": None}
        self._save()

    def on_go(self, ply: int, engine_color: str, mode: str, wtime, btime, winc, binc, infinite: bool,
              opponent_rating: float | None, model: str | None) -> None:
        c = self.cur
        if not c:
            return
        if infinite:
            c["analysis"] = True
        c["mode"] = c["mode"] or mode
        c["engine_color"] = c["engine_color"] or engine_color
        c["opponent_rating"] = c["opponent_rating"] or opponent_rating
        c["model"] = c["model"] or model
        if wtime is not None and btime is not None:
            c["clocks"][str(ply)] = [wtime / 1000.0, btime / 1000.0]
            if not c["tc"]:
                inc = (winc or 0) / 1000.0
                c["tc"] = [round(max(wtime, btime) / 1000.0), round(inc)]
        self._save()

    def on_bestmove(self, uci: str) -> None:
        if self.cur:
            self.cur["last_best"] = uci
            self._save()

    def close_game(self) -> None:
        c, self.cur = self.cur, None
        if self.path.exists():
            self.path.unlink()
        if not c or c.get("analysis") or c["consistent"] < MIN_ENGINE_MOVES:
            return
        if c.get("last_best"):
            b = chess.Board(c["root"])
            try:
                for m in c["moves"]:
                    b.push_uci(m)
                if chess.Move.from_uci(c["last_best"]) in b.legal_moves:
                    c["moves"] = c["moves"] + [c["last_best"]]
            except ValueError:
                pass
        b = chess.Board(c["root"])
        for m in c["moves"]:
            b.push_uci(m)
        outcome = b.outcome(claim_draw=True)
        c["result"] = outcome.result() if outcome else None
        c["ended"] = dt.datetime.now().isoformat(timespec="seconds")
        (self.dir / "pending" / f"{c['id']}.json").write_text(json.dumps(c))
        if c["result"]:
            try:
                from ..db import open_db
                finalize_pending(open_db(self.cfg), self.cfg, c["id"], c["result"])
            except Exception:
                pass


def build_pgn(c: dict, player_name: str, result: str) -> str:
    game = chess.pgn.Game()
    board = chess.Board(c["root"])
    if c["root"] != chess.STARTING_FEN:
        game.setup(board)
    engine_white = c["engine_color"] == "white"
    if c["mode"] == "EloTest":
        opp = f"Stockfish UCI_Elo {int(c['opponent_rating'] or 0)}"
    else:
        opp = f"Mirror {c.get('model') or ''}".strip()
    h = game.headers
    h["Event"] = f"{c['mode'] or 'Mirror'} game vs engine"
    h["Site"] = "mirror-engine"
    day = c["started"][:10]
    h["Date"] = h["UTCDate"] = day.replace("-", ".")
    h["UTCTime"] = c["started"][11:19]
    h["White"], h["Black"] = (opp, player_name) if engine_white else (player_name, opp)
    if c["mode"] == "EloTest" and c.get("opponent_rating"):
        h["WhiteElo" if engine_white else "BlackElo"] = str(int(c["opponent_rating"]))
    if c.get("tc"):
        h["TimeControl"] = f"{c['tc'][0]}+{c['tc'][1]}"
    h["Result"] = result
    node = game
    clocks = {int(k): v for k, v in c.get("clocks", {}).items()}
    for i, m in enumerate(c["moves"]):
        node = node.add_variation(chess.Move.from_uci(m))
        # clocks at the engine's next "go" show the mover's remaining time after this move
        nxt = clocks.get(i + 1)
        if nxt:
            node.set_clock(nxt[0] if board.turn == chess.WHITE else nxt[1])
        board.push_uci(m)
    return str(game)


def pending_list(cfg: Config) -> list[dict]:
    d = cfg.state_dir / "pending"
    if not d.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def pending_text(cfg: Config) -> str:
    items = pending_list(cfg)
    if not items:
        return "No games waiting for a result."
    lines = ["Games recorded by the engine that need a result (`mirror games finalize --id ID --result 1-0`):"]
    for c in items:
        lines.append(f"  {c['id']}  {c['started']}  {c['mode'] or 'Mirror'}  you={'black' if c['engine_color'] == 'white' else 'white'}"
                     f"  {len(c['moves'])} plies  opponent={c.get('opponent_rating') or c.get('model')}")
    return "\n".join(lines)


def finalize_pending(db, cfg: Config, gid: str, result: str) -> str:
    from ..pgn_import import import_pgn_text
    path = cfg.state_dir / "pending" / f"{gid}.json"
    if not path.exists():
        return f"No pending game {gid}"
    c = json.loads(path.read_text())
    pgn = build_pgn(c, cfg.player.name, result)
    before = db.scalar("SELECT MAX(id) FROM games") or 0
    import_pgn_text(db, cfg, pgn, f"mirror-engine:{c['mode'] or 'Mirror'}",
                    force_color="black" if c["engine_color"] == "white" else "white")
    game_id = db.scalar("SELECT MAX(id) FROM games")
    if game_id and game_id > before:
        # engine games are never used as population data and never enter the Elo pools of real platforms
        db.conn.execute("UPDATE games SET platform = 'mirror-engine', opponent_is_engine = 1 WHERE id = ?", (game_id,))
        if c["mode"] == "EloTest" and c.get("opponent_rating"):
            human_white = c["engine_color"] != "white"
            score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[result]
            score = score if human_white else 1.0 - score
            db.conn.execute("""INSERT INTO elo_games(game_id, played_at, scale, opponent, opponent_rating, result,
                               status, time_control) VALUES (?,?,?,?,?,?,?,?)""",
                            (game_id, c["started"], "stockfish-uci", "Stockfish UCI_LimitStrength",
                             c["opponent_rating"], score, "final",
                             f"{c['tc'][0]}+{c['tc'][1]}" if c.get("tc") else None))
        db.conn.commit()
    path.unlink()
    if c["mode"] == "EloTest":
        from ..rating.elotest import prepare_next
        return f"Recorded game {gid} ({result}). " + prepare_next(db, cfg)
    return f"Recorded game {gid} ({result}) as game #{game_id}. Run `mirror analyze` to include it."
