"""Play against the bot inside the app -- and let it learn from those games.

Opponents
---------
mirror     your clone: moves are sampled from the personal model (it becomes more
           like you with every game you play)
elotest    Stockfish UCI_LimitStrength at an adaptively chosen rating; the result
           feeds the Elo-test estimate automatically
stockfish  Stockfish UCI_LimitStrength at a fixed rating (sparring)

Clocks are kept by the server (time between sending you the position and
receiving your move), so your thinking times are real data.  Finished games are
saved as *your* games (your moves only are training data; the bot is marked as
an engine opponent and never enters the population prior) and a background job
analyses them and retrains the Mirror.
"""

from __future__ import annotations

import datetime as dt
import random
import threading
import time
import uuid

import chess
import chess.engine
import chess.pgn

from .analysis.winprob import cp_to_wp
from .config import Config
from .db import open_db

MIN_ELO, MAX_ELO = 1320, 3190


class PlayManager:
    def __init__(self, cfg: Config, on_saved=None):
        self.cfg = cfg
        self.sessions: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.predictor = None
        self.limited = None
        self.limited_elo = None
        self.on_saved = on_saved

    # ------------------------------------------------------------ engines
    def _pred(self):
        if self.predictor is None:
            from .uci.predictor import LivePredictor
            self.predictor = LivePredictor(self.cfg, threads=2)
        return self.predictor

    def reload_model(self) -> None:
        """Pick up the newest trained model (called after each learning job)."""
        with self.lock:
            if self.predictor is not None:
                self.predictor.close()
                self.predictor = None

    def _limited(self, elo: int):
        from .engine.stockfish import open_engine
        if self.limited is None:
            self.limited = open_engine(self.cfg.stockfish(), threads=1, hash_mb=64)
        if self.limited_elo != elo:
            self.limited.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            self.limited_elo = elo
        return self.limited

    # ------------------------------------------------------------ lifecycle
    def new(self, opts: dict) -> dict:
        color = opts.get("color") or "white"
        if color == "random":
            color = random.choice(["white", "black"])
        opponent = opts.get("opponent") or "mirror"
        elo = None
        if opponent == "elotest":
            from .rating.elotest import current_opponent, prepare_next
            db = open_db(self.cfg)
            prepare_next(db, self.cfg)
            elo = int(current_opponent(self.cfg))
        elif opponent == "stockfish":
            elo = int(max(MIN_ELO, min(MAX_ELO, int(opts.get("elo") or 1500))))
        base = opts.get("base")
        base = float(base) if base else None
        inc = float(opts.get("inc") or 0) if base else 0.0
        s = {
            "id": uuid.uuid4().hex[:10], "player_color": color, "opponent": opponent, "opponent_elo": elo,
            "base": base, "inc": inc, "board": chess.Board(), "moves": [], "clock_after": [],
            "clocks": {chess.WHITE: base, chess.BLACK: base}, "turn_started": time.time(),
            "status": "playing", "result": None, "termination": None, "game_id": None,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "human_timing": bool(opts.get("human_timing")) and opponent == "mirror",
            "bot_info": None, "bad_streak": 0, "model": None,
        }
        with self.lock:
            for old in list(self.sessions.values()):
                if old["status"] == "playing":  # starting over mid-game: keep your moves, no result
                    self._finish(old, "*", "abandoned")
            self.sessions = {s["id"]: s}
            if opponent == "mirror":
                s["model"] = self._pred().model.version
            if not self._players_turn(s):
                self._bot_move(s)
            s["turn_started"] = time.time()
        return self.state(s)

    def get(self, sid: str) -> dict:
        s = self.sessions.get(sid)
        if s is None:
            raise ValueError("That game is no longer available (the app was restarted?)")
        return s

    def current(self) -> dict | None:
        live = [s for s in self.sessions.values() if s["status"] == "playing"]
        return self.state(live[-1]) if live else None

    def _players_turn(self, s: dict) -> bool:
        return (s["board"].turn == chess.WHITE) == (s["player_color"] == "white")

    # ------------------------------------------------------------ moves
    def user_move(self, sid: str, uci: str) -> dict:
        with self.lock:
            s = self.get(sid)
            if s["status"] != "playing":
                return self.state(s)
            if not self._players_turn(s):
                raise ValueError("It is not your move")
            board = s["board"]
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                raise ValueError("Unreadable move")
            if move not in board.legal_moves:
                raise ValueError("Illegal move")
            me = board.turn
            if s["base"]:
                s["clocks"][me] -= time.time() - s["turn_started"]
                if s["clocks"][me] <= 0:
                    s["clocks"][me] = 0.0
                    self._finish(s, self._result(winner=not me), "time forfeit")
                    return self.state(s)
                s["clocks"][me] += s["inc"]
            board.push(move)
            s["moves"].append(uci)
            s["clock_after"].append(s["clocks"][me])
            if not self._check_over(s):
                self._bot_move(s)
            s["turn_started"] = time.time()
            return self.state(s)

    def _bot_move(self, s: dict) -> None:
        board = s["board"]
        bot = board.turn
        t0 = time.time()
        clock = s["clocks"][bot] if s["base"] else None
        if s["opponent"] == "mirror":
            pred = self._pred()
            s["model"] = pred.model.version
            res = pred.decide(board, clock, s["base"], explain=True, temperature=1.0)
            uci = res["uci"]
            top = res["core"]["top"]
            wp = cp_to_wp(top[0]["cp"], top[0]["mate"]) if top else 0.5
            ex = res["explanation"]
            s["bot_info"] = {
                "chosen": board.san(chess.Move.from_uci(uci)),
                "predicted": [{"san": board.san(chess.Move.from_uci(m["uci"])), "p": m["p"],
                               "book": m["book_count"]} for m in ex["moves"]],
                "model": pred.model.version, "win_prob": wp,
            }
            if s["human_timing"] and clock:
                wait = min(20.0, pred.human_think_time(board, clock)) - (time.time() - t0)
                if wait > 0:
                    time.sleep(wait)
        else:
            elo = s["opponent_elo"]
            eng = self._limited(elo)
            # a human pace (Stockfish's own time management would spend far longer per move)
            pace = min(2.0, clock / 60 + s["inc"] * 0.5) if clock else 0.8
            limit = chess.engine.Limit(time=max(0.15, random.uniform(0.6, 1.2) * pace))
            r = eng.play(board, limit, info=chess.engine.INFO_SCORE)
            uci = r.move.uci()
            sc = r.info.get("score")
            wp = cp_to_wp(sc.pov(bot).score(mate_score=10000)) if sc is not None else 0.5
            s["bot_info"] = {"chosen": board.san(r.move), "model": f"Stockfish UCI_Elo {elo}", "win_prob": wp}
        # a human-like opponent resigns hopeless positions instead of playing to mate
        s["bad_streak"] = s["bad_streak"] + 1 if (wp < 0.03 and board.ply() > 30) else 0
        if s["bad_streak"] >= 3:
            self._finish(s, self._result(winner=not bot), "resignation")
            return
        if s["base"]:
            s["clocks"][bot] -= time.time() - t0
            if s["clocks"][bot] <= 0:
                s["clocks"][bot] = 0.0
                self._finish(s, self._result(winner=not bot), "time forfeit")
                return
            s["clocks"][bot] += s["inc"]
        board.push_uci(uci)
        s["moves"].append(uci)
        s["clock_after"].append(s["clocks"][bot])
        self._check_over(s)

    def _check_over(self, s: dict) -> bool:
        out = s["board"].outcome(claim_draw=True)
        if out is None:
            return False
        term = {chess.Termination.CHECKMATE: "checkmate", chess.Termination.STALEMATE: "stalemate",
                chess.Termination.INSUFFICIENT_MATERIAL: "insufficient material",
                chess.Termination.THREEFOLD_REPETITION: "threefold repetition",
                chess.Termination.FIFTY_MOVES: "fifty-move rule"}.get(out.termination, out.termination.name.lower())
        self._finish(s, out.result(), term)
        return True

    @staticmethod
    def _result(winner: chess.Color | None) -> str:
        return "1/2-1/2" if winner is None else ("1-0" if winner == chess.WHITE else "0-1")

    # ------------------------------------------------------------ actions
    def resign(self, sid: str) -> dict:
        with self.lock:
            s = self.get(sid)
            if s["status"] == "playing":
                me = chess.WHITE if s["player_color"] == "white" else chess.BLACK
                self._finish(s, self._result(winner=not me), "resignation")
            return self.state(s)

    def flag(self, sid: str) -> dict:
        """The client's clock hit zero: verify with the server clock."""
        with self.lock:
            s = self.get(sid)
            if s["status"] == "playing" and s["base"] and self._players_turn(s):
                me = s["board"].turn
                if time.time() - s["turn_started"] >= s["clocks"][me] - 0.05:
                    s["clocks"][me] = 0.0
                    self._finish(s, self._result(winner=not me), "time forfeit")
            return self.state(s)

    def offer_draw(self, sid: str) -> dict:
        """The bot accepts only when it is not better and the game is past the opening."""
        with self.lock:
            s = self.get(sid)
            if s["status"] != "playing":
                return self.state(s)
            info = s.get("bot_info") or {}
            wp_bot = info.get("win_prob", 0.5)
            accept = s["board"].ply() >= 40 and wp_bot <= 0.55
            if accept:
                self._finish(s, "1/2-1/2", "agreement")
            st = self.state(s)
            st["draw_offer"] = "accepted" if accept else "declined"
            return st

    def abort(self, sid: str) -> dict:
        with self.lock:
            s = self.get(sid)
            if s["status"] == "playing" and len(s["moves"]) < 2:
                s["status"], s["result"], s["termination"] = "over", "*", "aborted"
            return self.state(s)

    # ------------------------------------------------------------ saving
    def _finish(self, s: dict, result: str, termination: str) -> None:
        s["status"], s["result"], s["termination"] = "over", result, termination
        if len(s["moves"]) < 2:
            return
        try:
            s["game_id"] = self._save(s)
        except Exception as e:  # never lose the UI over a save problem
            s["save_error"] = str(e)
            return
        if self.on_saved:
            self.on_saved(s)

    def _opponent_label(self, s: dict) -> str:
        if s["opponent"] == "mirror":
            return f"Mirror {s.get('model') or ''}".strip()
        return f"Stockfish UCI_Elo {s['opponent_elo']}"

    def _save(self, s: dict) -> int | None:
        from .pgn_import import import_pgn_text
        game = chess.pgn.Game()
        h = game.headers
        h["Event"] = {"mirror": "Chess Mirror: vs your Mirror", "elotest": "Chess Mirror: Elo test",
                      "stockfish": "Chess Mirror: sparring"}[s["opponent"]]
        h["Site"] = "chess-mirror-app"
        h["Date"] = h["UTCDate"] = s["started_at"][:10].replace("-", ".")
        h["UTCTime"] = s["started_at"][11:19]
        me, opp = self.cfg.player.name, self._opponent_label(s)
        h["White"], h["Black"] = (me, opp) if s["player_color"] == "white" else (opp, me)
        if s["opponent_elo"]:
            h["BlackElo" if s["player_color"] == "white" else "WhiteElo"] = str(s["opponent_elo"])
        h["TimeControl"] = f"{int(s['base'])}+{int(s['inc'])}" if s["base"] else "-"
        h["Result"] = s["result"]
        h["Termination"] = s["termination"]
        node = game
        for uci, clk in zip(s["moves"], s["clock_after"]):
            node = node.add_variation(chess.Move.from_uci(uci))
            if clk is not None:
                node.set_clock(max(0.0, clk))
        db = open_db(self.cfg)
        before = db.scalar("SELECT MAX(id) FROM games") or 0
        import_pgn_text(db, self.cfg, str(game), f"app-play:{s['opponent']}", force_color=s["player_color"])
        gid = db.scalar("SELECT MAX(id) FROM games")
        if not gid or gid <= before:
            return None
        db.conn.execute("UPDATE games SET platform = 'mirror-engine', opponent_is_engine = 1 WHERE id = ?", (gid,))
        if s["opponent"] == "elotest" and s["result"] in ("1-0", "0-1", "1/2-1/2"):
            score = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[s["result"]]
            score = score if s["player_color"] == "white" else 1.0 - score
            db.conn.execute("""INSERT INTO elo_games(game_id, played_at, scale, opponent, opponent_rating, result,
                               status, time_control) VALUES (?,?,?,?,?,?,?,?)""",
                            (gid, s["started_at"], "stockfish-uci", "Stockfish UCI_LimitStrength", s["opponent_elo"],
                             score, "final", h["TimeControl"]))
        db.conn.commit()
        return gid

    # ------------------------------------------------------------ view
    def state(self, s: dict) -> dict:
        board = s["board"]
        replay = chess.Board()
        sans = []
        for u in s["moves"]:
            m = chess.Move.from_uci(u)
            sans.append(replay.san(m))
            replay.push(m)
        clocks = dict(s["clocks"])
        if s["base"] and s["status"] == "playing":
            clocks[board.turn] = max(0.0, clocks[board.turn] - (time.time() - s["turn_started"]))
        your_turn = s["status"] == "playing" and self._players_turn(s)
        king = board.king(board.turn) if board.is_check() else None
        return {
            "id": s["id"], "fen": board.fen(), "orientation": s["player_color"], "moves": sans,
            "last": s["moves"][-1] if s["moves"] else None, "turn": "white" if board.turn == chess.WHITE else "black",
            "your_turn": your_turn, "legal": [m.uci() for m in board.legal_moves] if your_turn else [],
            "clocks": {"white": clocks[chess.WHITE], "black": clocks[chess.BLACK]} if s["base"] else None,
            "base": s["base"], "inc": s["inc"], "status": s["status"], "result": s["result"],
            "termination": s["termination"], "opponent": s["opponent"], "opponent_label": self._opponent_label(s),
            "opponent_elo": s["opponent_elo"], "bot": s["bot_info"], "game_id": s["game_id"],
            "check": chess.square_name(king) if king is not None else None, "save_error": s.get("save_error"),
            "player_name": self.cfg.player.name,
        }
