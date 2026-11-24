"""UCI front-end: to En Croissant this is an ordinary chess engine.

Modes
-----
Mirror   the move is *sampled from the personal model's distribution*
         (Temperature=100 reproduces the modelled distribution, 0 = most likely move).
         In analysis (``go infinite``) the MultiPV lines are the most likely
         moves *for the player*, ordered by probability, each shown with
         Stockfish's real evaluation of that move.
EloTest  plays a calibrated opponent (Stockfish UCI_LimitStrength at a rating
         chosen adaptively from the current strength posterior) and records the
         game for the rating estimate.

Protocol notes for En Croissant 0.15 (verified against its source):
  * it only sends ``position fen <fen> [moves ...]`` and never ``ucinewgame``;
  * during analysis it sends ``stop`` and 50 ms later a new position without
    waiting for ``bestmove``, so stale results are suppressed by a search id;
  * at least one ``info ... pv`` line must precede ``bestmove``.
"""

from __future__ import annotations

import sys
import threading
import time
import traceback

import chess
import chess.engine

from ..config import load_config

NAME = "Mirror"
AUTHOR = "Personal Chess Mirror"


class UCIEngine:
    def __init__(self):
        self.cfg = load_config()
        self.opts = {
            "Mode": "Mirror", "Temperature": 100, "UseBook": True, "Explain": False, "HumanTiming": False,
            "RecordGames": True, "EloTestOpponent": 0, "ModelVersion": "", "MultiPV": 3,
            "ThinkDepth": self.cfg.engine.think_depth,
        }
        self.board = chess.Board()
        self.root_fen = chess.STARTING_FEN
        self.moves: list[str] = []
        self.predictor = None
        self.limited_sf = None
        self.limited_elo = None
        self.out_lock = threading.Lock()
        self.search_id = 0
        self.thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.recorder = None
        self.bases: dict[bool, float] = {}

    # ------------------------------------------------------------ io
    def send(self, line: str) -> None:
        with self.out_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def log(self, msg: str) -> None:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()

    # ------------------------------------------------------------ lazy init
    def ensure_ready(self) -> None:
        if self.predictor is None:
            from .predictor import LivePredictor
            self.predictor = LivePredictor(self.cfg, version=self.opts["ModelVersion"] or None,
                                           think_depth=int(self.opts["ThinkDepth"]))
        if self.recorder is None:
            from .recorder import GameRecorder
            self.recorder = GameRecorder(self.cfg)

    # ------------------------------------------------------------ commands
    def cmd_uci(self) -> None:
        self.send(f"id name {NAME}")
        self.send(f"id author {AUTHOR}")
        self.send("option name Mode type combo default Mirror var Mirror var EloTest")
        self.send("option name Temperature type spin default 100 min 0 max 200")
        self.send("option name UseBook type check default true")
        self.send("option name Explain type check default false")
        self.send("option name HumanTiming type check default false")
        self.send("option name RecordGames type check default true")
        self.send("option name EloTestOpponent type spin default 0 min 0 max 3190")
        self.send("option name ModelVersion type string default <active>")
        self.send("option name MultiPV type spin default 3 min 1 max 10")
        self.send(f"option name ThinkDepth type spin default {self.cfg.engine.think_depth} min 6 max 22")
        self.send("uciok")

    def cmd_setoption(self, args: list[str]) -> None:
        if "name" not in args:
            return
        i = args.index("name")
        j = args.index("value") if "value" in args else len(args)
        name = " ".join(args[i + 1:j])
        value = " ".join(args[j + 1:]) if j < len(args) else ""
        for k in self.opts:
            if k.lower() == name.lower():
                cur = self.opts[k]
                if isinstance(cur, bool):
                    self.opts[k] = value.lower() == "true"
                elif isinstance(cur, int):
                    try:
                        self.opts[k] = int(value)
                    except ValueError:
                        pass
                else:
                    self.opts[k] = "" if value in ("<active>", "<empty>") else value
                if k in ("ModelVersion", "ThinkDepth") and self.predictor is not None:
                    self.predictor.close()
                    self.predictor = None

    def cmd_position(self, args: list[str]) -> None:
        if not args:
            return
        if args[0] == "startpos":
            fen, rest = chess.STARTING_FEN, args[1:]
        elif args[0] == "fen":
            k = args.index("moves") if "moves" in args else len(args)
            fen, rest = " ".join(args[1:k]), args[k:]
        else:
            return
        moves = rest[1:] if rest and rest[0] == "moves" else []
        board = chess.Board(fen)
        for m in moves:
            board.push_uci(m)
        if fen != self.root_fen or moves[:len(self.moves)] != self.moves:
            self.bases = {}  # new game / new line: forget initial clocks
        self.root_fen, self.moves, self.board = fen, moves, board
        if self.opts["RecordGames"]:
            self.ensure_ready()
            self.recorder.on_position(fen, moves)

    def cmd_go(self, args: list[str]) -> None:
        params = {}
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("wtime", "btime", "winc", "binc", "movetime", "depth", "nodes", "movestogo"):
                try:
                    params[a] = int(args[i + 1])
                except (IndexError, ValueError):
                    pass
                i += 2
            else:
                params[a] = True
                i += 1
        self.ensure_ready()
        self.stop_flag.clear()
        self.search_id += 1
        sid = self.search_id
        board = self.board.copy()
        self.thread = threading.Thread(target=self._search_safe, args=(sid, board, params), daemon=True)
        self.thread.start()

    def cmd_stop(self) -> None:
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=0.04)
            if self.thread.is_alive():
                # answer immediately; the stale computation is discarded by its search id
                self.search_id += 1
                self._emit_fallback(self.board)

    # ------------------------------------------------------------ search
    def _search_safe(self, sid: int, board: chess.Board, params: dict) -> None:
        try:
            self._search(sid, board, params)
        except Exception as e:  # never die silently: En Croissant would count it as abandonment
            self.log("search error: " + "".join(traceback.format_exception(e)))
            if sid == self.search_id:
                self._emit_fallback(board)

    def _emit_fallback(self, board: chess.Board) -> None:
        legal = list(board.legal_moves)
        if not legal:
            return
        self.send(f"info depth 1 score cp 0 pv {legal[0].uci()}")
        self.send(f"bestmove {legal[0].uci()}")

    def _clock(self, board: chess.Board, params: dict) -> tuple[float | None, float | None]:
        key = "wtime" if board.turn == chess.WHITE else "btime"
        if key not in params or params[key] >= 4294967295:
            return None, None
        clock = params[key] / 1000.0
        self.bases.setdefault(board.turn, clock)
        return clock, self.bases[board.turn]

    def _search(self, sid: int, board: chess.Board, params: dict) -> None:
        if board.legal_moves.count() == 0:
            self.send("info depth 0 score cp 0")
            self.send("bestmove 0000")
            return
        infinite = bool(params.get("infinite"))
        mode = self.opts["Mode"]
        clock, base = self._clock(board, params)
        opp_rating = None
        if mode == "EloTest":
            opp_rating = self._elotest_rating()
        if self.opts["RecordGames"] and self.recorder:
            self.recorder.on_go(len(self.moves), "white" if board.turn == chess.WHITE else "black", mode,
                                params.get("wtime"), params.get("btime"), params.get("winc"), params.get("binc"),
                                infinite, opp_rating, self.predictor.model.version)
        if mode == "EloTest":
            move = self._play_limited(board, params, opp_rating, sid)
            if move and sid == self.search_id:
                self._finish(move)
            return

        t0 = time.time()
        temp = self.opts["Temperature"] / 100.0
        res = self.predictor.decide(board, clock, base, explain=self.opts["Explain"], temperature=temp,
                                    use_book=self.opts["UseBook"])
        if sid != self.search_id:
            return
        depth = self.predictor.settings.deep_depth
        ms = int((time.time() - t0) * 1000)
        if infinite:
            n = max(1, min(int(self.opts["MultiPV"]), board.legal_moves.count()))
            for k, ln in enumerate(self.predictor.lines(res, n), 1):
                self.send(f"info depth {depth} seldepth {depth} multipv {k} {self._score(ln)} nodes {k} "
                          f"nps 1000 time {ms} pv {' '.join(ln['pv'])}")
            self._explain_out(res)
            self.stop_flag.wait()
            if sid == self.search_id:
                best = self.predictor.lines(res, 1)[0]["uci"]
                self.send(f"bestmove {best}")
            return

        chosen = res["uci"]
        ln = next((x for x in self.predictor.lines(res, len(res["p"])) if x["uci"] == chosen), None)
        pv = ln["pv"] if ln else [chosen]
        self._explain_out(res)
        if self.opts["HumanTiming"] and clock:
            wait = self.predictor.human_think_time(board, clock) - (time.time() - t0)
            if wait > 0:
                self.stop_flag.wait(timeout=wait)
        if sid != self.search_id:
            return
        self.send(f"info depth {depth} {self._score(ln or {'cp': 0, 'mate': None})} time {ms} pv {' '.join(pv)}")
        self._finish(chosen)

    def _finish(self, uci: str) -> None:
        if self.opts["RecordGames"] and self.recorder:
            self.recorder.on_bestmove(uci)
        self.send(f"bestmove {uci}")

    @staticmethod
    def _score(ln: dict) -> str:
        if ln.get("mate") is not None:
            return f"score mate {ln['mate']}"
        return f"score cp {int(ln.get('cp') or 0)}"

    def _explain_out(self, res: dict) -> None:
        if not self.opts["Explain"]:
            return
        try:
            for line in self.predictor.format_explanation(res).splitlines():
                self.send(f"info string {line}")
            (self.cfg.state_dir / "last_decision.json").write_text(self.predictor.explanation_json(res))
        except Exception as e:
            self.log(f"explain failed: {e}")

    # ------------------------------------------------------------ Elo test
    def _elotest_rating(self) -> float:
        if self.opts["EloTestOpponent"] >= 1320:
            return float(self.opts["EloTestOpponent"])
        from ..rating.elotest import current_opponent
        return float(current_opponent(self.cfg))

    def _play_limited(self, board: chess.Board, params: dict, elo: float, sid: int) -> str | None:
        from ..engine.stockfish import open_engine
        elo = int(max(1320, min(3190, elo)))
        if self.limited_sf is None:
            self.limited_sf = open_engine(self.cfg.stockfish(), threads=1, hash_mb=64)
        if self.limited_elo != elo:
            self.limited_sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            self.limited_elo = elo
        kw = {}
        if "wtime" in params and params["wtime"] < 4294967295:
            kw.update(white_clock=params["wtime"] / 1000, black_clock=params.get("btime", 0) / 1000,
                      white_inc=params.get("winc", 0) / 1000, black_inc=params.get("binc", 0) / 1000)
        if "movetime" in params:
            kw["time"] = params["movetime"] / 1000
        if "depth" in params:
            kw["depth"] = params["depth"]
        if "nodes" in params:
            kw["nodes"] = params["nodes"]
        if not kw:
            kw["time"] = 1.0
        r = self.limited_sf.play(board, chess.engine.Limit(**kw), info=chess.engine.INFO_SCORE)
        if sid != self.search_id or r.move is None:
            return None
        sc = r.info.get("score")
        score = "score cp 0"
        if sc is not None:
            pov = sc.pov(board.turn)
            score = f"score mate {pov.mate()}" if pov.is_mate() else f"score cp {pov.score()}"
        self.send(f"info depth 1 {score} pv {r.move.uci()}")
        self.send(f"info string EloTest opponent: Stockfish UCI_Elo {elo}")
        return r.move.uci()

    # ------------------------------------------------------------ main loop
    def run(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            cmd, args = parts[0], parts[1:]
            try:
                if cmd == "uci":
                    self.cmd_uci()
                elif cmd == "isready":
                    self.ensure_ready()
                    self.send("readyok")
                elif cmd == "setoption":
                    self.cmd_setoption(args)
                elif cmd == "ucinewgame":
                    self.bases = {}
                elif cmd == "position":
                    self.cmd_position(args)
                elif cmd == "go":
                    self.cmd_go(args)
                elif cmd == "stop":
                    self.cmd_stop()
                elif cmd == "quit":
                    break
                elif cmd == "d":
                    self.send(str(self.board))
            except Exception as e:
                self.log("command error: " + "".join(traceback.format_exception(e)))
        self.stop_flag.set()
        for obj in (self.predictor, self.limited_sf):
            try:
                if obj is not None:
                    (obj.close if hasattr(obj, "close") else obj.quit)()
            except Exception:
                pass


def main() -> None:
    UCIEngine().run()


if __name__ == "__main__":
    main()
