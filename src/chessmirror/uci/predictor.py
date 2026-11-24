"""Live decision-making used by the UCI engine and ``mirror explain``.

Stockfish supplies the evaluations (teacher); the Mirror model decides.
"""

from __future__ import annotations

import json
import math
import time

import chess
import numpy as np

from ..analysis.features import Context, phase_of
from ..analysis.winprob import cp_to_wp
from ..config import Config
from ..db import DB, open_db
from ..engine.stockfish import AnalysisSettings, analyse_core, open_engine
from ..model.base import sample_move
from ..model.dataset import make_decision
from ..model.logit import default_prior
from ..model.predictors import BookModel, LogitPredictor, MirrorModel
from ..model.train import load_active


class LivePredictor:
    def __init__(self, cfg: Config, db: DB | None = None, version: str | None = None,
                 think_depth: int | None = None, threads: int = 2):
        self.cfg = cfg
        self.db = db or open_db(cfg)
        try:
            self.model = load_active(self.db, cfg, version or None)
        except Exception as e:  # e.g. feature-version mismatch: keep the engine usable
            import sys
            sys.stderr.write(f"model load failed ({e}); falling back to the generic prior\n")
            self.model = None
        if self.model is None:
            elo = (self.db.scalar("SELECT AVG(player_elo) FROM games WHERE player_elo IS NOT NULL")
                   or cfg.player.rating or 1500)
            prior = LogitPredictor(default_prior(elo), None, "uninformed Elo prior")
            self.model = MirrorModel(prior, prior, BookModel(), 4.0,
                                     {"sigma_note": "no trained model yet: generic Elo-based prior"}, "prior-only")
        self.settings = AnalysisSettings(think_depth or cfg.engine.think_depth, cfg.engine.think_multipv,
                                         cfg.engine.broad_depth)
        self.sf = open_engine(cfg.stockfish(), threads=threads, hash_mb=128)
        self.rng = np.random.default_rng()
        self.time_model = self._time_model()

    def close(self) -> None:
        try:
            self.sf.quit()
        except Exception:
            pass

    # ------------------------------------------------------------ analysis
    def analyse(self, board: chess.Board) -> dict:
        key = f"{board.epd()}|{self.settings.key}"
        core = self.db.cache_get(key)
        if core is None:
            core = analyse_core(self.sf, board, self.settings)
            try:
                self.db.cache_put(key, core)
                self.db.conn.commit()
            except Exception:
                pass
        return core

    def decide(self, board: chess.Board, clock: float | None = None, base: float | None = None,
               explain: bool = False, temperature: float = 1.0, use_book: bool = True) -> dict:
        t0 = time.time()
        core = self.analyse(board)
        prev = board.peek().uci() if board.move_stack else None
        d = make_decision(board.fen(), core["top"], core["broad"], Context(clock, base, prev))
        if use_book:
            p = self.model.predict(d)
        else:
            p = self.model.personal.predict(d)
        idx = sample_move(p, temperature, self.rng)
        res = {"uci": d.ucis[idx], "p": p, "decision": d, "core": core, "index": idx,
               "compute_s": time.time() - t0, "model": self.model.version}
        if explain:
            res["explanation"] = self.model.explain(d)
        return res

    def lines(self, res: dict, n: int) -> list[dict]:
        """Top-n candidates by modelled probability, each with Stockfish's own score/PV."""
        d, p, core = res["decision"], res["p"], res["core"]
        deep = {t["uci"]: t for t in core["top"]}
        out = []
        for i in np.argsort(-p)[:n]:
            u = d.ucis[i]
            if u in deep:
                t = deep[u]
                out.append({"uci": u, "p": float(p[i]), "cp": t["cp"], "mate": t["mate"], "pv": t["pv"]})
            else:
                out.append({"uci": u, "p": float(p[i]), "cp": core["broad"].get(u, 0), "mate": None, "pv": [u]})
        return out

    # ------------------------------------------------------------ timing
    def _time_model(self) -> dict:
        """Fraction of remaining clock the player spends per move, by phase (log-normal fit)."""
        rows = self.db.q("""SELECT phase, time_spent, clock_before FROM positions
                            WHERE is_player = 1 AND time_spent IS NOT NULL AND clock_before > 5 AND phase IS NOT NULL""")
        out = {}
        for ph in ("opening", "middlegame", "endgame"):
            fr = [r["time_spent"] / r["clock_before"] for r in rows if r["phase"] == ph and r["time_spent"] > 0]
            if len(fr) >= 30:
                lf = np.log(np.clip(fr, 1e-3, 1.0))
                out[ph] = {"mu": float(lf.mean()), "sigma": float(lf.std()), "n": len(fr)}
        return out

    def human_think_time(self, board: chess.Board, clock: float) -> float:
        m = self.time_model.get(phase_of(board))
        if not m:
            frac = 0.03
        else:
            frac = math.exp(self.rng.normal(m["mu"], m["sigma"]))
        return float(np.clip(frac * clock, 0.2, max(0.2, 0.25 * clock)))

    # ------------------------------------------------------------ text
    def format_explanation(self, res: dict) -> str:
        ex = res.get("explanation") or self.model.explain(res["decision"])
        board = chess.Board(res["decision"].fen)
        lines = [f"Model: {self.model.version}   ({self.model.meta.get('sigma_note', '')})", "Predicted moves:"]
        for m in ex["moves"]:
            san = board.san(chess.Move.from_uci(m["uci"]))
            extra = f", book {m['book_count']}x" if m["book_count"] else ""
            lines.append(f"  {san:8s} {m['p'] * 100:5.1f}%   (engine loss {m['engine_loss'] * 100:.1f}% win prob{extra})")
        lines.append(f"  other    {ex['other_p'] * 100:5.1f}%")
        for c in ex["context"]:
            lines.append(f"Context: {c}")
        for m in ex["moves"][:3]:
            san = board.san(chess.Move.from_uci(m["uci"]))
            lines.append(f"Why {san}:")
            if m["book_count"]:
                lines.append(f"  - observed fact: you played it {m['book_count']}x from this exact position (training games)")
            for r in m["reasons"]:
                lines.append(f"  {r['text']}  [{r['evidence']}]")
        return "\n".join(lines)

    def explanation_json(self, res: dict) -> str:
        ex = res.get("explanation") or self.model.explain(res["decision"])
        board = chess.Board(res["decision"].fen)
        for m in ex["moves"]:
            m["san"] = board.san(chess.Move.from_uci(m["uci"]))
        wp = cp_to_wp(res["core"]["top"][0]["cp"], res["core"]["top"][0]["mate"]) if res["core"]["top"] else None
        return json.dumps({"fen": res["decision"].fen, "chosen": res["uci"], "model": res["model"],
                           "win_prob": wp, "explanation": ex, "time": time.time()})
