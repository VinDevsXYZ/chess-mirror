"""Concrete move predictors: baselines, logit models and the Mirror composite."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..analysis.features import FEATURE_VERSION, FEATURES, F_INDEX
from .base import MovePredictor
from .dataset import Decision
from .logit import proba

# Human-readable description of what a positive coefficient means.
FEATURE_TEXT = {
    "loss": "avoids moves that lose winning chances", "loss_sqrt": "avoids even small engine losses",
    "is_best": "finds the engine's top move", "top3": "chooses among the engine's top-3",
    "in_deep": "stays within the engine's top-5", "shallow_loss": "avoids moves that look bad at first glance",
    "hidden": "finds moves that only work tactically deep down", "trap": "falls for moves that look good but fail deep down",
    "shallow_best": "plays the move that looks best at first glance",
    "loss_x_tp": "accuracy change under time pressure", "loss_x_sharp": "accuracy change in sharp positions",
    "loss_x_winning": "accuracy change when winning", "loss_x_losing": "accuracy change when losing",
    "loss_x_endgame": "accuracy change in endgames", "loss_x_opening": "accuracy change in the opening",
    "capture": "captures", "cap_value": "captures high-value material", "recapture": "recaptures immediately",
    "check": "gives check", "castle": "castles", "promotion": "promotes",
    "knight": "moves knights", "bishop": "moves bishops", "rook": "moves rooks", "queen": "moves the queen",
    "king": "moves the king", "pawn_push": "pushes pawns", "pawn_storm": "storms pawns at the enemy king",
    "retreat": "retreats pieces", "advance": "advances pieces", "kz_attack": "adds pressure on the enemy king zone",
    "develop": "develops minor pieces", "early_queen": "brings the queen out early", "king_walk": "walks the king (not castling)",
    "center": "occupies the centre", "trade": "offers/accepts equal trades", "queen_trade": "trades queens",
    "trade_x_winning": "simplifies when winning", "trade_x_losing": "trades when losing",
    "sac": "sacrifices material", "sac_x_losing": "sacrifices when losing (swindles)",
    "hangs": "leaves material hanging", "escape": "moves attacked pieces to safety", "threat": "creates threats",
    "fork": "plays forking moves", "pin": "creates pins", "skewer": "creates skewers",
    "discovered": "gives discovered checks", "allows_fork": "allows enemy forks", "allows_mate": "allows mate in one",
    "passed_create": "creates passed pawns", "self_weak": "accepts pawn weaknesses", "shield_weaken": "pushes king-shield pawns",
    "check_x_tp": "gives checks under time pressure",
}

ENGINE_FEATURES = ["loss", "loss_sqrt", "is_best", "top3", "in_deep", "shallow_loss", "hidden", "trap",
                   "shallow_best", "loss_x_tp", "loss_x_sharp", "loss_x_winning", "loss_x_losing",
                   "loss_x_endgame", "loss_x_opening"]
STYLE_FEATURES = [f for f in FEATURES if f not in ENGINE_FEATURES]


class Uniform(MovePredictor):
    kind = name = "random-legal"

    def predict(self, d: Decision) -> np.ndarray:
        return np.full(len(d.ucis), 1.0 / len(d.ucis))


class EngineBest(MovePredictor):
    """Full-strength Stockfish: always the top move (tiny epsilon keeps log-loss finite)."""
    kind = name = "stockfish-best"

    def predict(self, d: Decision) -> np.ndarray:
        n = len(d.ucis)
        p = np.full(n, 1e-3 / max(1, n - 1))
        best = d.top[0]["uci"] if d.top else d.ucis[0]
        p[d.ucis.index(best) if best in d.ucis else 0] = 1 - 1e-3 if n > 1 else 1.0
        return p / p.sum()


class LogitPredictor(MovePredictor):
    kind = "logit"

    def __init__(self, beta: np.ndarray, se: np.ndarray | None = None, name: str = "logit",
                 sigma: float | None = None, n_train: int = 0):
        self.beta = np.asarray(beta, dtype=np.float64)
        self.se = None if se is None else np.asarray(se, dtype=np.float64)
        self.name = name
        self.sigma = sigma
        self.n_train = n_train

    def predict(self, d: Decision) -> np.ndarray:
        return proba(self.beta, d.X)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "name": self.name, "beta": self.beta.tolist(),
                "se": None if self.se is None else self.se.tolist(), "sigma": self.sigma, "n_train": self.n_train}

    @classmethod
    def from_dict(cls, d: dict) -> "LogitPredictor":
        return cls(np.array(d["beta"]), None if d.get("se") is None else np.array(d["se"]), d.get("name", "logit"),
                   d.get("sigma"), d.get("n_train", 0))


class BookModel:
    """Counts of the player's own moves per exact position (training games only)."""

    def __init__(self, counts: dict[str, dict[str, int]] | None = None):
        self.counts = counts or {}

    @classmethod
    def build(cls, decisions: list[Decision]) -> "BookModel":
        c: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for d in decisions:
            c[d.key][d.ucis[d.y]] += 1
        return cls({k: dict(v) for k, v in c.items()})

    def lookup(self, key: str) -> dict[str, int]:
        return self.counts.get(key, {})


def mix_book(p_model: np.ndarray, ucis: list[str], counts: dict[str, int], alpha: float) -> np.ndarray:
    """Dirichlet posterior mean with the model as base measure:
    p(m) = (n_m + alpha * p_model(m)) / (N + alpha)."""
    if not counts:
        return p_model
    n = np.array([counts.get(u, 0) for u in ucis], dtype=np.float64)
    N = n.sum()
    if N == 0:
        return p_model
    return (n + alpha * p_model) / (N + alpha)


class MirrorModel(MovePredictor):
    """Personal logit (shrunk toward the population) + the player's own position book."""
    kind = "mirror-logit-book"

    def __init__(self, personal: LogitPredictor, population: LogitPredictor, book: BookModel,
                 alpha: float = 4.0, meta: dict | None = None, version: str = "unsaved"):
        self.personal = personal
        self.population = population
        self.book = book
        self.alpha = alpha
        self.meta = meta or {}
        self.version = version
        self.name = f"mirror ({version})"

    def predict(self, d: Decision) -> np.ndarray:
        return mix_book(self.personal.predict(d), d.ucis, self.book.lookup(d.key), self.alpha)

    # ------------------------------------------------------------ explanation
    def explain(self, d: Decision, k: int = 4) -> dict:
        """Why the model expects each top move, read directly off the model.

        Each reason is a feature's contribution beta_j * (x_mj - xbar_j): how much it
        lifts/lowers this move relative to the model-weighted average alternative.
        All engine-quality terms are pooled into one 'objective soundness' reason.
        """
        p_model = self.personal.predict(d)
        counts = self.book.lookup(d.key)
        p = mix_book(p_model, d.ucis, counts, self.alpha)
        order = np.argsort(-p)[:k]
        beta, pop = self.personal.beta, self.population.beta
        se = self.personal.se if self.personal.se is not None else np.full(len(beta), np.nan)
        xbar = p_model @ d.X  # model-weighted average candidate
        eng_idx = [F_INDEX[f] for f in ENGINE_FEATURES]
        li = F_INDEX["loss"]
        moves = []
        for i in order:
            contrib = beta * (d.X[i] - xbar)
            reasons = []
            ce = float(contrib[eng_idx].sum())
            if abs(ce) >= 0.15:
                reasons.append({
                    "feature": "engine_quality", "contribution": ce, "evidence": "observed fact",
                    "text": f"{'+' if ce > 0 else '-'} objective soundness: loses {d.X[i, li] * 100:.1f}% win "
                            f"probability vs {xbar[li] * 100:.1f}% for a typical alternative",
                    "beta": float(beta[F_INDEX['loss_sqrt']]), "population_beta": float(pop[F_INDEX['loss_sqrt']]),
                    "z_vs_population": 0.0})
            style = [j for j in np.argsort(-np.abs(contrib)) if j not in eng_idx and abs(contrib[j]) >= 0.15]
            for j in style[:4]:
                reasons.append(self._reason(FEATURES[j], j, d.X[i, j] - xbar[j], contrib[j], beta, pop, se))
            moves.append({"uci": d.ucis[i], "p": float(p[i]), "p_model": float(p_model[i]),
                          "book_count": counts.get(d.ucis[i], 0), "reasons": reasons,
                          "engine_loss": float(d.X[i, li])})
        ctx = []
        if d.ctx and d.ctx.time_pressure:
            ctx.append("time pressure (the model applies your time-pressure behaviour)")
        if counts:
            ctx.append(f"exact position reached {sum(counts.values())}x in your training games")
        return {"moves": moves, "context": ctx, "other_p": float(1 - p[order].sum()),
                "personalization": self.meta.get("sigma_note", "")}

    @staticmethod
    def _reason(name, j, dx, c, beta, pop, se) -> dict:
        z = (beta[j] - pop[j]) / se[j] if se[j] and np.isfinite(se[j]) and se[j] > 0 else 0.0
        # |z| >= 3 ~ Bonferroni-corrected 5% level across the ~54 features
        if abs(z) >= 3.0:
            evidence = "statistical inference"
            cmp_ = f"you do this {'more' if z > 0 else 'less'} than similar-rated players"
        elif abs(z) >= 2.0:
            evidence = "weak hypothesis"
            cmp_ = f"possibly {'more' if z > 0 else 'less'} than similar-rated players"
        else:
            evidence = "population behaviour"
            cmp_ = "about as often as similar-rated players"
        what = FEATURE_TEXT.get(name, name)
        if dx > 0:
            head = f"this move {what}" if not what.startswith(("avoids", "allows", "leaves")) else f"this move: {what}"
        else:
            head = f"unlike the alternatives, it is not a move that {what}"
        text = f"{'+' if c > 0 else '-'} {head} ({cmp_}; weight {beta[j]:+.2f} vs {pop[j]:+.2f})"
        return {"feature": name, "contribution": float(c), "text": text, "beta": float(beta[j]),
                "population_beta": float(pop[j]), "z_vs_population": float(z), "evidence": evidence}

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"kind": self.kind, "version": self.version, "feature_version": FEATURE_VERSION,
                "features": FEATURES, "personal": self.personal.to_dict(), "population": self.population.to_dict(),
                "alpha": self.alpha, "book": self.book.counts, "meta": self.meta}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: Path) -> "MirrorModel":
        d = json.loads(Path(path).read_text())
        if d.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"Model feature version {d.get('feature_version')} != code {FEATURE_VERSION}; retrain.")
        return cls(LogitPredictor.from_dict(d["personal"]), LogitPredictor.from_dict(d["population"]),
                   BookModel(d["book"]), d["alpha"], d.get("meta", {}), d.get("version", "?"))
