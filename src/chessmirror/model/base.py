"""Model interface.  Every move model -- baseline, logit, future neural net --
implements ``predict(decision) -> probabilities over decision.ucis`` so that the
UCI engine, evaluator and explainer never depend on a concrete model class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .dataset import Decision


class MovePredictor(ABC):
    kind: str = "abstract"
    name: str = "model"

    @abstractmethod
    def predict(self, d: Decision) -> np.ndarray:
        """Probability for each move in ``d.ucis`` (sums to 1)."""

    def explain(self, d: Decision, k: int = 5) -> dict:  # pragma: no cover - optional
        p = self.predict(d)
        order = np.argsort(-p)[:k]
        return {"moves": [{"uci": d.ucis[i], "p": float(p[i])} for i in order], "reasons": []}

    def to_dict(self) -> dict:
        raise NotImplementedError


def sample_move(p: np.ndarray, temperature: float = 1.0, rng: np.random.Generator | None = None) -> int:
    """Sample a move index.  temperature=1 reproduces the modelled distribution,
    ->0 picks the single most likely move."""
    rng = rng or np.random.default_rng()
    if temperature <= 0.05:
        return int(np.argmax(p))
    logp = np.log(np.clip(p, 1e-12, 1.0)) / temperature
    logp -= logp.max()
    q = np.exp(logp)
    q /= q.sum()
    return int(rng.choice(len(q), p=q))
