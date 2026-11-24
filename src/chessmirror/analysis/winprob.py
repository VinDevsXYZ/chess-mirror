"""Evaluation -> win probability, accuracy and move classification.

Uses the published Lichess conventions so numbers are comparable with what the
player sees on Lichess:
  * win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)
  * move accuracy = 103.1668 * exp(-0.04354 * win%_drop) - 3.1669
  * inaccuracy / mistake / blunder = drop in winning chances >= 0.1 / 0.2 / 0.3
    on the [-1, 1] scale, i.e. win-probability drops of 0.05 / 0.10 / 0.15.
"""

from __future__ import annotations

import math

MATE_CP = 10000

INACCURACY = 0.05
MISTAKE = 0.10
BLUNDER = 0.15


def score_to_cp(cp: int | None, mate: int | None) -> int:
    """Collapse (cp, mate) into a single centipawn number for sorting/storage."""
    if mate is not None:
        if mate > 0:
            return MATE_CP - mate
        return -MATE_CP - mate  # mate<=0: being mated (mate=0 means already mated)
    return int(cp or 0)


def cp_to_wp(cp: int | None, mate: int | None = None) -> float:
    """Win probability (0..1) for the side the score is from."""
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    if cp is None:
        return 0.5
    if abs(cp) >= MATE_CP - 500:
        return 1.0 if cp > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-0.00368208 * cp))


def move_accuracy(wp_before: float, wp_after: float) -> float:
    drop = max(0.0, (wp_before - wp_after) * 100.0)
    return max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * drop) - 3.1669))


def classify(wp_loss: float, rank: int | None, n_legal: int) -> str:
    if n_legal == 1:
        return "forced"
    if wp_loss >= BLUNDER:
        return "blunder"
    if wp_loss >= MISTAKE:
        return "mistake"
    if wp_loss >= INACCURACY:
        return "inaccuracy"
    if rank == 1:
        return "best"
    if wp_loss < 0.02:
        return "excellent"
    return "good"


def game_accuracy(move_accuracies: list[float], wps: list[float] | None = None) -> float | None:
    """Game accuracy: mean of the per-move accuracies blended with their harmonic mean
    (a simplified version of the Lichess formula, which also volatility-weights)."""
    accs = [a for a in move_accuracies if a is not None]
    if not accs:
        return None
    mean = sum(accs) / len(accs)
    harm = len(accs) / sum(1.0 / max(a, 1.0) for a in accs)
    return (mean + harm) / 2.0


def eval_band(wp: float) -> str:
    if wp >= 0.70:
        return "winning"
    if wp <= 0.30:
        return "losing"
    return "equal"
