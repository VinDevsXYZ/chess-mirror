"""Training, versioning and comparison of Mirror models.

Data hygiene
------------
* Every game has a permanent ``split`` (train/test) assigned at import from its
  hash.  Test games are never used for fitting -- not for the personal model,
  not for the population prior, not for the opening book.
* Each model version stores the exact game ids used for personal training,
  population training and testing, so any two versions can be compared on the
  games held out from *both*.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np

from ..analysis.features import FEATURES, N_FEATURES
from ..config import Config
from ..db import DB, jload
from .dataset import Decision, load_decisions, stack
from .evaluate import evaluate, paired_bootstrap
from .logit import default_prior, fit_logit, group_folds, log_lik
from .predictors import (ENGINE_FEATURES, BookModel, EngineBest, LogitPredictor, MirrorModel, Uniform,
                         mix_book)

SIGMA_GRID = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
ALPHA_GRID = [0.5, 1, 2, 4, 8, 16, 32, 64]
SIGMA_TOLERANCE = 0.002  # nats per move; differences below this are treated as noise

PLAYER_TRAIN = "p.is_player = 1 AND g.split = 'train' AND g.player_color IS NOT NULL"
PLAYER_TEST = "p.is_player = 1 AND g.split = 'test' AND g.player_color IS NOT NULL"
POPULATION_TRAIN = "p.is_player = 0 AND g.split = 'train' AND g.opponent_is_engine = 0"


def _mask_sigma(features: list[str], on: float = 3.0, off: float = 1e-4) -> np.ndarray:
    return np.array([on if f in features else off for f in FEATURES])


def _cv_sigma(dec: list[Decision], mu: np.ndarray, grid: list[float], k: int, progress) -> tuple[float, dict, list]:
    gids = np.array([d.game_id for d in dec])
    folds = group_folds(gids, k)
    scores, oofs = {}, {}
    for s in grid:
        total = 0.0
        oof = [None] * len(dec)
        for fm in folds:
            tr = [d for d, m in zip(dec, fm) if not m]
            te_idx = np.where(fm)[0]
            beta, _ = fit_logit(stack(tr), mu=mu, sigma=s)
            te = [dec[i] for i in te_idx]
            total += log_lik(beta, stack(te))
            for i in te_idx:
                oof[i] = beta
        scores[s] = total / len(dec)
        oofs[s] = oof
    # smallest sigma (least personalisation) whose held-out fit is within noise of the best
    best_ll = max(scores.values())
    best_sigma = min(s for s in grid if scores[s] >= best_ll - SIGMA_TOLERANCE)
    oof_best = oofs[best_sigma]
    progress("  sigma CV (mean held-out log-lik per move): " +
             ", ".join(f"{s}:{v:.3f}" for s, v in scores.items()))
    return best_sigma, scores, oof_best


def _cv_alpha(dec: list[Decision], oof_betas: list, grid: list[float], k: int) -> tuple[float, dict]:
    gids = np.array([d.game_id for d in dec])
    folds = group_folds(gids, k, seed=3)
    from .logit import proba
    scores = {a: 0.0 for a in grid}
    for fm in folds:
        book = BookModel.build([d for d, m in zip(dec, fm) if not m])
        for i in np.where(fm)[0]:
            d = dec[i]
            pm = proba(oof_betas[i], d.X)
            counts = book.lookup(d.key)
            for a in grid:
                scores[a] += np.log(max(mix_book(pm, d.ucis, counts, a)[d.y], 1e-12))
    best = max(scores, key=scores.get)
    return best, {a: v / len(dec) for a, v in scores.items()}


def next_version(db: DB) -> str:
    n = db.scalar("SELECT COUNT(*) FROM models") or 0
    return f"mirror-v{n + 1}"


def train(db: DB, cfg: Config, notes: str = "", activate: bool = True, folds: int = 5,
          progress=print) -> dict:
    progress("Loading decisions...")
    w = cfg.analysis.workers
    pop = load_decisions(db, POPULATION_TRAIN, workers=w, progress=progress)
    mine = load_decisions(db, PLAYER_TRAIN, workers=w, progress=progress)
    test = load_decisions(db, PLAYER_TEST, workers=w, progress=progress)
    progress(f"  population (opponents, train games): {len(pop)} moves; "
             f"player train: {len(mine)} moves; player test: {len(test)} moves")
    if not mine:
        raise RuntimeError("No analysed player moves in training games. Import and analyse games first.")

    pop_elo = db.scalar("SELECT AVG(opponent_elo) FROM games WHERE split='train' AND opponent_elo IS NOT NULL")
    player_elo = db.scalar("SELECT AVG(player_elo) FROM games WHERE split='train' AND player_elo IS NOT NULL")
    prior0 = default_prior(player_elo or pop_elo or cfg.player.rating or 1500)

    # 1) population prior (Elo-matched humans = the player's opponents)
    if len(pop) >= 200:
        S_pop = stack(pop)
        b_pop, se_pop = fit_logit(S_pop, mu=prior0, sigma=2.0)
        b_pop_q, _ = fit_logit(S_pop, mu=np.zeros(N_FEATURES), sigma=_mask_sigma(ENGINE_FEATURES))
        progress(f"  population model fitted on {len(pop)} opponent moves")
    else:
        b_pop, se_pop = prior0, np.full(N_FEATURES, np.nan)
        b_pop_q = prior0
        progress("  too little population data: using the uninformed Elo prior")
    population = LogitPredictor(b_pop, se_pop, "population (Elo-matched opponents)", n_train=len(pop))
    pop_strength = LogitPredictor(b_pop_q, None, "generic Elo-matched strength only")

    # 2) personal model: shrink toward the population, sigma by grouped CV
    n_games = len({d.game_id for d in mine})
    if n_games >= max(5, folds) and len(mine) >= 150:
        progress("Cross-validating personalization strength...")
        sigma, sigma_scores, oof = _cv_sigma(mine, b_pop, SIGMA_GRID, folds, progress)
        alpha, alpha_scores = _cv_alpha(mine, oof, ALPHA_GRID, folds)
        progress(f"  chosen sigma={sigma}, book alpha={alpha}")
    else:
        sigma, sigma_scores, alpha, alpha_scores = 0.05, {}, 4.0, {}
        progress("  few games: personal model stays close to the population prior (sigma=0.05)")
    S_me = stack(mine)
    b_me, se_me = fit_logit(S_me, mu=b_pop, sigma=sigma, beta0=b_pop)
    personal = LogitPredictor(b_me, se_me, "personal logit (no book)", sigma=sigma, n_train=len(mine))
    b_me_q, _ = fit_logit(S_me, mu=np.zeros(N_FEATURES), sigma=_mask_sigma(ENGINE_FEATURES))
    my_strength = LogitPredictor(b_me_q, None, "my strength only (no style)")
    book = BookModel.build(mine)

    version = next_version(db)
    sigma_note = (f"personalization sigma={sigma} chosen by {folds}-fold CV over {n_games} training games"
                  if sigma_scores else "insufficient data for CV; staying near population prior")
    meta = {
        "sigma": sigma, "sigma_scores": {str(k): v for k, v in sigma_scores.items()},
        "alpha": alpha, "alpha_scores": {str(k): v for k, v in alpha_scores.items()},
        "n_player_train": len(mine), "n_population": len(pop), "n_test": len(test),
        "n_train_games": n_games, "population_mean_elo": pop_elo, "player_mean_elo": player_elo,
        "sigma_note": sigma_note,
    }
    mirror = MirrorModel(personal, population, book, alpha, meta, version)

    # 3) held-out evaluation vs baselines
    results = {}
    lls = {}
    if test:
        progress(f"Evaluating on {len(test)} held-out moves from "
                 f"{len({d.game_id for d in test})} test games...")
        for m in [Uniform(), EngineBest(), pop_strength, population, my_strength, personal, mirror]:
            r = evaluate(m, test)
            lls[m.name] = r.pop("_per_decision_ll")
            results[m.name] = r
            progress(f"  {m.name:40s} top1={r['top1']:.3f} top3={r['top3']:.3f} "
                     f"logloss={r['log_loss']:.3f}")
        gids = [d.game_id for d in test]
        results["_significance"] = {
            "mirror_vs_population": paired_bootstrap(lls[mirror.name], lls[population.name], gids),
            "mirror_vs_generic_strength": paired_bootstrap(lls[mirror.name], lls[pop_strength.name], gids),
            "mirror_vs_my_strength": paired_bootstrap(lls[mirror.name], lls[my_strength.name], gids),
        }

    path = cfg.models_dir / f"{version}.json"
    mirror.save(path)
    ids = lambda ds: sorted({d.game_id for d in ds})  # noqa: E731
    with db.tx() as conn:
        if activate:
            conn.execute("UPDATE models SET is_active = 0")
        conn.execute(
            """INSERT INTO models(version, kind, created_at, params, artifact_path, feature_version,
               train_game_ids, test_game_ids, population_game_ids, metrics, notes, is_active)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (version, mirror.kind, dt.datetime.now().isoformat(timespec="seconds"),
             json.dumps({"sigma": sigma, "alpha": alpha, "folds": folds, "features": FEATURES}), str(path),
             "fv1", json.dumps(ids(mine)), json.dumps(ids(test)), json.dumps(ids(pop)), json.dumps(results),
             notes, int(activate)))
    progress(f"Saved {version} -> {path}{' (active)' if activate else ''}")
    return {"version": version, "metrics": results, "meta": meta}


def active_version(db: DB) -> dict | None:
    row = db.one("SELECT * FROM models WHERE is_active = 1 ORDER BY id DESC LIMIT 1")
    return dict(row) if row else None


def load_active(db: DB, cfg: Config, version: str | None = None) -> MirrorModel | None:
    row = (db.one("SELECT * FROM models WHERE version = ?", (version,)) if version
           else db.one("SELECT * FROM models WHERE is_active = 1 ORDER BY id DESC LIMIT 1"))
    if not row:
        return None
    return MirrorModel.load(row["artifact_path"])


def compare(db: DB, cfg: Config, va: str, vb: str, progress=print) -> dict:
    """Compare two versions on test games held out from BOTH (no leakage either way)."""
    ra, rb = (db.one("SELECT * FROM models WHERE version = ?", (v,)) for v in (va, vb))
    if not ra or not rb:
        raise ValueError("unknown version")
    train_any = set(jload(ra["train_game_ids"], [])) | set(jload(rb["train_game_ids"], [])) \
        | set(jload(ra["population_game_ids"], [])) | set(jload(rb["population_game_ids"], []))
    common = (set(jload(ra["test_game_ids"], [])) & set(jload(rb["test_game_ids"], []))) - train_any
    if not common:
        return {"error": "no common held-out games"}
    ids = sorted(common)
    test = load_decisions(db, f"p.is_player = 1 AND p.game_id IN ({','.join('?' * len(ids))})", tuple(ids),
                          workers=cfg.analysis.workers)
    ma, mb = MirrorModel.load(ra["artifact_path"]), MirrorModel.load(rb["artifact_path"])
    ea, eb = evaluate(ma, test), evaluate(mb, test)
    sig = paired_bootstrap(ea.pop("_per_decision_ll"), eb.pop("_per_decision_ll"), [d.game_id for d in test])
    return {"games": len(ids), "moves": len(test), va: ea, vb: eb, "logloss_diff_a_minus_b": sig}
