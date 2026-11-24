"""Unit tests that do not need Stockfish."""

import chess
import numpy as np
import pytest

from chessmirror.analysis.features import FEATURES, N_FEATURES, candidate_features
from chessmirror.analysis.motifs import see, static_motifs, threats_against
from chessmirror.analysis.winprob import classify, cp_to_wp, move_accuracy
from chessmirror.model.dataset import Decision, stack
from chessmirror.model.logit import fit_logit, proba
from chessmirror.model.predictors import mix_book
from chessmirror.model.train import _mask_sigma
from chessmirror.pgn_import import parse_time_control, split_for_hash
from chessmirror.rating.bayes import RatingFilter, expected


def test_winprob_and_accuracy():
    assert cp_to_wp(0) == pytest.approx(0.5)
    assert cp_to_wp(None, mate=3) == 1.0 and cp_to_wp(None, mate=-2) == 0.0
    assert cp_to_wp(300) > 0.7
    assert move_accuracy(0.6, 0.6) == pytest.approx(100, abs=0.1)
    assert classify(0.2, None, 30) == "blunder"
    assert classify(0.0, 1, 30) == "best"
    assert classify(0.5, None, 1) == "forced"


def test_see_and_motifs():
    b = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    assert see(b, chess.Move.from_uci("d1d5")) == 900
    assert "knight_fork" in static_motifs(chess.Board("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1"), chess.Move.from_uci("b5c7"))
    assert "pin" in static_motifs(chess.Board("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1"), chess.Move.from_uci("f1b5"))
    assert "skewer" in static_motifs(chess.Board("q7/8/8/k7/8/8/8/6KR w - - 0 1"), chess.Move.from_uci("h1a1"))
    assert {"mate", "back_rank"} <= static_motifs(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"),
                                                  chess.Move.from_uci("a1a8"))
    assert "knight_fork" in threats_against(chess.Board("4k3/8/8/8/1n6/8/8/R3K3 w - - 0 1"))
    assert "mate_threat" in threats_against(chess.Board("r5k1/8/8/8/8/8/5PPP/6K1 w - - 0 1"))


def test_candidate_features_do_not_depend_on_played_move():
    """Leakage guard: features are a function of the position analysis only."""
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    moves = [m.uci() for m in b.legal_moves]
    top = [{"uci": u, "cp": 40 - 10 * i, "mate": None} for i, u in enumerate(moves[:5])]
    broad = {u: 30 - i for i, u in enumerate(moves)}
    u1, X1 = candidate_features(b, top, broad)
    u2, X2 = candidate_features(b, top, broad)
    assert u1 == u2 and np.array_equal(X1, X2)
    assert X1.shape == (len(moves), N_FEATURES)


def _synthetic_decisions(beta, n=1500, k=12, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        X = rng.normal(size=(k, N_FEATURES)).astype(np.float32) * 0.5
        p = proba(beta, X)
        y = int(rng.choice(k, p=p))
        out.append(Decision(fen=chess.STARTING_FEN, ucis=[str(j) for j in range(k)], X=X, top=[], broad={},
                            ctx=None, y=y, game_id=i // 40))
    return out


def test_logit_recovers_coefficients():
    true = np.zeros(N_FEATURES)
    true[FEATURES.index("capture")] = 1.5
    true[FEATURES.index("check")] = -1.0
    true[FEATURES.index("loss_sqrt")] = -2.0
    S = stack(_synthetic_decisions(true))
    beta, se = fit_logit(S, sigma=10.0)
    for name in ("capture", "check", "loss_sqrt"):
        j = FEATURES.index(name)
        assert abs(beta[j] - true[j]) < 4 * se[j] + 0.05


def test_prior_shrinkage_with_little_data():
    """With few decisions and a tight prior the personal model stays at the population prior."""
    pop = np.zeros(N_FEATURES)
    pop[FEATURES.index("capture")] = 0.8
    true = pop.copy()
    true[FEATURES.index("capture")] = 2.0
    few = stack(_synthetic_decisions(true, n=20))
    b_tight, _ = fit_logit(few, mu=pop, sigma=0.05)
    b_loose, _ = fit_logit(stack(_synthetic_decisions(true, n=2000)), mu=pop, sigma=3.0)
    j = FEATURES.index("capture")
    assert abs(b_tight[j] - 0.8) < 0.1
    assert abs(b_loose[j] - 2.0) < 0.3


def test_mask_sigma_excludes_style():
    s = _mask_sigma(["loss"])
    assert s[FEATURES.index("loss")] == 3.0 and s[FEATURES.index("capture")] < 1e-3


def test_book_mixing():
    p = np.array([0.7, 0.2, 0.1])
    mixed = mix_book(p, ["a", "b", "c"], {"b": 9}, alpha=1.0)
    assert mixed.argmax() == 1 and mixed.sum() == pytest.approx(1.0)
    assert np.allclose(mix_book(p, ["a", "b", "c"], {}, 4.0), p)


def test_rating_filter_recovers_rating():
    rng = np.random.default_rng(1)
    true = 1650
    f = RatingFilter(prior_mean=1500, prior_sd=400, drift_per_day=0)
    for _ in range(300):
        opp = rng.normal(1600, 150)
        f.update(opp, float(rng.random() < expected(true, opp)))
    s = f.summary()
    assert s.lo < true < s.hi and s.hi - s.lo < 200


def test_time_control_and_split():
    assert parse_time_control("180+2") == (180, 2, "blitz")
    assert parse_time_control("60") == (60, 0, "bullet")
    assert parse_time_control("900+10")[2] == "rapid"
    assert parse_time_control("1/259200")[2] == "correspondence"
    assert split_for_hash("0" * 40) == "test"
    assert split_for_hash("f" * 40) == "train"


def test_adaptive_elotest_converges(tmp_path, monkeypatch):
    """Simulated player (true 1700 on the Stockfish scale) vs adaptively chosen opponents."""
    monkeypatch.setenv("MIRROR_DATA_DIR", str(tmp_path))
    from chessmirror.config import load_config
    from chessmirror.db import open_db
    from chessmirror.rating.elotest import current_opponent, prepare_next
    from chessmirror.rating.strength import elotest_estimate
    cfg = load_config()
    db = open_db(cfg)
    rng = np.random.default_rng(4)
    true, opps = 1700, []
    for i in range(40):
        prepare_next(db, cfg)
        opp = current_opponent(cfg)
        opps.append(opp)
        score = float(rng.random() < expected(true, opp))
        db.conn.execute("INSERT INTO elo_games(played_at, scale, opponent_rating, result, status) VALUES (?,?,?,?,?)",
                        (f"2026-09-01T{i // 60:02d}:{i % 60:02d}:00", "stockfish-uci", opp, score, "final"))
        db.conn.commit()
    est = elotest_estimate(db)["current"]
    assert est["lo"] < true < est["hi"]
    assert est["hi"] - est["lo"] < 450
    assert abs(np.mean(opps[-15:]) - true) < 250  # games concentrate near the player's level
