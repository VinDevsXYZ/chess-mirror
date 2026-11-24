"""Position features (for the profile / coach) and candidate-move features (for the move model).

Leakage rule
------------
Candidate features are a pure function of (board, engine analysis of the
*position*, context).  They must never depend on which move was actually played
-- e.g. the extra deep evaluation of the played move computed for statistics is
NOT a model input.  ``candidate_features`` therefore only reads ``top`` (deep
MultiPV) and ``broad`` (shallow eval of every legal move).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np

from .motifs import VAL, allows_after, is_hanging, pins_and_skewers, see, targets_of
from .winprob import cp_to_wp

FEATURE_VERSION = "fv1"

NPM = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
CENTER = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])


# ------------------------------------------------------------------ helpers
def non_pawn_material(board: chess.Board) -> int:
    return sum(NPM[pt] * len(board.pieces(pt, c)) for pt in NPM for c in (chess.WHITE, chess.BLACK))


def phase_of(board: chess.Board) -> str:
    npm = non_pawn_material(board)
    if npm <= 26:
        return "endgame"
    if board.fullmove_number <= 12 and npm >= 50:
        return "opening"
    return "middlegame"


def material_balance(board: chess.Board, color: chess.Color) -> int:
    vals = {chess.PAWN: 1, **NPM}
    return sum(v * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color))) for pt, v in vals.items())


def king_zone(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    k = board.king(color)
    if k is None:
        return chess.SquareSet()
    return chess.SquareSet(chess.BB_KING_ATTACKS[k] | chess.BB_SQUARES[k])


def king_zone_pressure(board: chess.Board, attacker: chess.Color) -> int:
    """Number of (attacker piece, king-zone square) attack pairs against the other king."""
    zone = king_zone(board, not attacker)
    total = 0
    for sq in chess.SquareSet(board.occupied_co[attacker]):
        if board.piece_type_at(sq) in (chess.PAWN, chess.KING):
            continue
        total += len(board.attacks(sq) & zone)
    return total


def pawn_shield(board: chess.Board, color: chess.Color) -> int:
    k = board.king(color)
    if k is None:
        return 0
    f, r = chess.square_file(k), chess.square_rank(k)
    step = 1 if color == chess.WHITE else -1
    n = 0
    for df in (-1, 0, 1):
        for dr in (1, 2):
            ff, rr = f + df, r + step * dr
            if 0 <= ff < 8 and 0 <= rr < 8 and board.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, color):
                n += 1
                break
    return n


def pawn_structure(board: chess.Board, color: chess.Color) -> dict:
    pawns = board.pieces(chess.PAWN, color)
    opp = board.pieces(chess.PAWN, not color)
    files = [chess.square_file(s) for s in pawns]
    counts = [files.count(f) for f in range(8)]
    doubled = sum(c - 1 for c in counts if c > 1)
    isolated = sum(counts[f] for f in range(8)
                   if counts[f] and (f == 0 or not counts[f - 1]) and (f == 7 or not counts[f + 1]))
    islands, inside = 0, False
    for c in counts:
        if c and not inside:
            islands += 1
        inside = bool(c)
    passed = 0
    backward = 0
    step = 1 if color == chess.WHITE else -1
    for s in pawns:
        f, r = chess.square_file(s), chess.square_rank(s)
        ahead = [o for o in opp if abs(chess.square_file(o) - f) <= 1
                 and (chess.square_rank(o) - r) * step > 0]
        if not ahead:
            passed += 1
        # backward: no friendly pawn on adjacent files level or behind, and stop square attacked by enemy pawn
        support = [p for p in pawns if abs(chess.square_file(p) - f) == 1 and (r - chess.square_rank(p)) * step >= 0]
        stop_r = r + step
        if not support and 0 <= stop_r < 8:
            stop = chess.square(f, stop_r)
            if board.attackers(not color, stop) & opp:
                backward += 1
    return {"doubled": doubled, "isolated": isolated, "islands": islands, "passed": passed, "backward": backward}


def open_files(board: chess.Board) -> tuple[int, dict]:
    wp, bp = board.pieces(chess.PAWN, chess.WHITE), board.pieces(chess.PAWN, chess.BLACK)
    n_open = 0
    semi = {chess.WHITE: 0, chess.BLACK: 0}
    for f in range(8):
        fm = chess.BB_FILES[f]
        w, b = bool(wp & fm), bool(bp & fm)
        if not w and not b:
            n_open += 1
        elif not w:
            semi[chess.WHITE] += 1
        elif not b:
            semi[chess.BLACK] += 1
    return n_open, semi


def rooks_on_open(board: chess.Board, color: chess.Color) -> int:
    n = 0
    own = board.pieces(chess.PAWN, color)
    for s in board.pieces(chess.ROOK, color):
        if not (own & chess.BB_FILES[chess.square_file(s)]):
            n += 1
    return n


def outposts(board: chess.Board, color: chess.Color) -> int:
    n = 0
    step = 1 if color == chess.WHITE else -1
    opp_pawns = board.pieces(chess.PAWN, not color)
    for s in board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color):
        r = chess.square_rank(s)
        if (r >= 4) if color == chess.WHITE else (r <= 3):
            protected = bool(board.attackers(color, s) & board.pieces(chess.PAWN, color))
            f = chess.square_file(s)
            attackable = any(abs(chess.square_file(o) - f) == 1 and (chess.square_rank(o) - r) * step > 0
                             for o in opp_pawns)
            if protected and not attackable:
                n += 1
    return n


def bad_bishops(board: chess.Board, color: chess.Color) -> int:
    pawns = board.pieces(chess.PAWN, color)
    n = 0
    for s in board.pieces(chess.BISHOP, color):
        light = (chess.square_file(s) + chess.square_rank(s)) % 2
        same = sum(1 for p in pawns if (chess.square_file(p) + chess.square_rank(p)) % 2 == light)
        if pawns and same / len(pawns) >= 0.6 and same >= 4:
            n += 1
    return n


def space(board: chess.Board, color: chess.Color) -> int:
    ranks = (3, 4, 5) if color == chess.WHITE else (2, 3, 4)
    n = 0
    for s in chess.SquareSet(chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F):
        if chess.square_rank(s) in ranks and board.attackers(color, s) & board.pieces(chess.PAWN, color):
            n += 1
    return n


def mobility(board: chess.Board, color: chess.Color) -> int:
    if board.turn == color:
        return board.legal_moves.count()
    if board.is_check():
        return 0
    b = board.copy(stack=False)
    b.push(chess.Move.null())
    return b.legal_moves.count()


def position_features(board: chess.Board) -> dict:
    me, op = board.turn, not board.turn
    n_open, semi = open_files(board)
    ps_me, ps_op = pawn_structure(board, me), pawn_structure(board, op)
    return {
        "phase": phase_of(board),
        "npm": non_pawn_material(board),
        "material": material_balance(board, me),
        "in_check": board.is_check(),
        "n_legal": board.legal_moves.count(),
        "mobility_me": mobility(board, me),
        "mobility_op": mobility(board, op),
        "shield_me": pawn_shield(board, me),
        "shield_op": pawn_shield(board, op),
        "kz_pressure_on_me": king_zone_pressure(board, op),
        "kz_pressure_on_op": king_zone_pressure(board, me),
        "pawns_me": ps_me,
        "pawns_op": ps_op,
        "open_files": n_open,
        "semi_open_me": semi[me],
        "rooks_open_me": rooks_on_open(board, me),
        "bishop_pair_me": len(board.pieces(chess.BISHOP, me)) >= 2,
        "bishop_pair_op": len(board.pieces(chess.BISHOP, op)) >= 2,
        "outposts_me": outposts(board, me),
        "outposts_op": outposts(board, op),
        "bad_bishops_me": bad_bishops(board, me),
        "space_me": space(board, me),
        "space_op": space(board, op),
        "hanging_me": sum(1 for s in chess.SquareSet(board.occupied_co[me]) if is_hanging(board, s)),
        "hanging_op": sum(1 for s in chess.SquareSet(board.occupied_co[op]) if is_hanging(board, s)),
        "queens_on": bool(board.pieces(chess.QUEEN, me) or board.pieces(chess.QUEEN, op)),
    }


# ------------------------------------------------------------ candidate features
FEATURES = [
    # engine quality
    "loss", "loss_sqrt", "is_best", "top3", "in_deep", "shallow_loss", "hidden", "trap", "shallow_best",
    # quality x context
    "loss_x_tp", "loss_x_sharp", "loss_x_winning", "loss_x_losing", "loss_x_endgame", "loss_x_opening",
    # move type
    "capture", "cap_value", "recapture", "check", "castle", "promotion",
    "knight", "bishop", "rook", "queen", "king",
    "pawn_push", "pawn_storm", "retreat", "advance", "kz_attack", "develop", "early_queen", "king_walk",
    "center", "trade", "queen_trade", "trade_x_winning", "trade_x_losing", "sac", "sac_x_losing",
    "hangs", "escape", "threat", "fork", "pin", "skewer", "discovered",
    "allows_fork", "allows_mate", "passed_create", "self_weak", "shield_weaken", "check_x_tp",
]
F_INDEX = {n: i for i, n in enumerate(FEATURES)}
N_FEATURES = len(FEATURES)


@dataclass
class Context:
    """Non-board context for a decision (clock etc.)."""
    clock: float | None = None       # mover's remaining seconds before the move
    base: float | None = None        # initial clock
    prev_move: str | None = None     # opponent's last move (uci)

    @property
    def time_pressure(self) -> bool:
        if self.clock is None:
            return False
        limit = 30.0
        if self.base:
            limit = max(10.0, min(60.0, 0.1 * self.base))
        return self.clock < limit


def engine_losses(board: chess.Board, top: list[dict], broad: dict[str, int]) -> dict[str, dict]:
    """Per-legal-move engine quality info derived from the deep and broad passes."""
    legal = [m.uci() for m in board.legal_moves]
    deep = {t["uci"]: t for t in top}
    deep_wps = {u: cp_to_wp(t.get("cp"), t.get("mate")) for u, t in deep.items()}
    best_wp = max(deep_wps.values()) if deep_wps else None
    b_wps = {u: cp_to_wp(cp) for u, cp in broad.items()}
    b_best = max(b_wps.values()) if b_wps else 0.5
    worst_deep_loss = (best_wp - min(deep_wps.values())) if deep_wps else 0.0
    ranked = sorted(deep_wps, key=lambda u: -deep_wps[u])
    out = {}
    for u in legal:
        s_loss = max(0.0, b_best - b_wps[u]) if u in b_wps else 0.5
        if u in deep_wps:
            loss = max(0.0, best_wp - deep_wps[u])
            rank = ranked.index(u) + 1
        else:
            loss = max(worst_deep_loss, s_loss)
            rank = None
        out[u] = {"loss": min(1.0, loss), "shallow_loss": min(1.0, s_loss), "rank": rank,
                  "shallow_best": u in b_wps and b_wps[u] >= b_best - 1e-9}
    return out


def _structure_damage(before: dict, after: dict) -> int:
    return max(0, (after["doubled"] + after["isolated"] + after["backward"])
               - (before["doubled"] + before["isolated"] + before["backward"]))


def candidate_features(board: chess.Board, top: list[dict], broad: dict[str, int],
                       ctx: Context | None = None, wp_before: float | None = None
                       ) -> tuple[list[str], np.ndarray]:
    """Feature matrix (n_legal x N_FEATURES) for every legal move."""
    ctx = ctx or Context()
    me = board.turn
    moves = list(board.legal_moves)
    q = engine_losses(board, top, broad)
    if wp_before is None:
        wp_before = cp_to_wp(top[0].get("cp"), top[0].get("mate")) if top else 0.5
    phase = phase_of(board)
    winning, losing = wp_before >= 0.70, wp_before <= 0.30
    tp = ctx.time_pressure
    sharp = sum(1 for v in q.values() if v["loss"] >= 0.10) / max(1, len(q))
    opp_king = board.king(not me)
    my_king = board.king(me)
    kz_before = king_zone_pressure(board, me)
    ps_before = pawn_structure(board, me)
    passed_before = ps_before["passed"]
    hanging_before = {s for s in chess.SquareSet(board.occupied_co[me]) if is_hanging(board, s)}
    prev_to = None
    if ctx.prev_move:
        try:
            prev_to = chess.Move.from_uci(ctx.prev_move).to_square
        except ValueError:
            prev_to = None
    castled_side = None
    if my_king is not None and chess.square_file(my_king) in (6, 7, 1, 2):
        castled_side = "k" if chess.square_file(my_king) >= 6 else "q"

    X = np.zeros((len(moves), N_FEATURES), dtype=np.float32)
    ucis = []
    for i, m in enumerate(moves):
        u = m.uci()
        ucis.append(u)
        f = X[i]
        e = q[u]
        loss = e["loss"]
        f[F_INDEX["loss"]] = loss
        f[F_INDEX["loss_sqrt"]] = math.sqrt(loss)
        f[F_INDEX["is_best"]] = e["rank"] == 1
        f[F_INDEX["top3"]] = e["rank"] is not None and e["rank"] <= 3
        f[F_INDEX["in_deep"]] = e["rank"] is not None
        f[F_INDEX["shallow_loss"]] = e["shallow_loss"]
        f[F_INDEX["hidden"]] = max(0.0, e["shallow_loss"] - loss)
        f[F_INDEX["trap"]] = max(0.0, loss - e["shallow_loss"])
        f[F_INDEX["shallow_best"]] = e["shallow_best"]
        f[F_INDEX["loss_x_tp"]] = loss * tp
        f[F_INDEX["loss_x_sharp"]] = loss * sharp
        f[F_INDEX["loss_x_winning"]] = loss * winning
        f[F_INDEX["loss_x_losing"]] = loss * losing
        f[F_INDEX["loss_x_endgame"]] = loss * (phase == "endgame")
        f[F_INDEX["loss_x_opening"]] = loss * (phase == "opening")

        pt = board.piece_type_at(m.from_square)
        captured = chess.PAWN if board.is_en_passant(m) else board.piece_type_at(m.to_square)
        is_cap = captured is not None
        gain = see(board, m)
        gives_check = board.gives_check(m)
        f[F_INDEX["capture"]] = is_cap
        f[F_INDEX["cap_value"]] = (VAL[captured] / 900.0) if is_cap else 0.0
        f[F_INDEX["recapture"]] = is_cap and prev_to is not None and m.to_square == prev_to
        f[F_INDEX["check"]] = gives_check
        f[F_INDEX["check_x_tp"]] = gives_check and tp
        f[F_INDEX["castle"]] = board.is_castling(m)
        f[F_INDEX["promotion"]] = m.promotion == chess.QUEEN
        for name, t in (("knight", chess.KNIGHT), ("bishop", chess.BISHOP), ("rook", chess.ROOK),
                        ("queen", chess.QUEEN), ("king", chess.KING)):
            f[F_INDEX[name]] = pt == t
        fr, tr = chess.square_rank(m.from_square), chess.square_rank(m.to_square)
        forward = (tr - fr) if me == chess.WHITE else (fr - tr)
        if pt == chess.PAWN and not is_cap:
            f[F_INDEX["pawn_push"]] = 1
            if opp_king is not None and phase != "endgame" and \
                    abs(chess.square_file(m.to_square) - chess.square_file(opp_king)) <= 2 and \
                    (my_king is None or abs(chess.square_file(my_king) - chess.square_file(opp_king)) >= 3
                     or chess.square_file(m.to_square) not in (5, 6, 7, 0, 1, 2)):
                f[F_INDEX["pawn_storm"]] = 1
            if castled_side and my_king is not None and \
                    abs(chess.square_file(m.from_square) - chess.square_file(my_king)) <= 1 and \
                    abs(fr - chess.square_rank(my_king)) <= 2:
                f[F_INDEX["shield_weaken"]] = 1
        if pt not in (chess.PAWN, chess.KING):
            f[F_INDEX["retreat"]] = forward < 0
            f[F_INDEX["advance"]] = forward > 0
        if pt in (chess.KNIGHT, chess.BISHOP) and phase == "opening":
            home = 0 if me == chess.WHITE else 7
            f[F_INDEX["develop"]] = fr == home and tr != home
        f[F_INDEX["early_queen"]] = pt == chess.QUEEN and board.fullmove_number <= 8
        f[F_INDEX["king_walk"]] = pt == chess.KING and not board.is_castling(m) and phase != "endgame"
        f[F_INDEX["center"]] = m.to_square in CENTER

        after = board.copy(stack=False)
        after.push(m)
        f[F_INDEX["kz_attack"]] = (king_zone_pressure(after, me) - kz_before) / 4.0
        # trades: capture of equal value that can be recaptured, or offering an even exchange
        opp_attacks_to = after.attackers(not me, m.to_square)
        if is_cap and captured != chess.PAWN and abs(VAL[captured] - VAL[pt]) <= 60 and opp_attacks_to:
            f[F_INDEX["trade"]] = 1
        elif not is_cap and pt not in (chess.PAWN, chess.KING) and opp_attacks_to and gain == 0:
            if any(after.piece_type_at(a) is not None and abs(VAL[after.piece_type_at(a)] - VAL[pt]) <= 60
                   for a in opp_attacks_to):
                f[F_INDEX["trade"]] = 1
        if pt == chess.QUEEN and (captured == chess.QUEEN or (
                not is_cap and any(after.piece_type_at(a) == chess.QUEEN for a in opp_attacks_to)
                and after.attackers(me, m.to_square))):
            f[F_INDEX["queen_trade"]] = 1
        f[F_INDEX["trade_x_winning"]] = f[F_INDEX["trade"]] * winning
        f[F_INDEX["trade_x_losing"]] = f[F_INDEX["trade"]] * losing
        sac = gain <= -150
        f[F_INDEX["sac"]] = sac
        f[F_INDEX["sac_x_losing"]] = sac and losing
        # what hangs after the move (opponent's best capture) and what we threaten
        opp_best = 0
        if not after.is_game_over():
            for c in after.generate_legal_captures():
                g = see(after, c)
                if g > opp_best:
                    opp_best = g
        f[F_INDEX["hangs"]] = min(opp_best, 900) / 900.0
        f[F_INDEX["escape"]] = m.from_square in hanging_before and not is_hanging(after, m.to_square)
        tg = targets_of(after, m.to_square)
        f[F_INDEX["threat"]] = min(len(tg), 3) / 3.0
        f[F_INDEX["fork"]] = len(tg) >= 2 and (not is_hanging(after, m.to_square) or after.is_check())
        pin, skewer = pins_and_skewers(after, m.to_square)
        f[F_INDEX["pin"]] = pin
        f[F_INDEX["skewer"]] = skewer
        checkers = after.checkers()
        f[F_INDEX["discovered"]] = bool(checkers) and any(c != m.to_square for c in checkers)
        al = allows_after(after) if not after.is_game_over() else set()
        f[F_INDEX["allows_fork"]] = bool(al & {"knight_fork", "pawn_fork"})
        f[F_INDEX["allows_mate"]] = "mate_in_1" in al
        if pt == chess.PAWN:
            ps_after = pawn_structure(after, me)
            f[F_INDEX["passed_create"]] = ps_after["passed"] > passed_before
            f[F_INDEX["self_weak"]] = _structure_damage(ps_before, ps_after) > 0
    return ucis, X
