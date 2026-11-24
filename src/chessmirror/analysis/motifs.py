"""Tactical motif detection.

Two families of detectors:

* **static** detectors look at one move on one board (fork, pin, skewer,
  discovered attack, hanging-piece capture, sacrifice, mate-in-1 ...). They are
  cheap enough to run for every legal move, so they double as model features.
* **line** detectors look at an engine principal variation to recognise
  multi-move ideas (removal of the defender, overloading, deflection,
  zwischenzug, mating attacks, material-winning combinations).

All detectors are heuristics. They are tagged with the motif name only when the
pattern is present on the board; they never infer intent.
"""

from __future__ import annotations

import chess

VAL = {chess.PAWN: 100, chess.KNIGHT: 300, chess.BISHOP: 310, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}
SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)
PIECE_NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
              chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}

MOTIFS = [
    "fork", "knight_fork", "pawn_fork", "double_attack", "pin", "skewer", "discovered_attack",
    "discovered_check", "double_check", "capture_hanging", "removal_of_defender", "overload",
    "deflection", "zwischenzug", "sacrifice", "mate", "mating_attack", "back_rank", "promotion",
    "trapped_piece_escape", "defensive_resource",
]


# --------------------------------------------------------------------------- SEE
def see(board: chess.Board, move: chess.Move) -> int:
    """Static exchange evaluation of ``move`` for the side making it (centipawns)."""
    to = move.to_square
    if board.is_en_passant(move):
        captured = chess.PAWN
    else:
        captured = board.piece_type_at(to)
    mover = board.piece_type_at(move.from_square)
    if mover is None:
        return 0
    gains = [VAL[captured] if captured else 0]
    on_square = VAL[move.promotion] if move.promotion else VAL[mover]
    if move.promotion:
        gains[0] += VAL[move.promotion] - VAL[chess.PAWN]
    b = board.copy(stack=False)
    b.push(move)
    side = b.turn
    while True:
        attackers = b.attackers(side, to)
        if not attackers:
            break
        sq = min(attackers, key=lambda s: VAL[b.piece_type_at(s)])
        pt = b.piece_type_at(sq)
        if pt == chess.KING and b.attackers(not side, to):
            break  # king cannot recapture into a defended square
        gains.append(on_square - gains[-1])
        on_square = VAL[pt]
        b.remove_piece_at(sq)
        b.set_piece_at(to, chess.Piece(pt, side))
        side = not side
    for i in range(len(gains) - 1, 0, -1):
        gains[i - 1] = -max(-gains[i - 1], gains[i])
    return gains[0]


def best_capture_gain(board: chess.Board, color: chess.Color) -> tuple[int, chess.Move | None]:
    """Best SEE among ``color``'s captures, as if it were ``color``'s turn."""
    b = board
    if board.turn != color:
        if board.is_check():
            return 0, None
        b = board.copy(stack=False)
        b.push(chess.Move.null())
    best, best_move = 0, None
    for m in b.generate_legal_captures():
        g = see(b, m)
        if g > best:
            best, best_move = g, m
    return best, best_move


def is_hanging(board: chess.Board, sq: chess.Square) -> bool:
    """Piece on ``sq`` can be won: attacked and undefended, or attacked by a cheaper piece."""
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type == chess.KING:
        return False
    attackers = board.attackers(not piece.color, sq)
    if not attackers:
        return False
    if not board.attackers(piece.color, sq):
        return True
    return min(VAL[board.piece_type_at(a)] for a in attackers) < VAL[piece.piece_type]


def targets_of(board: chess.Board, sq: chess.Square) -> list[chess.Square]:
    """Enemy targets attacked by the piece on ``sq`` that it can profitably hit."""
    piece = board.piece_at(sq)
    if piece is None:
        return []
    out = []
    for t in board.attacks(sq):
        tp = board.piece_at(t)
        if tp is None or tp.color == piece.color:
            continue
        if tp.piece_type == chess.KING:
            out.append(t)
        elif VAL[tp.piece_type] > VAL[piece.piece_type] or not board.attackers(tp.color, t):
            if tp.piece_type != chess.PAWN or not board.attackers(tp.color, t):
                out.append(t)
    return out


# ----------------------------------------------------------------- static motifs
def _ray_pieces(board: chess.Board, sq: chess.Square, direction: tuple[int, int], limit: int = 2):
    f, r = chess.square_file(sq), chess.square_rank(sq)
    found = []
    while True:
        f += direction[0]
        r += direction[1]
        if not (0 <= f < 8 and 0 <= r < 8):
            break
        s = chess.square(f, r)
        p = board.piece_at(s)
        if p is not None:
            found.append((s, p))
            if len(found) >= limit:
                break
    return found


def _directions(pt: int):
    rook = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    bishop = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return {chess.ROOK: rook, chess.BISHOP: bishop, chess.QUEEN: rook + bishop}.get(pt, [])


def pins_and_skewers(board: chess.Board, sq: chess.Square) -> tuple[bool, bool]:
    """Does the slider on ``sq`` pin or skewer something?"""
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type not in SLIDERS:
        return False, False
    pin = skewer = False
    for d in _directions(piece.piece_type):
        ray = _ray_pieces(board, sq, d, 2)
        if len(ray) < 2:
            continue
        (s1, p1), (s2, p2) = ray
        if p1.color == piece.color or p2.color == piece.color:
            continue
        v1, v2 = VAL[p1.piece_type], VAL[p2.piece_type]
        if p2.piece_type == chess.KING and p1.piece_type != chess.KING:
            pin = True
        elif v2 > v1 and v2 > VAL[piece.piece_type] - 1 and p1.piece_type != chess.KING:
            pin = True  # relative pin to a more valuable piece
        elif (p1.piece_type == chess.KING or v1 > v2) and p1.piece_type in (chess.KING, chess.QUEEN, chess.ROOK):
            if p2.piece_type != chess.PAWN or not board.attackers(p2.color, s2):
                skewer = True
    return pin, skewer


def static_motifs(board: chess.Board, move: chess.Move) -> set[str]:
    """Motifs created by ``move`` (for the side to move) visible on the resulting board."""
    tags: set[str] = set()
    mover_color = board.turn
    piece = board.piece_at(move.from_square)
    if piece is None:
        return tags
    captured_type = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
    gain = see(board, move)

    if captured_type is not None and captured_type != chess.PAWN and gain >= VAL[captured_type] - 20:
        tags.add("capture_hanging")
    elif captured_type == chess.PAWN and gain >= 100 and not board.attackers(not mover_color, move.to_square):
        tags.add("capture_hanging")
    if gain <= -150:
        tags.add("sacrifice")
    if move.promotion:
        tags.add("promotion")

    # attackers of our own pieces before the move (for "escape" and discovered detection)
    before_attacks = {s: board.attacks(s) for s in board.pieces(chess.BISHOP, mover_color)
                      | board.pieces(chess.ROOK, mover_color) | board.pieces(chess.QUEEN, mover_color)
                      if s != move.from_square}

    after = board.copy(stack=False)
    after.push(move)

    if after.is_checkmate():
        tags.add("mate")
        king = after.king(not mover_color)
        if king is not None and chess.square_rank(king) in (0, 7) and piece.piece_type in (chess.ROOK, chess.QUEEN):
            if chess.square_rank(move.to_square) == chess.square_rank(king):
                tags.add("back_rank")

    checkers = after.checkers()
    if len(checkers) >= 2:
        tags.add("double_check")
    if checkers and any(c != move.to_square for c in checkers):
        tags.add("discovered_check")

    to = move.to_square
    moved_is_safe = not is_hanging(after, to) or after.is_check()
    tg = targets_of(after, to)
    if len(tg) >= 2 and moved_is_safe:
        tags.add("fork")
        if piece.piece_type == chess.KNIGHT:
            tags.add("knight_fork")
        elif piece.piece_type == chess.PAWN:
            tags.add("pawn_fork")

    pin, skewer = pins_and_skewers(after, to)
    if pin:
        tags.add("pin")
    if skewer and moved_is_safe:
        tags.add("skewer")

    # discovered attack: another slider of ours now hits a valuable target
    for s, att_before in before_attacks.items():
        if after.piece_at(s) is None:
            continue
        new = after.attacks(s) & ~att_before
        for t in new:
            tp = after.piece_at(t)
            if tp and tp.color != mover_color and (
                tp.piece_type in (chess.QUEEN, chess.ROOK, chess.KING)
                and VAL[tp.piece_type] >= VAL[after.piece_type_at(s)]
                or not after.attackers(tp.color, t)
            ):
                if tp.piece_type != chess.PAWN:
                    tags.add("discovered_attack")
                    break

    # double attack by different pieces: >=2 enemy pieces hanging after the move
    if "fork" not in tags:
        hanging = [s for s in chess.SquareSet(after.occupied_co[not mover_color])
                   if after.piece_type_at(s) != chess.PAWN and is_hanging(after, s)]
        hanging_before = [s for s in chess.SquareSet(board.occupied_co[not mover_color])
                          if board.piece_type_at(s) != chess.PAWN and is_hanging(board, s)]
        if len(hanging) >= 2 and len(hanging) > len(hanging_before):
            tags.add("double_attack")
    return tags


# ----------------------------------------------------------------- line motifs
def line_motifs(board: chess.Board, pv: list[str], mate: int | None = None,
                prev_move: str | None = None) -> set[str]:
    """Motifs of a whole engine line starting with the mover's move.

    Multi-move motifs (removal of the defender, overload, deflection, sacrifice)
    are only tagged when the line actually wins material for the mover within
    six plies, so quiet coincidences are not labelled as combinations.
    """
    tags: set[str] = set()
    if not pv:
        return tags
    b = board.copy(stack=False)
    mover = b.turn
    try:
        moves = [chess.Move.from_uci(u) for u in pv[:8]]
    except ValueError:
        return tags
    if moves[0] not in b.legal_moves:
        return tags
    first = moves[0]
    tags |= static_motifs(b, first)
    if mate is not None and mate > 0:
        tags.add("mating_attack")

    # zwischenzug: opponent just captured; we insert a forcing move and recapture later
    if prev_move and len(moves) >= 3:
        try:
            sq = chess.Move.from_uci(prev_move).to_square
        except ValueError:
            sq = None
        if sq is not None and b.color_at(sq) == (not mover) and b.attackers(mover, sq):
            recapture_now = b.is_capture(first) and first.to_square == sq
            forcing = b.gives_check(first) or (b.is_capture(first) and see(b, first) >= 0)
            if not recapture_now and forcing:
                b2 = _after(_after(b, moves[0]), moves[1]) if moves[1] in _after(b, moves[0]).legal_moves else None
                if b2 is not None and moves[2] in b2.legal_moves and b2.is_capture(moves[2]) \
                        and moves[2].to_square == sq:
                    tags.add("zwischenzug")

    # replay up to six plies, collecting combination candidates
    candidates: set[str] = set()
    cur = b.copy(stack=False)
    material_start = _material(cur, mover)
    played = 0
    for i, m in enumerate(moves[:6]):
        if m not in cur.legal_moves:
            break
        if i == 0 and cur.is_capture(m):
            defended = {t for t in cur.attacks(m.to_square) if cur.color_at(t) == (not mover)}
            if any(mm.to_square in defended for mm in moves[2:6:2]):
                candidates.add("removal_of_defender")
        if i == 1 and len(moves) >= 3:
            defended_before = {t for t in cur.attacks(m.from_square)
                               if cur.color_at(t) == (not mover) and t != m.to_square}
            if moves[2].to_square in defended_before:
                candidates.add("overload" if cur.is_capture(m) else "deflection")
        cur.push(m)
        played += 1
    if played >= 3:
        swing = _material(cur, mover) - material_start
        if swing >= 150:
            tags |= candidates
            if see(b, first) < 0:
                tags.add("sacrifice")
    return tags


def _after(board: chess.Board, move: chess.Move) -> chess.Board:
    b = board.copy(stack=False)
    b.push(move)
    return b


def _is_capture_seq(board: chess.Board, moves: list[chess.Move], idx: int) -> bool:
    b = board.copy(stack=False)
    for i, m in enumerate(moves):
        if m not in b.legal_moves:
            return False
        if i == idx:
            return b.is_capture(m)
        b.push(m)
    return False


def _material(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        total += VAL[pt] * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color)))
    return total


# ------------------------------------------------------------------ threats
def threats_against(board: chess.Board) -> set[str]:
    """What could the opponent do if the side to move passed? (null-move threat scan)"""
    if board.is_check():
        return {"in_check"}
    b = board.copy(stack=False)
    b.push(chess.Move.null())
    tags: set[str] = set()
    for m in b.legal_moves:
        if b.gives_check(m):
            b.push(m)
            mate = b.is_checkmate()
            b.pop()
            if mate:
                tags.add("mate_threat")
                continue
        pt = b.piece_type_at(m.from_square)
        if pt in (chess.KNIGHT, chess.PAWN) and not b.is_capture(m):
            after = _after(b, m)
            if len(targets_of(after, m.to_square)) >= 2 and not is_hanging(after, m.to_square):
                tags.add("knight_fork" if pt == chess.KNIGHT else "pawn_fork")
    gain, _ = best_capture_gain(b, b.turn)
    if gain >= 200:
        tags.add("hanging_piece")
    return tags


def allows_after(board_after: chess.Board) -> set[str]:
    """Cheap scan of the opponent's immediate tactical shots after our move."""
    tags: set[str] = set()
    b = board_after
    for m in b.legal_moves:
        if b.gives_check(m):
            b.push(m)
            mate = b.is_checkmate()
            b.pop()
            if mate:
                tags.add("mate_in_1")
        pt = b.piece_type_at(m.from_square)
        if pt in (chess.KNIGHT, chess.PAWN):
            after = _after(b, m)
            if len(targets_of(after, m.to_square)) >= 2 and not is_hanging(after, m.to_square):
                tags.add("knight_fork" if pt == chess.KNIGHT else "pawn_fork")
    return tags
