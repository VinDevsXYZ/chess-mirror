"""PGN import: original games are preserved, each ply becomes a datapoint.

Only plies made by the tracked player get ``is_player = 1``.  Opponent plies are
stored with ``is_player = 0`` and are used exclusively as a *population* sample
(an Elo-matched prior / baseline) -- never as the player's own decisions.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from .config import Config
from .db import DB

TEST_FRACTION = 0.20  # share of games permanently held out for evaluation

ENGINE_NAME_RE = re.compile(r"stockfish|komodo|leela|lc0|\bai\b|computer|engine|bot\b|mirror", re.I)


@dataclass
class ImportStats:
    games_seen: int = 0
    imported: int = 0
    duplicates: int = 0
    skipped_variant: int = 0
    no_player: int = 0
    player_positions: int = 0


def is_nonstandard_start(headers) -> bool:
    """Games that do not start from the normal initial position (odds chess, custom set-ups).

    They are a different game from the one being modelled, and Stockfish 19 refuses many
    such positions outright ("Unsupported position. Too many pieces")."""
    fen = headers.get("FEN")
    if fen and fen.split()[:4] != chess.STARTING_FEN.split()[:4]:
        return True
    return "odds" in headers.get("Event", "").lower()


def purge_nonstandard(db: DB) -> int:
    """Remove previously imported non-standard games (the original PGN stays in data/raw)."""
    ids = []
    for gid, pgn in db.q("SELECT id, pgn FROM games"):
        game = chess.pgn.read_headers(io.StringIO(pgn))
        if game is not None and is_nonstandard_start(game):
            ids.append(gid)
    if ids:
        with db.tx() as conn:
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM training_items WHERE game_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM elo_games WHERE game_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM positions WHERE game_id IN ({marks})", ids)
            conn.execute(f"DELETE FROM games WHERE id IN ({marks})", ids)
    return len(ids)


def parse_time_control(tc: str | None) -> tuple[int | None, int | None, str]:
    if not tc or tc in ("-", "?"):
        return None, None, "unknown"
    if tc.startswith("1/") or "/" in tc:
        return None, None, "correspondence"
    m = re.match(r"^(\d+)(?:\+(\d+))?$", tc.strip())
    if not m:
        return None, None, "unknown"
    base, inc = int(m.group(1)), int(m.group(2) or 0)
    est = base + 40 * inc
    if est < 180:
        cls = "bullet"
    elif est < 480:
        cls = "blitz"
    elif est < 1500:
        cls = "rapid"
    else:
        cls = "classical"
    return base, inc, cls


def split_for_hash(game_hash: str) -> str:
    """Deterministic, permanent train/test assignment derived from the game hash."""
    return "test" if int(game_hash[:8], 16) % 1000 < TEST_FRACTION * 1000 else "train"


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v not in (None, "", "?") else None
    except ValueError:
        return None


def _platform(site: str, source: str) -> str:
    s = (site or "").lower()
    if s.startswith("synthetic"):
        return "synthetic"
    if "lichess" in s:
        return "lichess"
    if "chess.com" in s:
        return "chesscom"
    if source.startswith("mirror-engine"):
        return "mirror-engine"
    return "otb" if s and "?" not in s else "unknown"


def _game_hash(game: chess.pgn.Game) -> str:
    h = game.headers
    key = "|".join(
        [h.get("White", ""), h.get("Black", ""), h.get("UTCDate", h.get("Date", "")),
         h.get("UTCTime", ""), h.get("Site", ""), h.get("FEN", "")]
    )
    moves = " ".join(m.uci() for m in game.mainline_moves())
    return hashlib.sha1(f"{key}#{moves}".encode()).hexdigest()


def _date(h) -> tuple[str | None, str | None]:
    d = h.get("UTCDate") or h.get("Date") or ""
    d = d.replace(".", "-")
    if "?" in d or not re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        return (d[:4] if re.match(r"^\d{4}", d) else None), None
    t = h.get("UTCTime") or "00:00:00"
    return d, f"{d}T{t}"


def import_game(db: DB, cfg: Config, game: chess.pgn.Game, source: str,
                force_color: str | None = None, stats: ImportStats | None = None) -> int | None:
    stats = stats or ImportStats()
    stats.games_seen += 1
    h = game.headers
    variant = h.get("Variant", "Standard")
    if variant.lower() not in ("standard", "from position", "chess"):
        stats.skipped_variant += 1
        return None
    if is_nonstandard_start(h):
        stats.skipped_variant += 1  # odds chess / custom set-ups are not normal chess for this model
        return None
    moves = list(game.mainline_moves())
    if not moves:
        return None

    ghash = _game_hash(game)
    if db.one("SELECT id FROM games WHERE game_hash = ?", (ghash,)):
        stats.duplicates += 1
        return None

    names = cfg.player.all_names()
    white, black = h.get("White", "?"), h.get("Black", "?")
    if force_color in ("white", "black"):
        color = force_color
    elif white.strip().lower() in names:
        color = "white"
    elif black.strip().lower() in names:
        color = "black"
    else:
        color = None
        stats.no_player += 1

    result = h.get("Result", "*")
    score_white = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)
    player_score = None
    if color and score_white is not None:
        player_score = score_white if color == "white" else 1.0 - score_white

    base, inc, tc_class = parse_time_control(h.get("TimeControl"))
    date, utc = _date(h)
    welo, belo = _int(h.get("WhiteElo")), _int(h.get("BlackElo"))
    opp = black if color == "white" else white if color == "black" else None
    opp_elo = belo if color == "white" else welo if color == "black" else None
    player_elo = welo if color == "white" else belo if color == "black" else None
    opp_engine = False
    if color:
        oc = "Black" if color == "white" else "White"
        opp_engine = (
            h.get(f"{oc}IsComp", "").lower() == "yes"
            or h.get(f"{oc}Title", "") == "BOT"
            or bool(ENGINE_NAME_RE.search(opp or ""))
        )

    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
    pgn_text = game.accept(exporter)

    with db.tx() as conn:
        cur = conn.execute(
            """INSERT INTO games(game_hash, source, platform, site, event, date, utc_datetime, white, black,
                 white_elo, black_elo, player_color, opponent, opponent_elo, player_elo, result, player_score,
                 time_control, tc_base, tc_inc, tc_class, eco, opening, termination, variant, rated,
                 opponent_is_engine, num_plies, split, pgn, imported_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ghash, source, _platform(h.get("Site", ""), source), h.get("Site"), h.get("Event"), date, utc,
             white, black, welo, belo, color, opp, opp_elo, player_elo, result, player_score,
             h.get("TimeControl"), base, inc, tc_class, h.get("ECO"), h.get("Opening"),
             h.get("Termination"), variant,
             0 if "casual" in (h.get("Event", "").lower()) else 1,
             int(opp_engine), len(moves), split_for_hash(ghash), pgn_text,
             dt.datetime.now().isoformat(timespec="seconds")),
        )
        game_id = cur.lastrowid
        board = game.board()
        last_clock = {chess.WHITE: base, chess.BLACK: base}
        prev_uci = None
        rows = []
        for ply, node in enumerate(game.mainline()):
            move = node.move
            mover = board.turn
            is_player = int(color is not None and (mover == chess.WHITE) == (color == "white"))
            clock_after = node.clock()
            clock_before = last_clock[mover]
            time_spent = None
            emt = node.emt() if hasattr(node, "emt") else None
            if emt is not None:
                time_spent = float(emt)
            elif clock_after is not None and clock_before is not None:
                time_spent = max(0.0, clock_before + (inc or 0) - clock_after) if ply >= 2 else None
            rows.append((
                game_id, ply, board.fullmove_number, "white" if mover == chess.WHITE else "black",
                is_player, board.fen(), move.uci(), board.san(move), prev_uci,
                clock_before, clock_after, time_spent, last_clock[not mover],
            ))
            if clock_after is not None:
                last_clock[mover] = clock_after
            prev_uci = move.uci()
            board.push(move)
            stats.player_positions += is_player
        conn.executemany(
            """INSERT INTO positions(game_id, ply, move_number, side, is_player, fen, move_uci, move_san,
                 prev_move_uci, clock_before, clock_after, time_spent, opp_clock)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
    stats.imported += 1
    return game_id


def import_pgn_text(db: DB, cfg: Config, text: str, source: str, force_color: str | None = None,
                    stats: ImportStats | None = None) -> ImportStats:
    stats = stats or ImportStats()
    handle = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        if game.errors:
            # keep going: python-chess stops the mainline at the first illegal move
            pass
        import_game(db, cfg, game, source, force_color, stats)
    return stats


def import_pgn_file(db: DB, cfg: Config, path: str | Path, force_color: str | None = None) -> ImportStats:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    keep = cfg.raw_dir / f"{path.stem}-{digest}.pgn"
    if not keep.exists():
        shutil.copyfile(path, keep)  # the original file is kept verbatim
    return import_pgn_text(db, cfg, text, f"pgn:{path.name}", force_color)
