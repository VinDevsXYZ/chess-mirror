"""Personalised training from the player's OWN mistakes, with spaced repetition.

Every significant error becomes an exercise: the position before the mistake,
shown without the solution.  Any move within 3% win probability of the engine's
best move is accepted.  Scheduling is SM-2 (the Anki algorithm): correct answers
push the next review further out, failures bring it back tomorrow.
"""

from __future__ import annotations

import datetime as dt
import json

import chess
import chess.pgn

from ..analysis.winprob import MISTAKE, cp_to_wp
from ..config import Config
from ..db import DB
from .coach import classify_issue

ACCEPT_MARGIN = 0.03

CATEGORIES = {
    "tactics_missed": "Tactics I Missed",
    "misplayed": "Positions I Misplayed",
    "conversion": "Winning Positions I Failed to Convert",
    "defense": "Defensive Moves I Missed",
    "opening": "Opening Mistakes",
    "endgame": "Endgames I Misplayed",
}


def _category(r: dict, f: dict, m: dict) -> tuple[str, str | None]:
    cat, motif = classify_issue(r, f, m, None)
    if r["wp_before"] >= 0.80:
        return "conversion", motif
    if cat == "missed opportunity" or (motif and cat == "tactical oversight" and f.get("best_gap", 0) >= 0.10):
        return "tactics_missed", motif
    if (m.get("threats") and r["wp_before"] <= 0.6) or (f.get("only_move") and r["wp_before"] <= 0.5):
        return "defense", motif
    if r["phase"] == "opening":
        return "opening", motif
    if r["phase"] == "endgame":
        return "endgame", motif
    return "misplayed", motif


def build_items(db: DB) -> int:
    rows = [dict(r) for r in db.q("""
        SELECT p.* FROM positions p JOIN games g ON g.id = p.game_id
        WHERE p.is_player = 1 AND p.wp_loss >= ? AND p.wp_before BETWEEN 0.05 AND 0.97
          AND p.id NOT IN (SELECT position_id FROM training_items WHERE position_id IS NOT NULL)""", (MISTAKE,))]
    now = dt.datetime.now()
    n = 0
    with db.tx() as conn:
        for r in rows:
            f, m = json.loads(r["features"]), json.loads(r["motifs"])
            top = json.loads(r["top_moves"])
            if not top:
                continue
            best_wp = cp_to_wp(top[0]["cp"], top[0]["mate"])
            sols = [t["uci"] for t in top if best_wp - cp_to_wp(t["cp"], t["mate"]) <= ACCEPT_MARGIN]
            if r["move_uci"] in sols:
                continue
            cat, motif = _category(r, f, m)
            b = chess.Board(r["fen"])
            side = "White" if b.turn == chess.WHITE else "Black"
            prompt = f"{side} to move. " + {
                "tactics_missed": "There is a tactic here. Find it.",
                "defense": "Your position is under pressure. Find the best defence.",
                "conversion": "You are winning. Find the move that keeps control.",
                "opening": "Find the best continuation in this opening position.",
                "endgame": "Find the most precise endgame move.",
                "misplayed": "Find the best move.",
            }[cat]
            conn.execute("""INSERT INTO training_items(position_id, game_id, fen, category, solutions, best_san,
                            played_san, prompt, motif, created_at, due_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                         (r["id"], r["game_id"], r["fen"], cat, json.dumps(sols), r["best_san"], r["move_san"],
                          prompt, motif, now.isoformat(timespec="seconds"), now.isoformat(timespec="seconds")))
            n += 1
    return n


def due_items(db: DB, limit: int = 20, category: str | None = None) -> list[dict]:
    """Due reviews first (oldest first), then new exercises ordered by how costly the mistake was."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    q = """SELECT t.*, p.wp_loss AS cost FROM training_items t LEFT JOIN positions p ON p.id = t.position_id
           WHERE t.status = 'active' AND t.due_at <= ?"""
    params: list = [now]
    if category:
        q += " AND t.category = ?"
        params.append(category)
    q += " ORDER BY (t.reps = 0 AND t.lapses = 0), CASE WHEN t.reps + t.lapses > 0 THEN t.due_at END, cost DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in db.q(q, tuple(params))]


def sm2(ease: float, interval: float, reps: int, quality: int) -> tuple[float, float, int]:
    """SM-2 update. quality: 5 = correct quickly, 3 = correct slowly, 1 = wrong."""
    if quality < 3:
        return max(1.3, ease - 0.2), 1.0, 0
    reps += 1
    interval = 1.0 if reps == 1 else 3.0 if reps == 2 else interval * ease
    ease = max(1.3, ease + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return ease, interval, reps


def answer(db: DB, item_id: int, uci: str, seconds: float | None = None) -> dict:
    it = db.one("SELECT * FROM training_items WHERE id = ?", (item_id,))
    if it is None:
        raise ValueError("unknown item")
    sols = json.loads(it["solutions"])
    correct = uci in sols
    quality = (5 if (seconds is None or seconds < 30) else 3) if correct else 1
    ease, interval, reps = sm2(it["ease"], it["interval_days"], it["reps"], quality)
    now = dt.datetime.now()
    due = now + dt.timedelta(days=interval)
    with db.tx() as conn:
        conn.execute("""UPDATE training_items SET ease=?, interval_days=?, reps=?, lapses=lapses+?, due_at=?,
                        last_reviewed_at=?, ever_solved=MAX(ever_solved, ?) WHERE id=?""",
                     (ease, interval, reps, 0 if correct else 1, due.isoformat(timespec="seconds"),
                      now.isoformat(timespec="seconds"), int(correct), item_id))
        conn.execute("INSERT INTO training_reviews(item_id, reviewed_at, answer_uci, correct, seconds) VALUES (?,?,?,?,?)",
                     (item_id, now.isoformat(timespec="seconds"), uci, int(correct), seconds))
    b = chess.Board(it["fen"])
    best_uci = sols[0] if sols else None
    return {"correct": correct, "best_san": it["best_san"], "best_uci": best_uci, "played_in_game": it["played_san"],
            "accepted": [b.san(chess.Move.from_uci(s)) for s in sols], "next_due": due.date().isoformat()}


def stats(db: DB) -> dict:
    now = dt.datetime.now().isoformat(timespec="seconds")
    by_cat = {r["category"]: {"total": r["n"], "due": r["due"], "solved_later": r["solved"]} for r in db.q(
        """SELECT category, COUNT(*) n, SUM(due_at <= ?) due, SUM(ever_solved) solved FROM training_items
           WHERE status='active' GROUP BY category""", (now,))}
    first_fail_then_solved = db.scalar("""
        SELECT COUNT(*) FROM training_items t WHERE
          (SELECT correct FROM training_reviews r WHERE r.item_id = t.id ORDER BY r.id LIMIT 1) = 0
          AND t.ever_solved = 1""") or 0
    reviews = db.one("SELECT COUNT(*) n, SUM(correct) k FROM training_reviews")
    return {"by_category": {CATEGORIES[k]: v for k, v in by_cat.items()},
            "failed_then_solved": first_fail_then_solved,
            "reviews": reviews["n"] or 0, "correct": reviews["k"] or 0}


def due_text(db: DB) -> str:
    s = stats(db)
    if not s["by_category"]:
        return "No training positions yet: run `mirror trainer build` after analysing games."
    L = ["TRAINING"]
    for cat, v in s["by_category"].items():
        L.append(f"  {cat:40s} {v['due'] or 0:3d} due / {v['total']:3d} total, {v['solved_later'] or 0} solved at least once")
    L.append(f"  Positions you failed at first and later solved: {s['failed_then_solved']}")
    L.append(f"  Reviews: {s['reviews']} ({s['correct']} correct)")
    return "\n".join(L)


def run_cli(db: DB, limit: int = 10) -> None:
    import time
    items = due_items(db, limit)
    if not items:
        print("Nothing due. Come back later (or run `mirror trainer build`).")
        return
    for it in items:
        b = chess.Board(it["fen"])
        print("\n" + (str(b) if b.turn == chess.WHITE else str(b.transform(chess.flip_vertical).transform(chess.flip_horizontal))))
        print(f"[{CATEGORIES[it['category']]}] {it['prompt']}")
        t0 = time.time()
        ans = input("Your move (SAN, or 'skip'/'quit'): ").strip()
        if ans == "quit":
            break
        if ans == "skip":
            continue
        try:
            uci = b.parse_san(ans).uci()
        except ValueError:
            print("Illegal/unreadable move; counted as wrong.")
            uci = "0000"
        r = answer(db, it["id"], uci, time.time() - t0)
        print(("Correct! " if r["correct"] else f"Not quite. Best was {r['best_san']}. ") +
              f"In the game you played {r['played_in_game']}. Next review: {r['next_due']}")


def export_pgn(db: DB, cfg: Config, limit: int = 50) -> str:
    items = due_items(db, limit)
    path = cfg.reports_dir / "training-due.pgn"
    games = []
    for it in items:
        g = chess.pgn.Game()
        g.setup(chess.Board(it["fen"]))
        g.headers["Event"] = f"Training: {CATEGORIES[it['category']]}"
        g.headers["Site"] = f"mirror item {it['id']}"
        g.headers["White"] = g.headers["Black"] = "?"
        g.comment = it["prompt"]
        games.append(str(g))
    path.write_text("\n\n".join(games) + "\n")
    return str(path)
