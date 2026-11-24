"""COACH MODE: prioritised, explained review of a game.

Only significant errors are reviewed (win-probability drop >= 10%, position not
already decided), largest first.  Explanations are built from facts computed on
the board -- the engine's refutation, the motif it contains, the material it
wins, and the position features that differ between your move and the best
move -- never from invented psychology.
"""

from __future__ import annotations

import json

import chess

from ..analysis.features import position_features
from ..analysis.motifs import VAL
from ..analysis.winprob import BLUNDER, INACCURACY, MISTAKE
from ..config import Config
from ..db import DB
from .concepts import CATEGORY, CONCEPTS

TACTICAL = ["mate", "mating_attack", "back_rank", "knight_fork", "pawn_fork", "fork", "double_check",
            "discovered_check", "discovered_attack", "skewer", "pin", "removal_of_defender", "overload",
            "deflection", "zwischenzug", "double_attack", "capture_hanging", "sacrifice", "promotion"]
PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen"}
MAX_ISSUES = 5


def latest_game_id(db: DB) -> int | None:
    return db.scalar("SELECT id FROM games WHERE analyzed_at IS NOT NULL AND player_color IS NOT NULL "
                     "ORDER BY COALESCE(utc_datetime, date) DESC, id DESC LIMIT 1")


def san_line(fen: str, ucis: list[str], limit: int = 8) -> str:
    b = chess.Board(fen)
    out = []
    for u in ucis[:limit]:
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        num = f"{b.fullmove_number}." if b.turn == chess.WHITE else (f"{b.fullmove_number}..." if not out else "")
        out.append(f"{num}{b.san(m)}")
        b.push(m)
    return " ".join(out)


def material_change(fen: str, ucis: list[str], color: chess.Color, max_plies: int = 10) -> dict:
    """Net material change for ``color`` along a line, measured at the first quiet point
    (after >= 4 plies, last move not a capture, no check) so exchanges are complete."""
    b = chess.Board(fen)
    start = {pt: (len(b.pieces(pt, color)), len(b.pieces(pt, not color))) for pt in PIECE}
    for i, u in enumerate(ucis[:max_plies]):
        m = chess.Move.from_uci(u)
        if m not in b.legal_moves:
            break
        was_capture = b.is_capture(m)
        b.push(m)
        if i >= 3 and not was_capture and not b.is_check():
            break
    net = {pt: (len(b.pieces(pt, color)) - start[pt][0]) - (len(b.pieces(pt, not color)) - start[pt][1])
           for pt in PIECE}
    return {"net": net, "net_cp": sum(VAL[pt] * v for pt, v in net.items()), "mate": b.is_checkmate()}


def _count_words(items: dict) -> str:
    words = [f"a {PIECE[pt]}" if n == 1 else f"{n} {PIECE[pt]}s" for pt, n in items.items()]
    return " and ".join(words) if len(words) <= 2 else ", ".join(words[:-1]) + " and " + words[-1]


def describe_material(ch: dict) -> str:
    if ch["mate"]:
        return "it ends in checkmate"
    lost = {pt: -v for pt, v in ch["net"].items() if v < 0}
    won = {pt: v for pt, v in ch["net"].items() if v > 0}
    if abs(ch["net_cp"]) < 80:
        return "material stays roughly level"
    if lost and won:
        verb = "you give up" if ch["net_cp"] < 0 else "you trade"
        return f"{verb} {_count_words(lost)} for {_count_words(won)} (net {ch['net_cp'] / 100:+.0f} pawns)"
    if lost:
        return f"you lose {_count_words(lost)}"
    return f"you win {_count_words(won)}"


def _feature_diff(fen: str, move_a: str, move_b: str) -> list[str]:
    """Plain-language differences between the positions after move_a (played) and move_b (best)."""
    base = chess.Board(fen)

    def feats(u):
        b = base.copy()
        b.push_uci(u)
        f = position_features(b)  # side to move is now the opponent: "_op" fields describe us
        return {"my_shield": f["shield_op"], "my_weak": sum(f["pawns_op"][k] for k in ("isolated", "doubled", "backward")),
                "my_mob": f["mobility_op"], "their_mob": f["mobility_me"], "my_passed": f["pawns_op"]["passed"],
                "their_passed": f["pawns_me"]["passed"], "hanging_mine": f["hanging_op"],
                "outposts_theirs": f["outposts_me"]}
    a, b = feats(move_a), feats(move_b)
    out = []
    if a["hanging_mine"] > b["hanging_mine"]:
        out.append(f"it leaves {a['hanging_mine']} of your pieces loose (attacked and insufficiently defended)")
    if a["my_weak"] > b["my_weak"]:
        out.append("it creates a pawn weakness (isolated/doubled/backward) that the better move avoids")
    if a["my_shield"] < b["my_shield"]:
        out.append("it weakens the pawn cover in front of your king")
    if a["their_mob"] - b["their_mob"] >= 4:
        out.append(f"it gives your opponent's pieces more scope ({a['their_mob']} legal moves vs {b['their_mob']})")
    if b["my_mob"] - a["my_mob"] >= 4:
        out.append(f"your pieces end up less active ({a['my_mob']} legal moves vs {b['my_mob']} after the better move)")
    if a["their_passed"] > b["their_passed"]:
        out.append("it lets the opponent get a passed pawn")
    if b["my_passed"] > a["my_passed"]:
        out.append("the better move creates a passed pawn for you")
    if a["outposts_theirs"] > b["outposts_theirs"]:
        out.append("it hands the opponent an outpost square for a piece")
    return out


def classify_issue(r: dict, f: dict, m: dict, base: int | None) -> tuple[str, str | None]:
    allowed, best = m.get("allowed", []), m.get("best", [])
    unparried = m.get("threats_unparried", [])
    time_pressure = f.get("time_pressure")
    if r["time_spent"] is not None and r["time_spent"] < 4 and f.get("only_move"):
        return "time management", None
    for mo in ("mate", "mating_attack", "back_rank"):
        if mo in allowed:
            return "king safety", mo
    for th in unparried:
        mo = {"hanging_piece": "capture_hanging", "mate_threat": "mating_attack"}.get(th, th)
        return "tactical oversight", mo
    for mo in TACTICAL:
        if mo in allowed:
            return "tactical oversight", mo
    if f.get("best_gap", 0) >= 0.10:
        for mo in TACTICAL:
            if mo in best:
                return "missed opportunity", mo
    if time_pressure:
        return "time management", None
    if r["phase"] == "opening":
        return "opening knowledge", None
    if r["phase"] == "endgame":
        return "endgame technique", None
    if r["wp_before"] >= 0.75 and "sacrifice" in m.get("played", []):
        return "unnecessary risk", "sacrifice"
    if r["played_rank"] is not None:
        return "calculation error", None
    return "positional misunderstanding", None


def review_game(db: DB, game_id: int) -> dict:
    g = db.one("SELECT * FROM games WHERE id = ?", (game_id,))
    if g is None:
        raise ValueError(f"no game {game_id}")
    rows = [dict(r) for r in db.q("SELECT * FROM positions WHERE game_id = ? AND is_player = 1 AND wp_loss IS NOT NULL "
                                  "ORDER BY ply", (game_id,))]
    color = chess.WHITE if g["player_color"] == "white" else chess.BLACK
    counts = {"inaccuracies": 0, "mistakes": 0, "blunders": 0}
    candidates = []
    for r in rows:
        wl = r["wp_loss"]
        if wl >= BLUNDER:
            counts["blunders"] += 1
        elif wl >= MISTAKE:
            counts["mistakes"] += 1
        elif wl >= INACCURACY:
            counts["inaccuracies"] += 1
        decided = (r["wp_before"] > 0.97 and r["wp_after"] > 0.90) or r["wp_before"] < 0.03
        if wl >= MISTAKE and not decided:
            candidates.append(r)
    candidates.sort(key=lambda r: -r["wp_loss"])
    issues = []
    for r in candidates[:MAX_ISSUES]:
        f, m = json.loads(r["features"]), json.loads(r["motifs"])
        cat, motif = classify_issue(r, f, m, g["tc_base"])
        played_pv = json.loads(r["played_pv"] or "[]")
        top = json.loads(r["top_moves"] or "[]")
        best_pv = top[0]["pv"] if top else [r["best_uci"]]
        board = chess.Board(r["fen"])
        mat_played = material_change(r["fen"], played_pv, color)
        mat_best = material_change(r["fen"], best_pv, color)
        refutation = None
        if len(played_pv) >= 2:
            after = board.copy()
            after.push_uci(played_pv[0])
            try:
                refutation = after.san(chess.Move.from_uci(played_pv[1]))
            except Exception:
                refutation = None
        # a line that simply loses material (vs the better move) is a tactical problem, whatever the phase
        material_gap = mat_played["net_cp"] - mat_best["net_cp"]
        if cat not in ("tactical oversight", "king safety", "missed opportunity", "time management") \
                and material_gap <= -200 and refutation:
            cat, motif = ("calculation error" if r["played_rank"] is not None else "tactical oversight"), None
        why = []
        if cat in ("tactical oversight", "king safety", "calculation error") and refutation and material_gap <= -200 \
                and not motif:
            why.append(f"After your move the opponent has {refutation}; along the engine line "
                       f"{describe_material(mat_played)}, which {r['best_san']} avoids.")
        elif cat in ("tactical oversight", "king safety") and refutation:
            mo = CONCEPTS.get(motif, ("", "", ""))[0] if motif else "a forcing reply"
            why.append(f"After your move the opponent has {refutation}" + (f" ({mo.lower()})" if motif else "") +
                       f"; along the engine line {describe_material(mat_played)}.")
        if cat == "missed opportunity":
            why.append(f"{r['best_san']} was a {CONCEPTS.get(motif, ('tactic',))[0].lower()}: "
                       f"along its line {describe_material(mat_best)}.")
        if not why:
            diffs = _feature_diff(r["fen"], r["move_uci"], r["best_uci"])
            if diffs:
                why.append("Compared with the better move, " + "; ".join(diffs) + ".")
            else:
                why.append(f"No immediate tactic: along the engine line after your move {describe_material(mat_played)}, "
                           f"but the evaluation drops {r['wp_loss'] * 100:.0f}% in win chance compared with "
                           f"{r['best_san']} -- the difference is positional/long-term (see the lines).")
        if m.get("threats"):
            why.append("Threats that existed before your move: " +
                       ", ".join(t.replace("_", " ") for t in m["threats"]) + ".")
        concept = CONCEPTS.get(motif) if motif else None
        cat_c = CATEGORY[cat]
        issues.append({
            "ply": r["ply"], "move_label": f"{r['move_number']}{'.' if board.turn == chess.WHITE else '...'}",
            "fen": r["fen"], "played": r["move_san"], "best": r["best_san"],
            "played_uci": r["move_uci"], "best_uci": r["best_uci"],
            "eval_before": r["eval_before_cp"], "eval_after": r["eval_after_cp"],
            "wp_before": r["wp_before"], "wp_after": r["wp_after"], "wp_loss": r["wp_loss"],
            "category": cat, "motif": motif, "why": why,
            "overlooked": (f"the reply {refutation}" if refutation and cat in ("tactical oversight", "king safety", "calculation error")
                           else f"the idea {r['best_san']}" if cat == "missed opportunity" else None),
            "refutation_line": san_line(board.fen(), played_pv),
            "best_line": san_line(board.fen(), best_pv),
            "concept": {"name": (concept or cat_c)[0], "idea": (concept or cat_c)[1]},
            "recognize": (concept or cat_c)[2],
            "time_spent": r["time_spent"], "clock": r["clock_before"],
        })
    accs = [r["accuracy"] for r in rows if r["accuracy"] is not None]
    n_player = db.scalar("SELECT COUNT(*) FROM positions WHERE game_id = ? AND is_player = 1", (game_id,)) or 0
    return {"game_id": game_id, "analyzed": bool(rows), "analyzed_moves": len(rows), "player_moves": n_player, "white": g["white"], "black": g["black"], "result": g["result"],
            "date": g["date"], "opening": g["opening"], "player_color": g["player_color"],
            "accuracy": sum(accs) / len(accs) if accs else None, "counts": counts,
            "issues": issues, "minor_issues_skipped": max(0, len(candidates) - len(issues)) + counts["inaccuracies"]}


def _eval(cp) -> str:
    return f"{cp / 100:+.1f}" if cp is not None else "?"


def render_markdown(rv: dict, recurring: dict | None = None) -> str:
    if not rv.get("analyzed"):
        return (f"# Game review: {rv['white']} vs {rv['black']} ({rv['result']})\n\n"
                "This game has not been analysed yet, so there is nothing to review. "
                "Run `mirror analyze` (or *Update everything* in the app).")
    L = [f"# Game review: {rv['white']} vs {rv['black']} ({rv['result']}, {rv['date'] or ''})",
         f"You played **{rv['player_color']}**. Accuracy **{rv['accuracy']:.0f}**. "
         f"Blunders {rv['counts']['blunders']}, mistakes {rv['counts']['mistakes']}, inaccuracies {rv['counts']['inaccuracies']}.",
         ""]
    if not rv["issues"]:
        L.append("No significant mistakes (none lost 10%+ win probability in an undecided position). Well played.")
    for i, it in enumerate(rv["issues"], 1):
        L += [f"## {i}. {it['move_label']}{it['played']}  — {it['category']}"
              + (f" ({it['motif'].replace('_', ' ')})" if it["motif"] else ""),
              f"- **Your move:** {it['played']}   **Best:** {it['best']}",
              f"- **Evaluation (from your side):** {_eval(it['eval_before'])} → {_eval(it['eval_after'])} "
              f"(win chance {it['wp_before'] * 100:.0f}% → {it['wp_after'] * 100:.0f}%)"]
        if it["time_spent"] is not None:
            L.append(f"- **Time:** {it['time_spent']:.0f}s spent" + (f", {it['clock']:.0f}s on the clock" if it["clock"] else ""))
        L.append(f"- **Why:** {' '.join(it['why'])}")
        if it["overlooked"]:
            L.append(f"- **What you probably overlooked:** {it['overlooked']}")
        L.append(f"- **Line after your move:** {it['refutation_line']}")
        L.append(f"- **Better:** {it['best_line']}")
        L.append(f"- **Concept — {it['concept']['name']}:** {it['concept']['idea']}")
        L.append(f"- **Recognise it next time:** {it['recognize']}")
        if recurring and it["motif"] and recurring.get(it["motif"]):
            L.append(f"- **Pattern:** {recurring[it['motif']]}")
        L.append("")
    if rv["minor_issues_skipped"]:
        L.append(f"_{rv['minor_issues_skipped']} smaller inaccuracies were not reviewed to keep the focus on what matters._")
    return "\n".join(L)


def coach_game(db: DB, cfg: Config, game_id: int) -> tuple[str, str]:
    rv = review_game(db, game_id)
    recurring = {}
    rows = db.q("SELECT key, data FROM weaknesses")
    motif_keys = {"knight_fork": ["missed_knight_fork", "unparried_knight_fork"], "fork": ["allowed_fork", "missed_fork"],
                  "capture_hanging": ["unparried_hanging", "missed_free_material"], "pin": ["allowed_pin_skewer", "missed_pin"],
                  "skewer": ["allowed_pin_skewer", "missed_skewer"], "mating_attack": ["allowed_mating_attack", "missed_mate"]}
    data = {r["key"]: json.loads(r["data"]) for r in rows}
    for motif, keys in motif_keys.items():
        for k in keys:
            if k in data and data[k].get("is_weakness"):
                recurring[motif] = data[k]["text"]
                break
    md = render_markdown(rv, recurring)
    path = cfg.reports_dir / f"coach-game-{game_id}.md"
    path.write_text(md)
    return md, str(path)
