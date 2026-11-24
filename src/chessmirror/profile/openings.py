"""Opening fingerprint: repertoire, frequencies, performance, accuracy, recurring errors.

The Mirror engine reproduces this repertoire probabilistically through its
position book (Dirichlet-smoothed counts of the player's own moves), so the
numbers here are exactly the frequencies the clone plays with.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import chess

from ..analysis.winprob import MISTAKE
from ..db import DB
from .evidence import beta_ci

# Common opening names by SAN move prefix (longest match wins).
NAMES = {
    "e4": "King's Pawn", "e4 e5": "Open Game", "e4 c5": "Sicilian Defence", "e4 c6": "Caro-Kann Defence",
    "e4 e6": "French Defence", "e4 d5": "Scandinavian Defence", "e4 d6": "Pirc Defence", "e4 g6": "Modern Defence",
    "e4 Nf6": "Alekhine Defence", "e4 Nc6": "Nimzowitsch Defence", "e4 b6": "Owen Defence",
    "e4 e5 Nf3 Nc6 Bb5": "Ruy Lopez", "e4 e5 Nf3 Nc6 Bc4": "Italian Game", "e4 e5 Nf3 Nc6 d4": "Scotch Game",
    "e4 e5 Nf3 Nf6": "Petrov Defence", "e4 e5 Nf3 d6": "Philidor Defence", "e4 e5 f4": "King's Gambit",
    "e4 e5 Nc3": "Vienna Game", "e4 e5 Bc4": "Bishop's Opening", "e4 e5 Nf3 Nc6 Nc3": "Four Knights / Three Knights",
    "e4 c5 Nf3 d6": "Sicilian, 2...d6", "e4 c5 Nf3 Nc6": "Sicilian, 2...Nc6", "e4 c5 Nf3 e6": "Sicilian, 2...e6",
    "e4 c5 Nc3": "Closed Sicilian", "e4 c5 c3": "Alapin Sicilian", "e4 c5 d4": "Smith-Morra Gambit",
    "e4 c6 d4 d5 e5": "Caro-Kann, Advance", "e4 c6 d4 d5 Nc3": "Caro-Kann, Main line",
    "e4 e6 d4 d5 e5": "French, Advance", "e4 e6 d4 d5 Nc3": "French, 3.Nc3", "e4 e6 d4 d5 Nd2": "French, Tarrasch",
    "d4": "Queen's Pawn", "d4 d5": "Queen's Pawn Game", "d4 d5 c4": "Queen's Gambit", "d4 d5 c4 e6": "Queen's Gambit Declined",
    "d4 d5 c4 c6": "Slav Defence", "d4 d5 c4 dxc4": "Queen's Gambit Accepted", "d4 d5 Bf4": "London System",
    "d4 Nf6": "Indian Defence", "d4 Nf6 Bf4": "London System", "d4 Nf6 c4 g6": "King's Indian / Grünfeld",
    "d4 Nf6 c4 e6": "Nimzo / Queen's Indian complex", "d4 Nf6 c4 c5": "Benoni complex", "d4 f5": "Dutch Defence",
    "d4 e6": "Horwitz / French transposition", "d4 g6": "Modern vs d4", "d4 d5 Nf3": "Queen's Pawn, 2.Nf3",
    "c4": "English Opening", "Nf3": "Réti Opening", "g3": "King's Fianchetto", "b3": "Nimzo-Larsen Attack",
    "f4": "Bird Opening", "e3": "Van 't Kruijs", "Nc3": "Van Geet",
}


def name_line(sans: list[str]) -> str:
    for k in range(min(len(sans), 6), 0, -1):
        key = " ".join(sans[:k])
        if key in NAMES:
            return NAMES[key]
    return " ".join(sans[:4]) or "?"


def opening_profile(db: DB, plies: int = 24) -> dict:
    games = db.q("""SELECT id, player_color, player_score, opening, eco FROM games
                    WHERE player_color IS NOT NULL AND platform != 'mirror-engine' ORDER BY id""")
    pos = db.q(f"""SELECT game_id, ply, move_san, is_player, wp_loss, accuracy, fen, best_san FROM positions
                   WHERE ply < {plies} ORDER BY game_id, ply""")
    by_game = defaultdict(list)
    for r in pos:
        by_game[r["game_id"]].append(r)
    out = {}
    for color in ("white", "black"):
        gs = [g for g in games if g["player_color"] == color]
        lines = defaultdict(lambda: {"games": 0, "score": 0.0, "acc": [], "mistakes": 0, "moves": 0})
        first_choice = defaultdict(Counter)  # context -> my move counts
        errors = Counter()
        err_example = {}
        for g in gs:
            ps = by_game.get(g["id"], [])
            sans = [p["move_san"] for p in ps]
            name = g["opening"] or name_line(sans)
            L = lines[name]
            L["games"] += 1
            L["score"] += g["player_score"] or 0.0
            for p in ps:
                if p["is_player"]:
                    L["moves"] += 1
                    if p["accuracy"] is not None:
                        L["acc"].append(p["accuracy"])
                    if p["wp_loss"] is not None and p["wp_loss"] >= MISTAKE:
                        L["mistakes"] += 1
                        key = chess.Board(p["fen"]).epd()
                        errors[key] += 1
                        err_example[key] = (" ".join(sans[:p["ply"]]), p["move_san"], p["best_san"])
            # repertoire decisions: my first two moves in context
            mine = [i for i, p in enumerate(ps) if p["is_player"]][:3]
            for i in mine:
                ctx = " ".join(sans[:i]) or "(start)"
                first_choice[ctx][sans[i]] += 1
        rep = []
        for name, L in lines.items():
            n = L["games"]
            m, lo, hi = beta_ci(L["score"], n)
            rep.append({"opening": name, "games": n, "share": n / max(1, len(gs)),
                        "score": L["score"] / n, "score_ci": [lo, hi],
                        "accuracy": sum(L["acc"]) / len(L["acc"]) if L["acc"] else None,
                        "mistakes_per_game": L["mistakes"] / n})
        rep.sort(key=lambda x: -x["games"])
        choices = []
        for ctx, cnt in sorted(first_choice.items(), key=lambda kv: -sum(kv[1].values())):
            total = sum(cnt.values())
            if total < 3:
                continue
            choices.append({"after": ctx, "n": total,
                            "moves": [{"san": s, "count": c, "share": c / total} for s, c in cnt.most_common(5)]})
        recurring = [{"line": err_example[k][0], "played": err_example[k][1], "better": err_example[k][2],
                      "times": c} for k, c in errors.most_common(8) if c >= 2]
        comfortable = [r for r in rep if r["games"] >= 5]
        comfortable.sort(key=lambda r: -(r["accuracy"] or 0))
        out[color] = {"games": len(gs), "repertoire": rep[:15], "choices": choices[:10],
                      "recurring_opening_mistakes": recurring,
                      "most_comfortable": [r["opening"] for r in comfortable[:3]]}
    return out


def openings_text(p: dict) -> str:
    L = []
    for color in ("white", "black"):
        c = p[color]
        L.append(f"AS {color.upper()} ({c['games']} games)")
        for r in c["repertoire"][:8]:
            acc = f"{r['accuracy']:.0f}" if r["accuracy"] is not None else "-"
            L.append(f"  {r['opening']:34s} {r['games']:3d} games ({r['share'] * 100:4.0f}%)  score {r['score'] * 100:4.0f}% "
                     f"[{r['score_ci'][0] * 100:.0f}-{r['score_ci'][1] * 100:.0f}]  opening accuracy {acc}")
        for ch in c["choices"][:4]:
            mv = ", ".join(f"{m['san']} {m['share'] * 100:.0f}%" for m in ch["moves"])
            L.append(f"  after {ch['after']}: {mv}  (n={ch['n']})")
        for e in c["recurring_opening_mistakes"][:4]:
            L.append(f"  recurring mistake ({e['times']}x): after {e['line'] or 'start'} you played {e['played']}, better {e['better']}")
        if c["most_comfortable"]:
            L.append(f"  highest-accuracy openings (>=5 games): {', '.join(c['most_comfortable'])}")
        L.append("")
    return "\n".join(L)
