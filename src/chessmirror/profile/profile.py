"""The persistent statistical player profile.

Every number is computed from the player's own analysed moves.  Tendencies are
always stated *relative to a baseline* -- the Elo-matched opponents in the same
games (the "population") or the player's own behaviour elsewhere -- together
with sample sizes, credible intervals and an evidence tier.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from ..analysis.features import F_INDEX, FEATURES
from ..analysis.winprob import BLUNDER, INACCURACY, MISTAKE, eval_band
from ..config import Config
from ..db import DB
from ..rating.strength import strength_report
from .evidence import pct, FACT, INSUFFICIENT, bootstrap_mean, p_greater, rate_vs_baseline, tier

OFFENSE_MOTIFS = ["fork", "knight_fork", "pawn_fork", "double_attack", "pin", "skewer", "discovered_attack",
                  "discovered_check", "capture_hanging", "removal_of_defender", "overload", "deflection",
                  "zwischenzug", "sacrifice", "mating_attack", "back_rank", "promotion"]
THREATS = {"knight_fork": "knight-fork threats", "pawn_fork": "pawn-fork threats",
           "hanging_piece": "attacks on loose pieces", "mate_threat": "mate threats"}

POSITION_TYPES = {
    "my isolated pawn": lambda f: f["pawns_me"]["isolated"] > 0,
    "my doubled pawns": lambda f: f["pawns_me"]["doubled"] > 0,
    "my backward pawn": lambda f: f["pawns_me"]["backward"] > 0,
    "I have a passed pawn": lambda f: f["pawns_me"]["passed"] > 0,
    "opponent has a passed pawn": lambda f: f["pawns_op"]["passed"] > 0,
    "I have the bishop pair": lambda f: f["bishop_pair_me"] and not f["bishop_pair_op"],
    "opponent has the bishop pair": lambda f: f["bishop_pair_op"] and not f["bishop_pair_me"],
    "open position (2+ open files)": lambda f: f["open_files"] >= 2,
    "my king exposed": lambda f: f["phase"] != "endgame" and f["shield_me"] <= 1 and f["queens_on"],
    "enemy king exposed": lambda f: f["phase"] != "endgame" and f["shield_op"] <= 1 and f["queens_on"],
    "under king attack": lambda f: f["kz_pressure_on_me"] >= 6,
    "attacking the king": lambda f: f["kz_pressure_on_op"] >= 6,
    "less space": lambda f: f["space_op"] - f["space_me"] >= 2,
    "more space": lambda f: f["space_me"] - f["space_op"] >= 2,
    "opponent has an outpost": lambda f: f["outposts_op"] > 0,
    "I have an outpost": lambda f: f["outposts_me"] > 0,
    "my bad bishop": lambda f: f["bad_bishops_me"] > 0,
    "queenless middlegame": lambda f: f["phase"] != "endgame" and not f["queens_on"],
    "endgame": lambda f: f["phase"] == "endgame",
    "sharp position": lambda f: f.get("sharp", 0) >= 0.5,
    "only-move position": lambda f: bool(f.get("only_move")),
}

STYLE = {
    # trait: [(feature, label)]
    "aggression": [("check", "gives check when checks are available"),
                   ("kz_attack", "increases pressure on the enemy king when possible"),
                   ("pawn_storm", "storms pawns toward the enemy king when possible")],
    "materialism": [("capture", "captures when a capture is available")],
    "sacrifice tendency": [("sac", "plays a material-losing move when one is available")],
    "initiative": [("threat", "makes a direct threat when possible")],
    "simplification": [("trade", "trades pieces when a trade is available"),
                       ("queen_trade", "trades queens when possible"),
                       ("trade_x_winning", "trades pieces when winning")],
    "pawn play": [("pawn_push", "pushes a pawn")],
    "retreating": [("retreat", "retreats a piece")],
    "king safety awareness": [("shield_weaken", "pushes pawns in front of its own castled king"),
                              ("allows_fork", "allows a knight/pawn fork"),
                              ("hangs", "leaves material en prise")],
}


N_STYLE = sum(len(v) for v in STYLE.values())


def _rows(db: DB, is_player: int) -> list:
    extra = "" if is_player else "AND g.opponent_is_engine = 0"
    return db.q(f"""SELECT p.*, g.player_score, g.split, g.utc_datetime, g.date, g.tc_base FROM positions p
                    JOIN games g ON g.id = p.game_id
                    WHERE p.is_player = ? AND p.wp_loss IS NOT NULL AND g.player_color IS NOT NULL {extra}
                    ORDER BY COALESCE(g.utc_datetime, g.date), p.game_id, p.ply""", (is_player,))


def _parse(rows) -> list[dict]:
    out = []
    for r in rows:
        d = dict(r)
        d["f"] = json.loads(d["features"]) if d["features"] else {}
        d["m"] = json.loads(d["motifs"]) if d["motifs"] else {}
        out.append(d)
    return out


def _by_game(rows, fn) -> dict:
    g = defaultdict(list)
    for r in rows:
        v = fn(r)
        if v is not None:
            g[r["game_id"]].append(v)
    return g


def strength_stats(me: list[dict]) -> dict:
    moves = [r for r in me if r["classification"] != "forced"]
    n = len(moves)
    if not n:
        return {}
    acc = bootstrap_mean(_by_game(moves, lambda r: r["accuracy"]))
    acpl = bootstrap_mean(_by_game(moves, lambda r: min(r["cp_loss"], 1000)))
    best = bootstrap_mean(_by_game(moves, lambda r: float(r["played_rank"] == 1)))
    top3 = bootstrap_mean(_by_game(moves, lambda r: float(r["played_rank"] is not None and r["played_rank"] <= 3)))
    out = {"moves": n, "accuracy": acc, "acpl": acpl, "best_move_pct": best, "top3_pct": top3}
    for name, lo in (("blunders", BLUNDER), ("mistakes", MISTAKE), ("inaccuracies", INACCURACY)):
        hi = {"blunders": 9, "mistakes": BLUNDER, "inaccuracies": MISTAKE}[name]
        out[f"{name}_per_100"] = bootstrap_mean(_by_game(moves, lambda r, lo=lo, hi=hi: 100.0 * (lo <= r["wp_loss"] < hi)))
    per_phase = {}
    for ph in ("opening", "middlegame", "endgame"):
        pm = [r for r in moves if r["phase"] == ph]
        if pm:
            per_phase[ph] = {"moves": len(pm),
                             "accuracy": bootstrap_mean(_by_game(pm, lambda r: r["accuracy"])),
                             "mistakes_per_100": bootstrap_mean(_by_game(pm, lambda r: 100.0 * (r["wp_loss"] >= MISTAKE)))}
    out["by_phase"] = per_phase
    return out


def tactical_profile(me: list[dict], pop: list[dict]) -> dict:
    def offense(rows, motif):
        opp = [r for r in rows if motif in r["m"].get("best", []) and r["f"].get("best_gap", 0) >= 0.10
               and 0.03 <= r["wp_before"] <= 0.97]
        return sum(r["wp_loss"] < INACCURACY for r in opp), len(opp)

    def defense(rows, threat):
        opp = [r for r in rows if threat in r["m"].get("threats", [])]
        failed = sum(threat in r["m"].get("threats_unparried", []) and r["wp_loss"] >= INACCURACY for r in opp)
        return len(opp) - failed, len(opp)

    out = {"finding": {}, "defending": {}}
    for mo in OFFENSE_MOTIFS:
        k, n = offense(me, mo)
        kb, nb = offense(pop, mo)
        if n:
            out["finding"][mo] = rate_vs_baseline(k, n, kb, nb, higher_is="better", family=len(OFFENSE_MOTIFS))
    for th, label in THREATS.items():
        k, n = defense(me, th)
        kb, nb = defense(pop, th)
        if n:
            r = rate_vs_baseline(k, n, kb, nb, higher_is="better", family=len(THREATS))
            r["label"] = label
            out["defending"][th] = r
    # defensive resources: only-moves while worse
    res = lambda rows: [r for r in rows if r["f"].get("only_move") and r["wp_before"] <= 0.5]  # noqa: E731
    a, b = res(me), res(pop)
    if a:
        out["defensive_resources"] = rate_vs_baseline(sum(r["wp_loss"] < INACCURACY for r in a), len(a),
                                                      sum(r["wp_loss"] < INACCURACY for r in b), len(b),
                                                      higher_is="better")
    return out


def positional_profile(me: list[dict], pop: list[dict]) -> dict:
    out = {"performance_by_position_type": {}, "positions_reached": {}}
    for name, pred in POSITION_TYPES.items():
        try:
            inside = [r for r in me if r["f"] and pred(r["f"])]
            outside = [r for r in me if r["f"] and not pred(r["f"])]
            pin = [r for r in pop if r["f"] and pred(r["f"])]
        except (KeyError, TypeError):
            continue
        if len(inside) < 5:
            continue
        k_in = sum(r["wp_loss"] >= MISTAKE for r in inside)
        k_out = sum(r["wp_loss"] >= MISTAKE for r in outside)
        k_pop = sum(r["wp_loss"] >= MISTAKE for r in pin)
        vs_self = rate_vs_baseline(k_in, len(inside), k_out, len(outside), higher_is="worse", family=len(POSITION_TYPES))
        vs_pop = rate_vs_baseline(k_in, len(inside), k_pop, len(pin), higher_is="worse", family=len(POSITION_TYPES))
        out["performance_by_position_type"][name] = {"mistake_rate": vs_self["rate"], "n": len(inside),
                                                     "vs_rest_of_my_play": vs_self, "vs_opponents": vs_pop}
    # what kind of positions do I reach (my moves vs opponents' moves in the same games)?
    metrics = {"space": lambda f: f["space_me"] - f["space_op"],
               "mobility": lambda f: f["mobility_me"] - f["mobility_op"],
               "king shield": lambda f: f["shield_me"] - f["shield_op"],
               "pawn weaknesses": lambda f: sum(f["pawns_me"][k] for k in ("isolated", "doubled", "backward"))
               - sum(f["pawns_op"][k] for k in ("isolated", "doubled", "backward")),
               "outposts": lambda f: f["outposts_me"] - f["outposts_op"],
               "king-zone pressure (mine on them minus theirs on me)":
                   lambda f: f["kz_pressure_on_op"] - f["kz_pressure_on_me"]}
    for name, fn in metrics.items():
        vals = _by_game([r for r in me if r["f"] and r["phase"] != "opening"], lambda r, fn=fn: fn(r["f"]))
        mean, lo, hi = bootstrap_mean(vals)
        if np.isnan(mean):
            continue
        t = INSUFFICIENT if lo <= 0 <= hi else ("statistical inference" if len(vals) >= 30 else "weak hypothesis")
        out["positions_reached"][name] = {"mean_difference": mean, "ci": [lo, hi], "games": len(vals), "tier": t}
    return out


def _choice_rates(decisions, feat: str) -> tuple[int, int]:
    j = F_INDEX[feat]
    avail = [d for d in decisions if (d.X[:, j] > 0).any() and not (d.X[:, j] > 0).all()]
    return int(sum(bool(d.X[d.y, j] > 0) for d in avail)), len(avail)


def behavioral_profile(me: list[dict], pop: list[dict], dec_me, dec_pop, model) -> dict:
    out = {"style": {}, "by_eval": {}, "time": {}, "model_weights": {}}
    for trait, items in STYLE.items():
        rows = []
        for feat, label in items:
            k, n = _choice_rates(dec_me, feat)
            kb, nb = _choice_rates(dec_pop, feat)
            if n:
                r = rate_vs_baseline(k, n, kb, nb, higher_is="neutral", family=N_STYLE)
                r.update(feature=feat, label=label)
                rows.append(r)
        out["style"][trait] = rows
    # sharpness of positions I create (opponent's next decision) vs positions opponents create for me
    sharp_me = defaultdict(list)
    for r in pop:
        sharp_me[r["game_id"]].append(r["f"].get("sharp", 0))
    sharp_op = _by_game(me, lambda r: r["f"].get("sharp", 0))
    a, b = bootstrap_mean(sharp_me), bootstrap_mean(sharp_op)
    out["risk"] = {"sharpness_i_create": a, "sharpness_created_for_me": b,
                   "note": "share of legal moves that are mistakes in the position handed to the other side"}
    for band in ("winning", "equal", "losing"):
        m = [r for r in me if eval_band(r["wp_before"]) == band]
        p = [r for r in pop if eval_band(r["wp_before"]) == band]
        if m:
            out["by_eval"][band] = rate_vs_baseline(sum(r["wp_loss"] >= MISTAKE for r in m), len(m),
                                                    sum(r["wp_loss"] >= MISTAKE for r in p), len(p), "worse", family=3)
    tp = [r for r in me if r["f"].get("time_pressure")]
    ntp = [r for r in me if not r["f"].get("time_pressure") and r["clock_before"] is not None]
    if tp and ntp:
        out["time"]["mistakes_under_time_pressure"] = rate_vs_baseline(
            sum(r["wp_loss"] >= MISTAKE for r in tp), len(tp), sum(r["wp_loss"] >= MISTAKE for r in ntp), len(ntp),
            "worse")
    ts = [r for r in me if r["time_spent"] is not None]
    if ts:
        by_phase = {}
        for ph in ("opening", "middlegame", "endgame"):
            v = [r["time_spent"] for r in ts if r["phase"] == ph]
            if v:
                by_phase[ph] = {"median_s": float(np.median(v)), "mean_s": float(np.mean(v)), "n": len(v)}
        crit = [r["time_spent"] for r in ts if r["f"].get("only_move")]
        rest = [r["time_spent"] for r in ts if not r["f"].get("only_move")]
        games_tp = len({r["game_id"] for r in tp})
        out["time"].update({
            "seconds_per_move_by_phase": by_phase,
            "critical_vs_normal_think_ratio": (float(np.median(crit)) / max(0.1, float(np.median(rest))))
            if crit and rest else None,
            "games_reaching_time_pressure": games_tp,
            "games_with_clock_data": len({r["game_id"] for r in ts}),
            "fast_moves_in_critical_positions": {"k": sum(1 for x in crit if x < 5), "n": len(crit), "tier": FACT},
        })
    if model is not None:
        b, se, pb = model.personal.beta, model.personal.se, model.population.beta
        for j, name in enumerate(FEATURES):
            z = (b[j] - pb[j]) / se[j] if se is not None and se[j] > 0 else 0.0
            t = ("statistical inference" if abs(z) >= 3.0 else "weak hypothesis" if abs(z) >= 2.0 else INSUFFICIENT)
            out["model_weights"][name] = {"personal": float(b[j]), "population": float(pb[j]), "z": float(z),
                                          "tier": t}
    return out


def conversion_and_defense(db: DB) -> dict:
    rows = db.q("""SELECT g.id, g.player_score, MAX(p.wp_before) mx, MIN(p.wp_before) mn FROM games g
                   JOIN positions p ON p.game_id = g.id
                   WHERE p.is_player = 1 AND p.wp_before IS NOT NULL AND p.ply >= 16 AND g.player_score IS NOT NULL
                   GROUP BY g.id""")
    won = [r for r in rows if r["mx"] >= 0.85]
    lost = [r for r in rows if r["mn"] <= 0.15]
    return {"conversion": {"k": sum(r["player_score"] == 1 for r in won), "n": len(won),
                           "label": "games won after reaching >=85% win probability", "tier": FACT},
            "saves": {"k": sum(r["player_score"] >= 0.5 for r in lost), "n": len(lost),
                      "label": "games drawn or won after falling to <=15% win probability", "tier": FACT}}


def development(me: list[dict]) -> dict:
    """Recent vs earlier games (chronological), per key metric."""
    games = list(dict.fromkeys(r["game_id"] for r in me))
    if len(games) < 20:
        return {"note": f"needs >= 20 analysed games (have {len(games)})"}
    cut = set(games[int(len(games) * 0.7):])
    recent = [r for r in me if r["game_id"] in cut]
    early = [r for r in me if r["game_id"] not in cut]
    out = {"recent_games": len(cut), "earlier_games": len(games) - len(cut), "metrics": {}}
    tests = {"mistake rate": lambda r: r["wp_loss"] >= MISTAKE, "blunder rate": lambda r: r["wp_loss"] >= BLUNDER,
             "engine top move rate": lambda r: r["played_rank"] == 1}
    for name, fn in tests.items():
        k1, n1 = sum(map(fn, recent)), len(recent)
        k0, n0 = sum(map(fn, early)), len(early)
        p = p_greater(k1, n1, k0, n0)
        worse_if_higher = name != "engine top move rate"
        improving = (p < 0.5) == worse_if_higher
        conf = max(p, 1 - p)
        out["metrics"][name] = {"recent": k1 / max(1, n1), "earlier": k0 / max(1, n0), "confidence": conf,
                                "trend": "improving" if improving else "declining", "tier": tier(conf, n1, family=len(tests))}
    return out


def summarize(prof: dict) -> dict:
    strengths, weaknesses = [], []

    def add(label, r):
        if not r or r.get("tier") in (None, INSUFFICIENT):
            return
        item = {"label": label, "tier": r["tier"], "confidence": r["confidence"], "n": r["n"]}
        (strengths if r["verdict"] == "better" else weaknesses).append(item)

    t = prof.get("tactical", {})
    for mo, r in t.get("finding", {}).items():
        add(f"finding {mo.replace('_', ' ')} tactics", r)
    for th, r in t.get("defending", {}).items():
        add(f"defending against {r['label']}", r)
    add("finding defensive resources", t.get("defensive_resources"))
    for name, r in prof.get("positional", {}).get("performance_by_position_type", {}).items():
        add(f"play in positions with: {name}", r["vs_opponents"])
    for band, r in prof.get("behavioral", {}).get("by_eval", {}).items():
        add(f"accuracy when {band}", r)
    add("accuracy under time pressure", prof.get("behavioral", {}).get("time", {}).get("mistakes_under_time_pressure"))
    key = lambda x: (x["tier"] != "statistical inference", -x["confidence"])  # noqa: E731
    return {"strengths": sorted(strengths, key=key), "weaknesses": sorted(weaknesses, key=key)}


def build_profile(db: DB, cfg: Config) -> dict:
    from ..model.dataset import load_decisions
    from ..model.train import load_active
    me, pop = _parse(_rows(db, 1)), _parse(_rows(db, 0))
    model = None
    try:
        model = load_active(db, cfg)
    except Exception:
        pass
    dec_me = load_decisions(db, "p.is_player = 1 AND g.player_color IS NOT NULL", workers=cfg.analysis.workers)
    dec_pop = load_decisions(db, "p.is_player = 0 AND g.player_color IS NOT NULL AND g.opponent_is_engine = 0",
                             workers=cfg.analysis.workers)
    prof = {
        "player": cfg.player.name,
        "overview": {"games": len({r["game_id"] for r in me}), "positions": len(me),
                     "population_positions": len(pop)},
        "strength": strength_stats(me),
        "rating": strength_report(db),
        "tactical": tactical_profile(me, pop),
        "positional": positional_profile(me, pop),
        "behavioral": behavioral_profile(me, pop, dec_me, dec_pop, model),
        "results": conversion_and_defense(db),
        "development": development(me),
        "model": ({"version": model.version, "meta": model.meta} if model else None),
    }
    prof["summary"] = summarize(prof)
    return prof


# ------------------------------------------------------------------ text
def _ci(t) -> str:
    return f"{t[0]:.1f} (95% CI {t[1]:.1f}-{t[2]:.1f})"


def profile_text(p: dict) -> str:
    L = [f"PLAYER PROFILE -- {p['player']}",
         f"Games analysed: {p['overview']['games']}   your moves: {p['overview']['positions']}   "
         f"opponent moves (baseline): {p['overview']['population_positions']}"]
    s = p["strength"]
    if s:
        L += ["", "STRENGTH",
              f"  Accuracy {_ci(s['accuracy'])}   ACPL {_ci(s['acpl'])}",
              f"  Engine top move {s['best_move_pct'][0] * 100:.1f}%   top-3 {s['top3_pct'][0] * 100:.1f}%",
              f"  Per 100 moves: blunders {s['blunders_per_100'][0]:.1f}, mistakes {s['mistakes_per_100'][0]:.1f}, "
              f"inaccuracies {s['inaccuracies_per_100'][0]:.1f}"]
        for ph, v in s.get("by_phase", {}).items():
            L.append(f"  {ph:10s} accuracy {_ci(v['accuracy'])}, mistakes/100 {v['mistakes_per_100'][0]:.1f} ({v['moves']} moves)")
    rep = p["rating"]
    prim = rep.get("primary_pool")
    if prim and prim in rep["results"]:
        c = rep["results"][prim]["current"]
        L.append(f"  Estimated strength [{prim}]: {c['mean']}  (95% interval {c['lo']}-{c['hi']})  -- `mirror elo` for details")
    L += ["", "STRENGTHS (vs similar-rated opponents)"]
    for x in p["summary"]["strengths"][:8]:
        L.append(f"  + {x['label']}  [{x['tier']}, confidence {pct(x['confidence'])}, n={x['n']}]")
    if not p["summary"]["strengths"]:
        L.append("  (none with sufficient evidence yet)")
    L += ["", "WEAKNESSES (vs similar-rated opponents)"]
    for x in p["summary"]["weaknesses"][:8]:
        L.append(f"  - {x['label']}  [{x['tier']}, confidence {pct(x['confidence'])}, n={x['n']}]")
    if not p["summary"]["weaknesses"]:
        L.append("  (none with sufficient evidence yet)")
    L += ["", "STYLE (choice rate when the option exists; you vs opponents)"]
    for trait, rows in p["behavioral"]["style"].items():
        for r in rows:
            mark = "" if r["tier"] == INSUFFICIENT else f"  -> {r['direction']} than baseline [{r['tier']}]"
            L.append(f"  {trait:22s} {r['label']}: {r['rate'] * 100:.0f}% vs {r['baseline_rate'] * 100:.0f}% (n={r['n']}){mark}")
    rk = p["behavioral"].get("risk")
    if rk and not np.isnan(rk["sharpness_i_create"][0]):
        L.append(f"  risk: sharpness of positions you hand over {rk['sharpness_i_create'][0]:.2f} vs "
                 f"handed to you {rk['sharpness_created_for_me'][0]:.2f}")
    t = p["behavioral"].get("time", {})
    if t.get("seconds_per_move_by_phase"):
        L += ["", "TIME"]
        L.append("  median seconds/move: " + ", ".join(f"{k} {v['median_s']:.1f}s" for k, v in t["seconds_per_move_by_phase"].items()))
        if t.get("critical_vs_normal_think_ratio"):
            L.append(f"  you think {t['critical_vs_normal_think_ratio']:.1f}x longer in only-move positions than elsewhere")
        L.append(f"  games reaching time pressure: {t['games_reaching_time_pressure']} of {t['games_with_clock_data']}")
    r = p["results"]
    L += ["", "CONVERSION & DEFENCE (observed facts)",
          f"  {r['conversion']['label']}: {r['conversion']['k']}/{r['conversion']['n']}",
          f"  {r['saves']['label']}: {r['saves']['k']}/{r['saves']['n']}"]
    dv = p["development"]
    L += ["", "RECENT DEVELOPMENT"]
    if "metrics" in dv:
        for name, m in dv["metrics"].items():
            L.append(f"  {name}: recent {m['recent'] * 100:.1f}% vs earlier {m['earlier'] * 100:.1f}% -> "
                     f"{m['trend']} [{m['tier']}, confidence {pct(m['confidence'])}]")
    else:
        L.append(f"  {dv['note']}")
    if p.get("model"):
        L += ["", f"MIRROR MODEL: {p['model']['version']} -- {p['model']['meta'].get('sigma_note', '')}"]
    return "\n".join(L)
