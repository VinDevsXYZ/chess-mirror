"""Recurring weaknesses across games.

Each weakness is a (opportunity, failure) definition evaluated on every analysed
move.  For each we report
  * observed fact   -- counts, overall and in the most recent N games
  * comparison      -- the same rate for Elo-matched opponents in the same games
  * trend           -- recent window vs everything before it
  * evidence tier   -- statistical inference / weak hypothesis / insufficient
Snapshots are stored in the ``weaknesses`` table so changes can be tracked.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Callable

from ..analysis.winprob import BLUNDER, INACCURACY, MISTAKE
from ..db import DB
from .evidence import pct, INSUFFICIENT, p_greater, rate_vs_baseline, tier
from .profile import _parse, _rows

Pred = Callable[[dict], bool]


def _missed(motif: str) -> tuple[Pred, Pred]:
    return (lambda r: motif in r["m"].get("best", []) and r["f"].get("best_gap", 0) >= 0.10
            and 0.03 <= r["wp_before"] <= 0.97,
            lambda r: r["wp_loss"] >= INACCURACY)


def _unparried(threat: str) -> tuple[Pred, Pred]:
    return (lambda r: threat in r["m"].get("threats", []),
            lambda r: threat in r["m"].get("threats_unparried", []) and r["wp_loss"] >= INACCURACY)


def _allowed(motif: str) -> tuple[Pred, Pred]:
    return (lambda r: r["classification"] != "forced",
            lambda r: motif in r["m"].get("allowed", []) and r["wp_loss"] >= MISTAKE)


DEFINITIONS: dict[str, dict] = {
    "missed_knight_fork": {"cat": "tactical oversight", "label": "missing knight forks", "unit": "opportunities", "def": _missed("knight_fork")},
    "missed_fork": {"cat": "tactical oversight", "label": "missing forks (any piece)", "unit": "opportunities", "def": _missed("fork")},
    "missed_pin": {"cat": "tactical oversight", "label": "missing pins", "unit": "opportunities", "def": _missed("pin")},
    "missed_skewer": {"cat": "tactical oversight", "label": "missing skewers", "unit": "opportunities", "def": _missed("skewer")},
    "missed_discovered": {"cat": "tactical oversight", "label": "missing discovered attacks", "unit": "opportunities", "def": _missed("discovered_attack")},
    "missed_free_material": {"cat": "missed opportunity", "label": "not taking loose/hanging material", "unit": "opportunities", "def": _missed("capture_hanging")},
    "missed_mate": {"cat": "missed opportunity", "label": "missing mating attacks", "unit": "opportunities", "def": _missed("mating_attack")},
    "missed_removal_of_defender": {"cat": "tactical oversight", "label": "missing removal-of-the-defender tactics", "unit": "opportunities", "def": _missed("removal_of_defender")},
    "missed_sacrifice": {"cat": "missed opportunity", "label": "missing winning sacrifices", "unit": "opportunities", "def": _missed("sacrifice")},
    "unparried_knight_fork": {"cat": "tactical oversight", "label": "not stopping knight-fork threats", "unit": "threats", "def": _unparried("knight_fork")},
    "unparried_hanging": {"cat": "tactical oversight", "label": "leaving attacked pieces loose", "unit": "threats", "def": _unparried("hanging_piece")},
    "unparried_mate": {"cat": "king safety", "label": "not addressing mate threats", "unit": "threats", "def": _unparried("mate_threat")},
    "allowed_fork": {"cat": "tactical oversight", "label": "moves that allow a fork", "unit": "moves", "def": _allowed("fork")},
    "allowed_pin_skewer": {"cat": "tactical oversight", "label": "moves that walk into pins/skewers", "unit": "moves",
                           "def": (lambda r: r["classification"] != "forced",
                                   lambda r: bool({"pin", "skewer"} & set(r["m"].get("allowed", []))) and r["wp_loss"] >= MISTAKE)},
    "allowed_mating_attack": {"cat": "king safety", "label": "moves that allow a mating attack", "unit": "moves", "def": _allowed("mating_attack")},
    "time_trouble_blunders": {"cat": "time management", "label": "blunders under time pressure", "unit": "moves in time pressure",
                              "def": (lambda r: bool(r["f"].get("time_pressure")), lambda r: r["wp_loss"] >= BLUNDER)},
    "rushed_critical": {"cat": "time management", "label": "rushing critical (only-move) positions", "unit": "critical positions",
                        "def": (lambda r: bool(r["f"].get("only_move")) and r["time_spent"] is not None,
                                lambda r: r["time_spent"] < 5 and r["wp_loss"] >= MISTAKE)},
    "conversion": {"cat": "missed opportunity", "label": "errors in clearly winning positions", "unit": "winning positions",
                   "def": (lambda r: r["wp_before"] >= 0.85, lambda r: r["wp_loss"] >= MISTAKE)},
    "defensive_resources": {"cat": "calculation error", "label": "missing only-move defences", "unit": "critical defensive positions",
                            "def": (lambda r: bool(r["f"].get("only_move")) and r["wp_before"] <= 0.5,
                                    lambda r: r["wp_loss"] >= INACCURACY)},
    "endgame_technique": {"cat": "endgame technique", "label": "endgame mistakes", "unit": "endgame moves",
                          "def": (lambda r: r["phase"] == "endgame", lambda r: r["wp_loss"] >= MISTAKE)},
    "opening_mistakes": {"cat": "opening knowledge", "label": "opening mistakes", "unit": "opening moves",
                         "def": (lambda r: r["phase"] == "opening", lambda r: r["wp_loss"] >= MISTAKE)},
    "king_exposure": {"cat": "king safety", "label": "mistakes while under king attack", "unit": "positions under attack",
                      "def": (lambda r: r["f"].get("kz_pressure_on_me", 0) >= 6, lambda r: r["wp_loss"] >= MISTAKE)},
}


def _count(rows, opp: Pred, fail: Pred) -> tuple[int, int]:
    o = [r for r in rows if opp(r)]
    return sum(1 for r in o if fail(r)), len(o)


def compute_weaknesses(db: DB, window: int = 40, store: bool = True) -> list[dict]:
    me, pop = _parse(_rows(db, 1)), _parse(_rows(db, 0))
    games = list(dict.fromkeys(r["game_id"] for r in me))
    recent_ids = set(games[-window:])
    recent = [r for r in me if r["game_id"] in recent_ids]
    earlier = [r for r in me if r["game_id"] not in recent_ids]
    out = []
    for key, d in DEFINITIONS.items():
        opp, fail = d["def"]
        k, n = _count(me, opp, fail)
        if n == 0:
            continue
        kb, nb = _count(pop, opp, fail)
        cmp_ = rate_vs_baseline(k, n, kb, nb, higher_is="worse", family=len(DEFINITIONS))
        kr, nr = _count(recent, opp, fail)
        ke, ne = _count(earlier, opp, fail)
        trend = None
        if nr >= 5 and ne >= 5:
            p = p_greater(kr, nr, ke, ne)
            conf = max(p, 1 - p)
            trend = {"direction": "worsening" if p > 0.5 else "improving", "confidence": conf,
                     "tier": tier(conf, nr, min_n=20), "recent_rate": kr / nr, "earlier_rate": ke / ne}
        w = {"key": key, "category": d["cat"], "label": d["label"], "unit": d["unit"],
             "overall": cmp_, "recent": {"games": len(recent_ids), "failures": kr, "opportunities": nr},
             "trend": trend}
        w["is_weakness"] = cmp_["tier"] != INSUFFICIENT and cmp_["verdict"] == "worse"
        w["text"] = _text(w)
        out.append(w)
    order = {"statistical inference": 0, "weak hypothesis": 1, INSUFFICIENT: 2}
    out.sort(key=lambda w: (not w["is_weakness"], order[w["overall"]["tier"]], -w["overall"]["confidence"]))
    if store:
        now = dt.datetime.now().isoformat(timespec="seconds")
        with db.tx() as conn:
            for w in out:
                conn.execute("INSERT OR REPLACE INTO weaknesses(key, category, label, data, updated_at) VALUES (?,?,?,?,?)",
                             (w["key"], w["category"], w["label"], json.dumps(w), now))
    return out


def _text(w: dict) -> str:
    o, r = w["overall"], w["recent"]
    fact = (f"In your last {r['games']} games: {r['failures']} of {r['opportunities']} {w['unit']} "
            f"({w['label']}). Overall: {o['k']}/{o['n']} ({o['rate'] * 100:.0f}%) vs similar-rated opponents "
            f"{o['baseline_rate'] * 100:.0f}% (n={o['baseline_n']}).")
    if w["is_weakness"]:
        head = f"{w['label'].capitalize()} is a {'recurring weakness' if o['tier'] == 'statistical inference' else 'possible weakness'}"
        head += f" (confidence {pct(o['confidence'])}, {o['tier']})."
    elif o["tier"] != INSUFFICIENT:
        head = f"{w['label'].capitalize()}: better than similar-rated opponents ({o['tier']})."
    else:
        head = f"{w['label'].capitalize()}: no clear difference from similar-rated opponents ({INSUFFICIENT})."
    t = w.get("trend")
    tr = ""
    if t and t["tier"] != INSUFFICIENT:
        tr = f" Trend: {t['direction']} ({t['recent_rate'] * 100:.0f}% recently vs {t['earlier_rate'] * 100:.0f}% before; {t['tier']})."
    return f"{head} {fact}{tr}"


def weakness_text(ws: list[dict]) -> str:
    L = ["RECURRING WEAKNESSES"]
    shown = [w for w in ws if w["is_weakness"]]
    for w in shown:
        L.append(f"  [{w['category']}] {w['text']}")
    if not shown:
        L.append("  None with sufficient evidence yet.")
    L.append("\nOTHER TRACKED PATTERNS")
    for w in ws:
        if not w["is_weakness"]:
            L.append(f"  [{w['category']}] {w['text']}")
    return "\n".join(L)
