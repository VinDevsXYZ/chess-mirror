"""``mirror`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .db import open_db


def _ctx():
    cfg = load_config()
    return cfg, open_db(cfg)


# ------------------------------------------------------------------ data
def cmd_import(a):
    from .pgn_import import import_pgn_file
    cfg, db = _ctx()
    for f in a.files:
        st = import_pgn_file(db, cfg, f, force_color=a.color)
        print(f"{f}: {st.imported} imported, {st.duplicates} duplicates, {st.no_player} without "
              f"'{cfg.player.name}' (check [player] aliases), {st.skipped_variant} odds/variant games skipped, "
              f"{st.player_positions} of your moves")


def cmd_fetch(a):
    from .fetch import fetch_chesscom, fetch_lichess
    from .pgn_import import import_pgn_text
    cfg, db = _ctx()
    user = a.user or (cfg.player.lichess if a.site == "lichess" else cfg.player.chesscom)
    if not user:
        sys.exit(f"No username: pass --user or set [player] {a.site} in mirror.toml")
    print(f"Downloading games of {user} from {a.site}...")
    text = fetch_lichess(user, max_games=a.max) if a.site == "lichess" else fetch_chesscom(user, months=a.months)
    raw = cfg.raw_dir / f"{a.site}-{user}.pgn"
    raw.write_text(text)
    if user.lower() not in cfg.player.all_names():
        cfg.player.aliases.append(user)
    st = import_pgn_text(db, cfg, text, f"{a.site}:{user}")
    print(f"{st.imported} new games, {st.duplicates} already present, {st.skipped_variant} odds/variant games "
          f"skipped, {st.player_positions} of your moves")


def cmd_analyze(a):
    from .analysis.analyze import analyze_games
    from .engine.stockfish import AnalysisSettings
    cfg, db = _ctx()
    s = AnalysisSettings(a.depth or cfg.analysis.deep_depth, cfg.analysis.deep_multipv,
                         a.broad_depth or cfg.analysis.broad_depth)
    analyze_games(db, cfg, settings=s, workers=a.workers, reanalyze=a.reanalyze, limit=a.limit)


def cmd_status(a):
    cfg, db = _ctx()
    g = db.one("""SELECT COUNT(*) n, SUM(player_color IS NOT NULL) mine, SUM(analyzed_at IS NOT NULL) analyzed,
                  SUM(split='test') test FROM games""")
    p = db.one("SELECT COUNT(*) n, SUM(is_player) mine, SUM(is_player AND wp_loss IS NOT NULL) mine_an FROM positions")
    print(f"data dir: {cfg.data_dir}")
    print(f"games: {g['n']} ({g['mine'] or 0} with {cfg.player.name}, {g['analyzed'] or 0} analysed, "
          f"{g['test'] or 0} held out for testing)")
    print(f"positions: {p['n']} ({p['mine'] or 0} your moves, {p['mine_an'] or 0} analysed)")
    from .model.train import active_version
    av = active_version(db)
    print(f"active model: {av['version'] if av else 'none'}")


# ------------------------------------------------------------------ model
def cmd_train(a):
    from .model.train import train
    cfg, db = _ctx()
    train(db, cfg, notes=a.notes or "", activate=not a.no_activate)


def cmd_models(a):
    cfg, db = _ctx()
    rows = db.q("SELECT version, created_at, is_active, metrics, params FROM models ORDER BY id")
    if not rows:
        print("No models yet: run `mirror train`.")
    for r in rows:
        m = json.loads(r["metrics"] or "{}")
        mirror = next((v for k, v in m.items() if k.startswith("mirror")), None)
        pop = m.get("population (Elo-matched opponents)")
        line = f"{'*' if r['is_active'] else ' '} {r['version']:12s} {r['created_at']}"
        if mirror:
            line += f"  top1={mirror['top1']:.3f} top3={mirror['top3']:.3f} logloss={mirror['log_loss']:.3f}"
        if pop:
            line += f"  | population top1={pop['top1']:.3f} logloss={pop['log_loss']:.3f}"
        print(line)


def cmd_compare(a):
    from .model.train import compare
    cfg, db = _ctx()
    r = compare(db, cfg, a.a, a.b)
    if "error" in r:
        sys.exit(r["error"])
    print(f"{r['games']} games / {r['moves']} moves held out from both versions")
    for v in (a.a, a.b):
        print(f"  {v:12s} top1={r[v]['top1']:.3f} top3={r[v]['top3']:.3f} logloss={r[v]['log_loss']:.3f}")
    s = r["logloss_diff_a_minus_b"]
    print(f"  logloss {a.a} - {a.b} = {s['mean_diff']:+.4f} (95% CI {s['ci95'][0]:+.4f}..{s['ci95'][1]:+.4f}); "
          f"P({a.a} better) = {s['p_a_better']:.2f}")


def cmd_activate(a):
    cfg, db = _ctx()
    with db.tx() as c:
        c.execute("UPDATE models SET is_active = 0")
        n = c.execute("UPDATE models SET is_active = 1 WHERE version = ?", (a.version,)).rowcount
    print("activated" if n else "unknown version")


def cmd_explain(a):
    from .uci.predictor import LivePredictor
    cfg, db = _ctx()
    lp = LivePredictor(cfg, db)
    import chess
    board = chess.Board(a.fen)
    res = lp.decide(board, explain=True, temperature=0)
    print(lp.format_explanation(res))
    lp.close()


# ------------------------------------------------------------------ reports
def cmd_profile(a):
    from .profile.profile import build_profile, profile_text
    cfg, db = _ctx()
    prof = build_profile(db, cfg)
    out = cfg.reports_dir / "profile.json"
    from .dashboard.server import _json_default
    out.write_text(json.dumps(prof, indent=1, default=_json_default))
    print(profile_text(prof))
    print(f"\n(full profile JSON: {out})")


def cmd_weaknesses(a):
    from .profile.weaknesses import compute_weaknesses, weakness_text
    cfg, db = _ctx()
    ws = compute_weaknesses(db, window=a.window)
    print(weakness_text(ws))


def cmd_openings(a):
    from .profile.openings import opening_profile, openings_text
    cfg, db = _ctx()
    print(openings_text(opening_profile(db)))


def cmd_elo(a):
    from .rating.strength import strength_report
    from .rating.elotest import elo_text
    cfg, db = _ctx()
    print(elo_text(strength_report(db), db))


def cmd_elotest(a):
    from .rating import elotest
    cfg, db = _ctx()
    if a.action == "next":
        print(elotest.prepare_next(db, cfg, a.opponent))
    elif a.action == "pending":
        print(elotest.pending_text(db))
    elif a.action == "result":
        print(elotest.record_result(db, cfg, a.game, a.result))


def cmd_coach(a):
    from .coach.coach import coach_game, latest_game_id
    cfg, db = _ctx()
    gid = a.game or latest_game_id(db)
    if gid is None:
        sys.exit("No analysed games.")
    md, path = coach_game(db, cfg, gid)
    print(md)
    print(f"\n(saved: {path})")


def cmd_trainer(a):
    from .coach import training
    cfg, db = _ctx()
    if a.action == "build":
        n = training.build_items(db)
        print(f"{n} new training positions created from your mistakes.")
    elif a.action == "due":
        print(training.due_text(db))
    elif a.action == "run":
        training.run_cli(db, limit=a.limit)
    elif a.action == "export":
        path = training.export_pgn(db, cfg)
        print(f"Exported due positions to {path} (open it in En Croissant).")


def cmd_games(a):
    from .uci import recorder
    cfg, db = _ctx()
    if a.action == "pending":
        print(recorder.pending_text(cfg))
    elif a.action == "finalize":
        print(recorder.finalize_pending(db, cfg, a.id, a.result))


def cmd_dashboard(a):
    from .dashboard.server import serve
    cfg, db = _ctx()
    serve(cfg, port=a.port, open_browser=not a.no_browser)


def cmd_engine(a):
    from .uci import install
    cfg, _ = _ctx()
    if a.action == "path":
        print(install.launcher_path())
    elif a.action == "check":
        install.self_check(cfg)
    elif a.action == "install-encroissant":
        install.add_to_encroissant(dry_run=a.dry_run)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mirror", description="Personal Chess Mirror")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("import", help="import PGN files")
    s.add_argument("files", nargs="+")
    s.add_argument("--color", choices=["white", "black"], help="force your colour (OTB games with other names)")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("fetch", help="download your games from lichess / chess.com")
    s.add_argument("site", choices=["lichess", "chesscom"])
    s.add_argument("--user")
    s.add_argument("--max", type=int, help="lichess: max games")
    s.add_argument("--months", type=int, help="chess.com: last N monthly archives")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("analyze", help="run Stockfish over imported games")
    s.add_argument("--depth", type=int)
    s.add_argument("--broad-depth", type=int)
    s.add_argument("--workers", type=int)
    s.add_argument("--reanalyze", action="store_true", help="re-analyse every game (e.g. at a new depth)")
    s.add_argument("--limit", type=int, help="only the N most recent unanalysed games")
    s.set_defaults(func=cmd_analyze)

    sub.add_parser("status", help="data summary").set_defaults(func=cmd_status)

    s = sub.add_parser("train", help="fit a new Mirror model version and evaluate it on held-out games")
    s.add_argument("--notes")
    s.add_argument("--no-activate", action="store_true")
    s.set_defaults(func=cmd_train)
    sub.add_parser("models", help="list model versions").set_defaults(func=cmd_models)
    s = sub.add_parser("compare", help="compare two model versions on common held-out games")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_compare)
    s = sub.add_parser("activate", help="make a model version the one the engine uses")
    s.add_argument("version")
    s.set_defaults(func=cmd_activate)
    s = sub.add_parser("explain", help="what would I play here, and why?")
    s.add_argument("fen")
    s.set_defaults(func=cmd_explain)

    sub.add_parser("profile", help="player profile report").set_defaults(func=cmd_profile)
    s = sub.add_parser("weaknesses", help="recurring weaknesses")
    s.add_argument("--window", type=int, default=40, help="recent-games window")
    s.set_defaults(func=cmd_weaknesses)
    sub.add_parser("openings", help="opening fingerprint").set_defaults(func=cmd_openings)
    sub.add_parser("elo", help="strength estimates with uncertainty").set_defaults(func=cmd_elo)
    s = sub.add_parser("elotest", help="Elo test mode helpers")
    s.add_argument("action", choices=["next", "pending", "result"])
    s.add_argument("--opponent", type=int, help="next: force opponent rating")
    s.add_argument("--game", type=int)
    s.add_argument("--result", choices=["1-0", "0-1", "1/2-1/2", "win", "loss", "draw"])
    s.set_defaults(func=cmd_elotest)

    s = sub.add_parser("coach", help="teach me from a game")
    s.add_argument("game", nargs="?", type=int)
    s.set_defaults(func=cmd_coach)
    s = sub.add_parser("trainer", help="personalised training from your own mistakes")
    s.add_argument("action", choices=["build", "due", "run", "export"])
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_trainer)
    s = sub.add_parser("games", help="games recorded by the engine")
    s.add_argument("action", choices=["pending", "finalize"])
    s.add_argument("--id")
    s.add_argument("--result", choices=["1-0", "0-1", "1/2-1/2"])
    s.set_defaults(func=cmd_games)

    s = sub.add_parser("dashboard", help="open the local dashboard")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_dashboard)
    s = sub.add_parser("engine", help="UCI engine helpers")
    s.add_argument("action", choices=["path", "check", "install-encroissant"])
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_engine)

    a = ap.parse_args(argv)
    try:
        a.func(a)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        sys.exit(f"mirror: {e}")


if __name__ == "__main__":
    main()
