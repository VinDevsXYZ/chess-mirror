"""Local dashboard: a tiny JSON API + static single-page app (stdlib only)."""

from __future__ import annotations

import json
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess
import chess.svg

from ..config import Config
from ..db import open_db

STATIC = Path(__file__).parent / "static"
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml"}


def _json_default(o):
    """numpy scalars -> plain JSON numbers/bools; anything else -> str."""
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


class Jobs:
    """Background jobs (import / analysis / training) with a live log, one heavy job at a time."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.counter = 0

    def running(self) -> dict | None:
        return next((j for j in self.jobs.values() if j["status"] == "running"), None)

    def start(self, kind: str, fn, on_done=None) -> dict:
        with self.lock:
            if self.running():
                raise RuntimeError(f"'{self.running()['kind']}' is still running; wait for it to finish")
            self.counter += 1
            job = {"id": str(self.counter), "kind": kind, "status": "running", "log": [],
                   "started": time.time(), "ended": None, "result": None}
            self.jobs[job["id"]] = job

        def log(msg: str) -> None:
            job["log"].append(str(msg))
            del job["log"][:-400]

        def run():
            try:
                job["result"] = fn(log)
                job["status"] = "done"
            except Exception as e:  # surface the error in the UI log
                traceback.print_exc()
                log(f"ERROR: {e}")
                job["status"] = "error"
            finally:
                job["ended"] = time.time()
                if on_done:
                    on_done()
        threading.Thread(target=run, daemon=True).start()
        return job

    def list(self) -> list[dict]:
        return [{**j, "log": j["log"][-60:]} for j in sorted(self.jobs.values(), key=lambda j: -int(j["id"]))][:10]


class Api:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = open_db(cfg)
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[float, object]] = {}
        self.key_locks: dict[str, threading.Lock] = {}
        self.jobs = Jobs()
        from ..play import PlayManager
        self.play = PlayManager(cfg, on_saved=self._learn_after_game)
        self.learn_pending = False

    # ------------------------------------------------------------ learning from played games
    def _learn_after_game(self, session: dict) -> None:
        """A game vs the bot just ended: analyse it and retrain so the Mirror learns from it."""
        try:
            self.start_learning()
        except RuntimeError:  # another job is running: learn when it finishes
            self.learn_pending = True

    def start_learning(self):
        def run(log):
            self._update(log)
            log("The Mirror has learned from your latest games.")
            return True

        def done():
            self._after_update()
            if self.learn_pending:
                self.learn_pending = False
                try:
                    self.start_learning()
                except RuntimeError:
                    self.learn_pending = True
        return self.jobs.start("learning from your games", run, done)

    # ------------------------------------------------------------ setup / jobs
    def settings(self):
        from ..config import config_path, find_stockfish
        from ..uci.install import encroissant_status, launcher_path
        try:
            sf = find_stockfish(self.cfg.engine.stockfish_path)
        except FileNotFoundError:
            sf = None
        counts = self.db.one("""SELECT COUNT(*) games, SUM(player_color IS NOT NULL) mine,
                                SUM(analyzed_at IS NOT NULL) analyzed FROM games""")
        return {"player": vars(self.cfg.player), "analysis": vars(self.cfg.analysis),
                "data_dir": str(self.cfg.data_dir), "config_path": str(config_path()), "stockfish": sf,
                "engine_path": launcher_path(), "encroissant": encroissant_status(),
                "games": counts["games"] or 0, "my_games": counts["mine"] or 0, "analyzed": counts["analyzed"] or 0,
                "models": self.db.scalar("SELECT COUNT(*) FROM models") or 0}

    def save_settings(self, body):
        from ..config import load_config, write_config
        p = self.cfg.player
        p.name = (body.get("name") or p.name).strip()
        p.lichess = (body.get("lichess") or "").strip()
        p.chesscom = (body.get("chesscom") or "").strip()
        if "aliases" in body:
            al = body["aliases"]
            p.aliases = [a.strip() for a in (al.split(",") if isinstance(al, str) else al) if a.strip()]
        if body.get("deep_depth"):
            self.cfg.analysis.deep_depth = int(body["deep_depth"])
        if body.get("rating"):
            p.rating = int(body["rating"])
        write_config(self.cfg)
        self.cfg.player = load_config().player
        self.invalidate()
        return self.settings()

    def job_fetch(self, body):
        from ..config import load_config
        from ..fetch import fetch_chesscom, fetch_lichess
        from ..pgn_import import import_pgn_text
        site = body["site"]
        cfg = load_config()
        user = (body.get("user") or (cfg.player.lichess if site == "lichess" else cfg.player.chesscom)).strip()
        if not user:
            raise ValueError(f"Enter your {site} username first")

        def run(log):
            log(f"Downloading games of {user} from {site}...")
            text = (fetch_lichess(user, max_games=body.get("max") or None) if site == "lichess"
                    else fetch_chesscom(user, months=body.get("months") or None))
            cfg.raw_dir.mkdir(parents=True, exist_ok=True)
            (cfg.raw_dir / f"{site}-{user}.pgn").write_text(text)
            if user.lower() not in cfg.player.all_names():
                cfg.player.aliases.append(user)
            st = import_pgn_text(open_db(cfg), cfg, text, f"{site}:{user}")
            log(f"{st.imported} new games, {st.duplicates} already present, {st.skipped_variant} odds/variant games "
                f"skipped (not normal chess), {st.player_positions} of your moves.")
            if st.imported:
                log("Analysing the new games and updating your Mirror...")
                self._update(log, cfg)
                log("Done.")
            return vars(st)
        return self.jobs.start(f"fetch {site}", run, self._after_update)

    def job_import_paths(self, body):
        from ..config import load_config
        from ..pgn_import import import_pgn_file
        paths = body.get("paths") or []

        def run(log):
            cfg = load_config()
            db = open_db(cfg)
            total = 0
            for pth in paths:
                st = import_pgn_file(db, cfg, pth)
                total += st.imported
                log(f"{Path(pth).name}: {st.imported} imported, {st.duplicates} duplicates, "
                    f"{st.no_player} games without '{cfg.player.name}' (check your names), "
                    f"{st.player_positions} of your moves")
            if total:
                log("Analysing the new games and updating your Mirror...")
                self._update(log, cfg)
                log("Done.")
            return {"imported": total}
        return self.jobs.start("import PGN", run, self._after_update)

    def job_import_text(self, body):
        from ..config import load_config
        from ..pgn_import import import_pgn_text

        def run(log):
            cfg = load_config()
            name = body.get("filename") or "upload.pgn"
            cfg.raw_dir.mkdir(parents=True, exist_ok=True)
            (cfg.raw_dir / f"upload-{int(time.time())}-{Path(name).name}").write_text(body["text"])
            st = import_pgn_text(open_db(cfg), cfg, body["text"], f"pgn:{name}")
            log(f"{name}: {st.imported} imported, {st.duplicates} duplicates, {st.no_player} games without "
                f"'{cfg.player.name}' (check your names), {st.player_positions} of your moves")
            if st.imported:
                log("Analysing the new games and updating your Mirror...")
                self._update(log, cfg)
                log("Done.")
            return vars(st)
        return self.jobs.start("import PGN", run, self._after_update)

    def _update(self, log, cfg=None, depth=None, limit=None, steps=("analyze", "train", "trainer")) -> None:
        """Analyse new games -> train a new model version -> build training positions."""
        from ..analysis.analyze import analyze_games
        from ..coach.training import build_items
        from ..config import load_config
        from ..engine.stockfish import AnalysisSettings
        from ..model.train import train
        cfg = cfg or load_config()
        db = open_db(cfg)
        if "analyze" in steps:
            s = AnalysisSettings(int(depth or cfg.analysis.deep_depth), cfg.analysis.deep_multipv, cfg.analysis.broad_depth)
            analyze_games(db, cfg, settings=s, limit=limit or None, progress=log)
        if "train" in steps:
            n = db.scalar("""SELECT COUNT(*) FROM positions p JOIN games g ON g.id = p.game_id
                             WHERE p.is_player = 1 AND p.wp_loss IS NOT NULL AND g.split = 'train'""") or 0
            if n == 0:
                log("No analysed moves of yours in training games yet - skipping training.")
            else:
                train(db, cfg, notes="trained from the app", progress=log)
        if "trainer" in steps:
            log(f"{build_items(db)} new training positions from your mistakes.")

    def job_pipeline(self, body):
        steps = body.get("steps") or ["analyze", "train", "trainer"]

        def run(log):
            self._update(log, depth=body.get("depth"), limit=body.get("limit"), steps=steps)
            log("Done.")
            return True
        return self.jobs.start("update: " + " + ".join(steps), run, self._after_update)

    def _after_update(self) -> None:
        self.invalidate()
        self.play.reload_model()  # the bot picks up the newly trained model

    def engine_install(self):
        from ..uci.install import add_to_encroissant
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            add_to_encroissant()
        return {"message": buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else "done"}

    def cached(self, key: str, fn, ttl: float = 600, refresh: bool = False):
        def fresh():
            return key in self.cache and time.time() - self.cache[key][0] < ttl
        if not refresh and fresh():
            return self.cache[key][1]
        with self.key_locks.setdefault(key, threading.Lock()):
            if not refresh and fresh():  # another request computed it while we waited
                return self.cache[key][1]
            val = fn()
            self.cache[key] = (time.time(), val)
        return val

    # ------------------------------------------------------------ endpoints
    def profile(self, refresh=False):
        from ..profile.profile import build_profile
        path = self.cfg.reports_dir / "profile.json"

        def compute():
            p = build_profile(self.db, self.cfg)
            path.write_text(json.dumps(p, default=_json_default))
            return json.loads(json.dumps(p, default=_json_default))
        if not refresh and "profile" not in self.cache and path.exists():
            self.cache["profile"] = (path.stat().st_mtime, json.loads(path.read_text()))
        return self.cached("profile", compute, ttl=3600, refresh=refresh)

    def summary(self, refresh=False):
        p = self.profile(refresh)
        from ..coach.training import stats
        models = self.models()
        active = next((m for m in models if m["is_active"]), None)
        return {"player": p["player"], "overview": p["overview"], "strength": p["strength"],
                "rating": {k: v for k, v in p["rating"].items() if k != "results"} | {
                    "results": {k: {kk: vv for kk, vv in v.items() if kk != "trace"} for k, v in p["rating"]["results"].items()}},
                "summary": p["summary"], "style": p["behavioral"]["style"], "risk": p["behavioral"].get("risk"),
                "time": p["behavioral"].get("time"), "by_eval": p["behavioral"].get("by_eval"),
                "development": p["development"], "results": p["results"],
                "mirror": active, "training": stats(self.db)}

    def weaknesses(self, refresh=False):
        from ..profile.weaknesses import compute_weaknesses
        return self.cached("weak", lambda: compute_weaknesses(self.db), refresh=refresh)

    def openings(self, refresh=False):
        from ..profile.openings import opening_profile
        return self.cached("open", lambda: opening_profile(self.db), refresh=refresh)

    def models(self):
        rows = self.db.q("SELECT version, kind, created_at, params, metrics, notes, is_active FROM models ORDER BY id")
        out = []
        for r in rows:
            m = json.loads(r["metrics"] or "{}")
            out.append({"version": r["version"], "created_at": r["created_at"], "is_active": bool(r["is_active"]),
                        "params": json.loads(r["params"] or "{}"), "notes": r["notes"],
                        "metrics": {k: {kk: vv for kk, vv in v.items() if kk in ("n", "top1", "top3", "log_loss",
                                                                                  "perplexity", "ece", "behaviour", "groups")}
                                    if not k.startswith("_") else v for k, v in m.items()}})
        return out

    def elo(self, refresh=False):
        from ..rating.strength import strength_report
        return self.cached("elo", lambda: strength_report(self.db), refresh=refresh)

    def games(self, limit=200):
        rows = self.db.q("""SELECT g.id, g.date, g.white, g.black, g.result, g.player_color, g.opening, g.tc_class,
                               g.platform, g.split, g.analyzed_at IS NOT NULL AS analyzed,
                               AVG(CASE WHEN p.is_player = 1 THEN p.accuracy END) AS accuracy,
                               SUM(CASE WHEN p.is_player = 1 AND p.wp_loss >= 0.15 THEN 1 ELSE 0 END) AS blunders,
                               SUM(CASE WHEN p.is_player = 1 AND p.wp_loss >= 0.10 AND p.wp_loss < 0.15 THEN 1 ELSE 0 END) AS mistakes
                            FROM games g LEFT JOIN positions p ON p.game_id = g.id
                            WHERE g.player_color IS NOT NULL
                            GROUP BY g.id ORDER BY COALESCE(g.utc_datetime, g.date) DESC, g.id DESC LIMIT ?""", (limit,))
        return [dict(r) for r in rows]

    def coach(self, gid: int):
        from ..coach.coach import review_game
        return review_game(self.db, gid)

    def training_next(self, category=None):
        from ..coach.training import CATEGORIES, due_items, stats
        if category and category not in CATEGORIES:  # accept display names too
            category = next((k for k, v in CATEGORIES.items() if v == category), None)
        items = due_items(self.db, 1, category)
        it = items[0] if items else None
        if it:
            b = chess.Board(it["fen"])
            it = {"id": it["id"], "fen": it["fen"], "category": CATEGORIES[it["category"]], "prompt": it["prompt"],
                  "orientation": "white" if b.turn == chess.WHITE else "black", "reps": it["reps"],
                  "lapses": it["lapses"], "game_id": it["game_id"]}
        return {"item": it, "stats": stats(self.db), "categories": CATEGORIES}

    def training_build(self):
        from ..coach.training import build_items
        return {"created": build_items(self.db)}

    def training_answer(self, body):
        from ..coach.training import answer
        return answer(self.db, int(body["id"]), body["uci"], body.get("seconds"))

    def pending(self):
        from ..uci.recorder import pending_list
        return pending_list(self.cfg)

    def finalize(self, body):
        from ..uci.recorder import finalize_pending
        return {"message": finalize_pending(self.db, self.cfg, body["id"], body["result"])}

    def last_decision(self):
        p = self.cfg.state_dir / "last_decision.json"
        return json.loads(p.read_text()) if p.exists() else None

    def elotest_next(self, body):
        from ..rating.elotest import prepare_next
        return {"message": prepare_next(self.db, self.cfg, body.get("opponent"))}

    @staticmethod
    def board_svg(q) -> str:
        fen = q.get("fen", [chess.STARTING_FEN])[0]
        b = chess.Board(fen)
        if q.get("apply"):  # show a just-played move immediately (the reply comes later)
            try:
                mv = chess.Move.from_uci(q["apply"][0])
                if mv in b.legal_moves:
                    b.push(mv)
            except ValueError:
                pass
        flip = q.get("orientation", ["white"])[0] == "black"
        fill = {}
        if q.get("sel"):
            fill[chess.parse_square(q["sel"][0])] = "#f2c14e88"
        for t in q.get("targets", [""])[0].split(","):
            if t:
                fill[chess.parse_square(t)] = "#7fb06988"
        last = None
        if q.get("last"):
            try:
                last = chess.Move.from_uci(q["last"][0])
            except ValueError:
                last = None
        arrows = []
        if q.get("arrow"):
            for a in q["arrow"][0].split(","):
                if len(a) >= 4:
                    arrows.append(chess.svg.Arrow(chess.parse_square(a[:2]), chess.parse_square(a[2:4]), color="#3b82f6cc"))
        check = b.king(b.turn) if b.is_check() else None
        return chess.svg.board(b, orientation=not flip, fill=fill, lastmove=last, arrows=arrows, size=480,
                               coordinates=True, check=check)

    @staticmethod
    def legal(q):
        b = chess.Board(q.get("fen", [chess.STARTING_FEN])[0])
        return [m.uci() for m in b.legal_moves]


def make_handler(api: Api):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (
                body.encode() if isinstance(body, str) else json.dumps(body, default=_json_default).encode())
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            refresh = q.get("refresh", ["0"])[0] == "1"
            try:
                p = u.path
                if p in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), CONTENT_TYPES[".html"])
                if p.startswith("/static/"):
                    f = (STATIC / p[len("/static/"):]).resolve()
                    if STATIC.resolve() in f.parents and f.is_file():
                        return self._send(200, f.read_bytes(), CONTENT_TYPES.get(f.suffix, "application/octet-stream"))
                    return self._send(404, {"error": "not found"})
                routes = {
                    "/api/summary": lambda: api.summary(refresh),
                    "/api/profile": lambda: api.profile(refresh),
                    "/api/weaknesses": lambda: api.weaknesses(refresh),
                    "/api/openings": lambda: api.openings(refresh),
                    "/api/models": api.models,
                    "/api/elo": lambda: api.elo(refresh),
                    "/api/games": lambda: api.games(min(2000, int(q.get("limit", ["200"])[0]))),
                    "/api/training/next": lambda: api.training_next(q.get("category", [None])[0]),
                    "/api/pending": api.pending,
                    "/api/last-decision": api.last_decision,
                    "/api/legal": lambda: api.legal(q),
                    "/api/settings": api.settings,
                    "/api/jobs": api.jobs.list,
                    "/api/play/current": api.play.current,
                    "/api/play/state": lambda: api.play.state(api.play.get(q.get("id", [""])[0])),
                }
                if p in routes:
                    return self._send(200, routes[p]())
                if p.startswith("/api/coach/"):
                    return self._send(200, api.coach(int(p.rsplit("/", 1)[1])))
                if p == "/api/board.svg":
                    return self._send(200, api.board_svg(q), CONTENT_TYPES[".svg"])
                self._send(404, {"error": "not found"})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                routes = {"/api/training/answer": lambda: api.training_answer(body),
                          "/api/training/build": api.training_build,
                          "/api/pending/finalize": lambda: api.finalize(body),
                          "/api/elotest/next": lambda: api.elotest_next(body),
                          "/api/settings": lambda: api.save_settings(body),
                          "/api/import/fetch": lambda: api.job_fetch(body),
                          "/api/import/paths": lambda: api.job_import_paths(body),
                          "/api/import/text": lambda: api.job_import_text(body),
                          "/api/jobs/pipeline": lambda: api.job_pipeline(body),
                          "/api/engine/install": api.engine_install,
                          "/api/play/new": lambda: api.play.new(body),
                          "/api/play/move": lambda: api.play.user_move(body["id"], body["uci"]),
                          "/api/play/resign": lambda: api.play.resign(body["id"]),
                          "/api/play/draw": lambda: api.play.offer_draw(body["id"]),
                          "/api/play/flag": lambda: api.play.flag(body["id"]),
                          "/api/play/abort": lambda: api.play.abort(body["id"])}
                if u.path in routes:
                    return self._send(200, routes[u.path]())
                self._send(404, {"error": "not found"})
            except (RuntimeError, ValueError) as e:  # user-facing problems (busy, missing username...)
                self._send(409, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": str(e)})
    return H


def make_server(cfg: Config, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Api(cfg)))
    httpd.daemon_threads = True
    return httpd


def serve(cfg: Config, port: int = 8765, open_browser: bool = True) -> None:
    httpd = make_server(cfg, port)
    url = f"http://127.0.0.1:{port}/"
    print(f"Dashboard: {url}  (data: {cfg.data_dir})  Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
