"""Helpers for connecting the engine to En Croissant."""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from ..config import ENCROISSANT_DIR, FROZEN, SOURCE_ROOT, Config

ENGINES_JSON = ENCROISSANT_DIR / "engines" / "engines.json"


def launcher_path() -> str:
    """The file En Croissant should run: inside the app bundle, or bin/ in the checkout."""
    if FROZEN:
        return str(Path(sys.executable).parent / "mirror-engine")
    return str(SOURCE_ROOT / "bin" / "mirror-engine")


def encroissant_status() -> dict:
    installed_app = Path("/Applications/en-croissant.app").exists()
    registered = False
    if ENGINES_JSON.exists():
        try:
            registered = any(e.get("path") == launcher_path() for e in json.loads(ENGINES_JSON.read_text()))
        except Exception:
            pass
    running = subprocess.run(["pgrep", "-x", "en-croissant"], capture_output=True).returncode == 0
    return {"installed": installed_app, "registered": registered, "running": running}


def self_check(cfg: Config) -> None:
    """Talk UCI to the launcher exactly as En Croissant would."""
    script = (
        "uci\nisready\n"
        "setoption name UCI_Chess960 value false\nsetoption name MultiPV value 3\n"
        "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4\n"
        "go wtime 180000 btime 180000 winc 2000 binc 2000\n"
    )
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"}
    if os.environ.get("MIRROR_DATA_DIR"):
        env["MIRROR_DATA_DIR"] = os.environ["MIRROR_DATA_DIR"]
    t0 = time.time()
    p = subprocess.Popen([launcher_path()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         cwd=str(Path(launcher_path()).parent), env=env)
    p.stdin.write(script)
    p.stdin.flush()
    got_best = False
    for line in p.stdout:
        print(f"  < {line.rstrip()}")
        if line.startswith("bestmove"):
            got_best = True
            break
    p.stdin.write("quit\n")
    p.stdin.flush()
    p.wait(timeout=10)
    print(f"{'OK' if got_best else 'FAILED'}: engine answered in {time.time() - t0:.1f}s (with a Finder-like PATH).")


def add_to_encroissant(dry_run: bool = False) -> None:
    """Append a local-engine entry to En Croissant's engines.json (backup first).

    Quit En Croissant before running this: it keeps the engine list in memory and
    rewrites the file, and a single invalid entry makes it reset the whole list.
    """
    path = ENGINES_JSON
    running = subprocess.run(["pgrep", "-x", "en-croissant"], capture_output=True).returncode == 0
    if running and not dry_run:
        print("En Croissant is running. Quit it first (it would overwrite engines.json).")
        return
    if not path.parent.exists():
        print("En Croissant engine folder not found - open En Croissant once first.")
        return
    engines = json.loads(path.read_text()) if path.exists() else []
    if any(e.get("path") == launcher_path() for e in engines):
        print("Mirror is already registered in En Croissant.")
        return
    entry = {
        "type": "local",
        "id": str(uuid.uuid4()),
        "name": "Mirror",
        "version": "0.1",
        "path": launcher_path(),
        "loaded": True,
        "go": {"t": "Infinite"},
        "settings": [{"name": "MultiPV", "value": 3}],
    }
    print(json.dumps(entry, indent=2))
    if dry_run:
        print("(dry run: nothing written)")
        return
    backup = path.with_name(f"engines.json.bak-{dt.datetime.now():%Y%m%d-%H%M%S}")
    if path.exists():
        shutil.copyfile(path, backup)
    engines.append(entry)
    path.write_text(json.dumps(engines, indent=4))
    print(f"Added 'Mirror' to En Croissant (previous list backed up to {backup.name}). Start En Croissant to use it.")
