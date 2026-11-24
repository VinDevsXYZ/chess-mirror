"""Integration tests: talk UCI to the launcher like En Croissant does (needs Stockfish)."""

import json
import os
import queue
import random
import subprocess
import threading
import time
from pathlib import Path

import chess
import pytest

from chessmirror.config import PROJECT_ROOT, find_stockfish

try:
    find_stockfish()
    HAVE_SF = True
except FileNotFoundError:
    HAVE_SF = False

pytestmark = pytest.mark.skipif(not HAVE_SF, reason="Stockfish not installed")


class Engine:
    def __init__(self, data_dir: Path):
        env = {"PATH": "/usr/bin:/bin", "MIRROR_DATA_DIR": str(data_dir)}
        self.p = subprocess.Popen([str(PROJECT_ROOT / "bin" / "mirror-engine")], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, env=env, cwd=str(PROJECT_ROOT / "bin"))
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            self.q.put(line.rstrip())

    def send(self, s: str):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def until(self, prefix: str, timeout: float = 60) -> list[str]:
        out, end = [], time.time() + timeout
        while time.time() < end:
            try:
                line = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            out.append(line)
            if line.startswith(prefix):
                return out
        raise TimeoutError(f"no {prefix!r}; got {out[-5:]}")

    def close(self):
        self.send("quit")
        self.p.wait(timeout=10)


@pytest.fixture
def eng(tmp_path):
    e = Engine(tmp_path)
    e.send("uci")
    e.until("uciok")
    e.send("isready")
    e.until("readyok", timeout=60)
    yield e, tmp_path
    e.close()


def test_analysis_multipv_and_stop(eng):
    e, _ = eng
    e.send("setoption name MultiPV value 3")
    e.send("position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4")
    e.send("go infinite")
    lines = [ln for ln in e.until("info depth") if ln.startswith("info depth")]
    got = lines + [e.q.get(timeout=30) for _ in range(2)]
    assert [f"multipv {k}" in got[k - 1] for k in (1, 2, 3)] == [True, True, True]
    t = time.time()
    e.send("stop")
    out = e.until("bestmove", timeout=5)
    assert time.time() - t < 1.0 and out[-1].split()[1] in [m.uci() for m in chess.Board(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").legal_moves]


def test_quick_switch_answers_immediately(eng):
    """En Croissant sends stop and a new position 50 ms later without waiting."""
    e, _ = eng
    e.send("position fen r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    e.send("go infinite")
    time.sleep(0.01)
    t = time.time()
    e.send("stop")
    e.until("bestmove", timeout=2)
    assert time.time() - t < 0.5


def test_play_game_is_recorded(eng):
    e, data = eng
    random.seed(3)
    root = chess.STARTING_FEN
    board = chess.Board()
    moves: list[str] = []
    for _ in range(6):  # human = White (random moves), engine = Black
        human = random.choice(list(board.legal_moves))
        board.push(human)
        moves.append(human.uci())
        e.send(f"position fen {root} moves {' '.join(moves)}")
        e.send("go wtime 170000 btime 175000 winc 2000 binc 2000")
        best = e.until("bestmove")[-1].split()[1]
        board.push_uci(best)
        moves.append(best)
    e.send("position fen 8/8/8/8/8/5k2/8/5K2 w - - 0 1")  # a different game starts -> previous closes
    time.sleep(1.0)
    pending = list((data / "state" / "pending").glob("*.json"))
    assert len(pending) == 1
    rec = json.loads(pending[0].read_text())
    assert rec["engine_color"] == "black" and len(rec["moves"]) == 12 and rec["result"] is None
    assert rec["clocks"]  # human clock data captured from go wtime/btime


def test_elotest_mode(eng):
    e, _ = eng
    e.send("setoption name Mode value EloTest")
    e.send("setoption name EloTestOpponent value 1500")
    e.send(f"position fen {chess.STARTING_FEN}")
    e.send("go movetime 200")
    out = e.until("bestmove", timeout=20)
    assert any("pv" in ln for ln in out) and any("UCI_Elo 1500" in ln for ln in out)
