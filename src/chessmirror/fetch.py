"""Download the player's games from the public Lichess / Chess.com APIs (no auth needed)."""

from __future__ import annotations

import datetime as dt
import json
import urllib.request

USER_AGENT = "chessmirror/0.1 (personal chess analysis tool)"


def _get(url: str, accept: str = "application/x-chess-pgn", timeout: int = 120) -> str:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_lichess(username: str, max_games: int | None = None, since: dt.datetime | None = None,
                  perf_types: str | None = None) -> str:
    params = ["clocks=true", "opening=true", "evals=false", "moves=true", "tags=true"]
    if max_games:
        params.append(f"max={int(max_games)}")
    if since:
        params.append(f"since={int(since.timestamp() * 1000)}")
    if perf_types:
        params.append(f"perfType={perf_types}")
    return _get(f"https://lichess.org/api/games/user/{username}?{'&'.join(params)}")


def fetch_chesscom(username: str, months: int | None = None) -> str:
    archives = json.loads(
        _get(f"https://api.chess.com/pub/player/{username.lower()}/games/archives", accept="application/json")
    )["archives"]
    if months:
        archives = archives[-months:]
    chunks = []
    for url in archives:
        chunks.append(_get(url + "/pgn"))
    return "\n\n".join(chunks)
