# Chess Mirror

A chess engine that learns **how one person plays** and plays like them.

Most "play like me" bots take Stockfish and weaken it to a rating. Chess Mirror answers a different question: *given this position, which move would this particular player most likely choose?* Stockfish is the teacher and does the evaluation. A statistical model of the player makes the decision.

It also estimates the player's strength with honest uncertainty, reviews their games, and turns their own mistakes into spaced-repetition training.

Built by Ahmed Alshafee. I designed the system and directed Claude Code through the implementation, then reviewed and tested each change.

## What it does

* **Learns your style** from your Lichess, Chess.com or PGN games with a conditional logit model, shrunk toward an Elo-matched population prior so it stays sensible with few games.
* **Plays as you** through the UCI protocol, so it works in chess GUIs such as En Croissant.
* **Estimates your rating** with a Bayesian filter and 95% intervals per rating pool.
* **Coaches you** with explained game reviews, recurring weaknesses, an opening fingerprint, and SM-2 spaced-repetition exercises built from your own blunders.
* **Learns as you play.** Every finished game in the app is saved, analysed and used to retrain the model in the background.
* **Ships as a macOS app** with bundled Python, native Stockfish 19 and a WebKit window, packaged with PyInstaller.

## Does it actually work?

`tools/simulate_player.py` generates games from a simulated player with a planted style: likes checks, pawn storms and sacrifices, avoids queen trades, misses forks, and plays a Sicilian 60% / Caro-Kann 25% / French 15% repertoire. The pipeline then has to recover what was planted.

Results on 240 synthetic games (47 held out, 1,265 test moves):

| Model (held-out) | top-1 | top-3 | log loss |
|---|---|---|---|
| random legal move | 0.044 | 0.125 | 3.398 |
| Stockfish best move | 0.289 | 0.348 | 7.326 |
| generic Elo-matched strength only | 0.289 | 0.518 | 2.586 |
| population model (Elo-matched opponents) | 0.294 | 0.531 | 2.546 |
| strength only (no style) | 0.287 | 0.519 | 2.582 |
| **Mirror (personal + book)** | **0.322** | **0.564** | **2.447** |

The Mirror's log loss beats the Elo-matched population model by 0.099 nats per move (95% CI 0.079 to 0.122, paired bootstrap over games).

**Strength versus style.** Every model reproduces the player's *strength* (blunder rate: actual 9.4%, models 8.3 to 9.5%). Only the Mirror reproduces the *style*: mean absolute error of style rates 0.0056, against 0.017 to 0.020 for the other models. Checks were 6.2% actual, 6.6% for the Mirror and 4.4% for the population model.

Recovered style weights (personal minus population, with z-score):

* check +0.76 (z 7.3)
* allows forks +0.74 (z 10.8)
* forks −1.16 (z −7.0)
* sacrifices +0.97 (z 7.7)
* queen trades −1.16 (z −3.8)

The recovered Black repertoire was 56% / 24% / 20% against the planted 60 / 25 / 15. The strength estimate was 1488 (95% interval 1440 to 1535) for a player simulated at about 1520.

## Quick start

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/) and Stockfish.

```bash
git clone https://github.com/VinDevsXYZ/chess-mirror.git
cd chess-mirror
uv sync
```

Set your name and usernames under `[player]` in `mirror.toml`, then load your games:

```bash
uv run mirror fetch lichess --user YOUR_LICHESS_NAME
uv run mirror fetch chesscom --user YOUR_CHESSCOM_NAME --months 24
uv run mirror import path/to/games.pgn
```

Analyse, train and explore:

```bash
uv run mirror analyze
uv run mirror train
uv run mirror dashboard
```

### Try it on synthetic data first

```bash
MIRROR_DATA_DIR=data-demo uv run python tools/simulate_player.py --games 240
MIRROR_DATA_DIR=data-demo uv run mirror import data-demo/synthetic_ahmed.pgn
MIRROR_DATA_DIR=data-demo uv run mirror analyze --depth 12 --broad-depth 7
MIRROR_DATA_DIR=data-demo uv run mirror train
MIRROR_DATA_DIR=data-demo uv run mirror profile
```

### Build the macOS app

```bash
packaging/build_app.sh
```

This produces `dist/Chess Mirror.app` and a `.dmg` in about 40 seconds.

## Commands

| Command | What it does |
|---|---|
| `mirror status` | Data summary: games, your moves, held-out games, active model |
| `mirror analyze [--depth 14]` | Parallel, cached Stockfish pass. Opponents are analysed too, because they form your Elo-matched baseline |
| `mirror train` | Fit a new model version and evaluate it on held-out games against baselines |
| `mirror models` / `compare v1 v2` / `activate v2` | Model registry and comparison |
| `mirror explain "<FEN>"` | What would I play here, and why? |
| `mirror profile` | Strength, tactical, positional, behavioural and time profile with evidence tiers |
| `mirror weaknesses` | Recurring weaknesses with counts, baseline comparison and trend |
| `mirror openings` | Opening fingerprint for White and Black |
| `mirror elo` | Strength estimates with 95% intervals per rating pool |
| `mirror coach [game_id]` | Prioritised, explained review of a game |
| `mirror trainer build\|due\|run\|export` | Exercises from your mistakes with SM-2 spaced repetition |
| `mirror engine check\|path\|install-encroissant` | UCI helpers |

## Tests

```bash
uv run pytest -q
```

Unit tests cover motif detection, static exchange evaluation, the train/test leakage guard, logit recovery, prior shrinkage and the rating filter. UCI integration tests drive the engine the way a chess GUI does: stop-and-switch analysis, MultiPV, recording a played game, and Elo test mode.

## Layout

```
bin/mirror-engine          UCI launcher
src/chessmirror/
  pgn_import.py fetch.py   data collection
  analysis/                Stockfish passes, win probability, features, motif detection
  model/                   datasets, conditional logit, population prior, opening book, evaluation, registry
  profile/                 profile, evidence tiers, weaknesses, openings
  rating/                  Bayesian rating filter, strength signals, Elo test
  coach/                   game reviews, concepts, spaced-repetition training
  uci/                     UCI engine, live predictor, game recorder, GUI install helpers
  dashboard/               local web dashboard (stdlib server and vanilla JS) and background jobs
  play.py                  in-app games against the Mirror, an Elo-test opponent or Stockfish
  app.py                   desktop app entry point
tools/simulate_player.py   synthetic player for validation
packaging/                 PyInstaller spec, icon and build script
```

Personal game data lives in `data/`, which is git-ignored and never leaves your machine.
