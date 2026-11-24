"""Configuration: project paths, player identity, analysis settings.

Config is read from ``mirror.toml`` in the project root (or ``$MIRROR_CONFIG``).
The data directory can be overridden with ``$MIRROR_DATA_DIR`` so that demo /
synthetic data never mixes with real games.
"""

from __future__ import annotations

import os
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# Packaged app (PyInstaller) vs running from the source checkout.
FROZEN = bool(getattr(sys, "frozen", False))
APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Chess Mirror"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
# Where mirror.toml and data/ live: the checkout when developing, Application Support in the app.
PROJECT_ROOT = APP_SUPPORT if (FROZEN or os.environ.get("MIRROR_APP_MODE")) else SOURCE_ROOT
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", SOURCE_ROOT))

ENCROISSANT_DIR = Path.home() / "Library" / "Application Support" / "org.encroissant.app"


@dataclass
class PlayerConfig:
    name: str = "Me"
    # Case-insensitive names/usernames that identify the player in PGN headers.
    aliases: list[str] = field(default_factory=list)
    lichess: str = ""
    chesscom: str = ""
    rating: int = 1500  # rough self-estimate; only seeds the prior until real data exists

    def all_names(self) -> set[str]:
        names = {self.name, *self.aliases, self.lichess, self.chesscom}
        return {n.strip().lower() for n in names if n and n.strip()}


@dataclass
class AnalysisConfig:
    deep_depth: int = 12          # depth of the MultiPV "teacher" pass
    deep_multipv: int = 5
    broad_depth: int = 8          # shallow pass over ALL legal moves
    workers: int = max(1, (os.cpu_count() or 4) - 2)
    hash_mb: int = 64
    analyze_opponents: bool = True  # opponents feed the population prior + baselines


@dataclass
class EngineConfig:
    stockfish_path: str = ""      # blank = auto-detect
    think_depth: int = 12         # deep pass depth used live in the UCI engine
    think_multipv: int = 5
    broad_depth: int = 7


@dataclass
class Config:
    player: PlayerConfig = field(default_factory=PlayerConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "mirror.db"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.raw_dir, self.models_dir, self.reports_dir, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)

    def stockfish(self) -> str:
        return find_stockfish(self.engine.stockfish_path)


def find_stockfish(explicit: str = "") -> str:
    """Locate a Stockfish binary: explicit path, PATH, Homebrew, then En Croissant's copy."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if FROZEN:
        candidates.append(str(BUNDLE_DIR / "stockfish"))  # native Stockfish shipped inside the app
    if os.environ.get("MIRROR_STOCKFISH"):
        candidates.append(os.environ["MIRROR_STOCKFISH"])
    found = shutil.which("stockfish")
    if found:
        candidates.append(found)
    candidates += ["/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"]
    sf_dir = ENCROISSANT_DIR / "engines" / "stockfish"
    if sf_dir.is_dir():
        candidates += sorted(str(p) for p in sf_dir.glob("stockfish*") if p.is_file())
    for c in candidates:
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return c
    raise FileNotFoundError(
        "Stockfish not found. Install it (brew install stockfish) or set "
        "[engine] stockfish_path in mirror.toml."
    )


def _apply(obj, values: dict) -> None:
    for k, v in values.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    path = Path(path or os.environ.get("MIRROR_CONFIG") or PROJECT_ROOT / "mirror.toml")
    if path.is_file():
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        _apply(cfg.player, raw.get("player", {}))
        _apply(cfg.analysis, raw.get("analysis", {}))
        _apply(cfg.engine, raw.get("engine", {}))
        if raw.get("paths", {}).get("data_dir"):
            d = Path(raw["paths"]["data_dir"]).expanduser()
            cfg.data_dir = d if d.is_absolute() else PROJECT_ROOT / d
    if os.environ.get("MIRROR_DATA_DIR"):
        cfg.data_dir = Path(os.environ["MIRROR_DATA_DIR"]).expanduser().resolve()
    return cfg


def config_path() -> Path:
    return Path(os.environ.get("MIRROR_CONFIG") or PROJECT_ROOT / "mirror.toml")


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_config(cfg: Config, path: Path | None = None) -> Path:
    """Persist the configuration (used by the app's Setup screen)."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = cfg.data_dir
    try:
        data_dir = data_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    sections = {"player": cfg.player, "analysis": cfg.analysis, "engine": cfg.engine}
    lines = ["# Personal Chess Mirror configuration (edited by the app's Setup screen).", ""]
    for name, obj in sections.items():
        lines.append(f"[{name}]")
        for k, v in vars(obj).items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    lines += ["[paths]", f"data_dir = {_toml_value(str(data_dir))}", ""]
    path.write_text("\n".join(lines))
    return path


def ensure_config() -> Config:
    """Load the config, creating a default file on first run (app mode)."""
    if not config_path().exists():
        cfg = Config()
        write_config(cfg)
    return load_config()
