"""Where this repo's files live.

Every path is anchored to the repo root, found by walking up from this file rather
than counted with `parents[n]`, so moving a module between subpackages cannot
silently point it somewhere else.

Nothing here is relative to the working directory. A bare `Path("logs/...")` would
resolve against wherever the eval was launched from, which quietly scatters output:
running from `src/opalite_bench/` once was enough to create a second, empty `logs/` there.
"""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """The directory holding `configs/` and `data/`."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "data").is_dir():
            return parent
    raise RuntimeError("could not locate the repo root (no parent holds configs/ and data/)")


ROOT = repo_root()
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
LOGS = ROOT / "logs"        # eval logs and the technician's intervention log
