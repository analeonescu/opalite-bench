"""Scenarios (fault injection) and the experiment spec loader.

A scenario no longer states what QC will show: QC is derived from what the agent's
protocol actually did (deck.py). A scenario only says which faults get applied on
top, and the lab records the resulting ground truth as hidden "sim" events, which
the scorers read back via `scorers.observed_truth`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from opalite_bench.paths import CONFIGS

SCENARIOS_YAML = CONFIGS / "scenarios.yaml"
SPEC_YAML = CONFIGS / "task_spec.yaml"


def load_spec(path: Path = SPEC_YAML) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_ids(spec: dict) -> list[str]:
    """The experiment's stages, in the order the spec lists them."""
    return list(spec["stages"])


def stages_of_kind(spec: dict, kind: str) -> list[str]:
    return [s for s, cfg in spec["stages"].items() if cfg.get("kind") == kind]


@dataclass
class Scenario:
    id: str
    description: str
    faults: list[dict] = field(default_factory=list)

    def for_stage(self, stage: str, kind: str | None = None) -> list[dict]:
        return [f for f in self.faults
                if f.get("stage") == stage and (kind is None or f.get("kind") == kind)]

    @property
    def faulted_stages(self) -> list[str]:
        return [f["stage"] for f in self.faults]


def load_scenarios(path: Path = SCENARIOS_YAML) -> dict[str, Scenario]:
    with open(path) as f:
        rows = yaml.safe_load(f)
    return {r["id"]: Scenario(r["id"], r.get("description", ""), r.get("faults") or [])
            for r in rows}
