"""Ground-truth taxonomy loader.

`data/ground_truth.csv` is the single source of truth for what a correct run of
the experiment looks like. Edit the CSV (or re-export from your sheet).

Two columns are not in the original sheet and drive the scorers:
  stage          which eval stage the step belongs to (fragment_generation ...
                 final_qc, or "records" for the auditing steps; blank = pre-run
                 technician prep that no stage covers)
  evidence_role  which kind of tool-call evidence credits the step (see
                 alignment.evidenced_steps for what each role means). Every
                 agent-owned ("y") step needs one, or coverage can never credit it.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from opalite_bench.paths import DATA

DEFAULT_CSV = DATA / "ground_truth.csv"


@dataclass(frozen=True)
class Step:
    step_id: int
    phase: str
    step: str
    details: str
    agent: str  # "y" | "n" | "n*"  (n* = not at the moment)
    agent_intervention: str
    human_intervention: str
    checkpoint: str
    failure_modes: tuple[str, ...]
    stage: str = ""
    evidence_role: str = ""

    @property
    def agent_owned(self) -> bool:
        return self.agent == "y"

    @property
    def future_agent(self) -> bool:
        return self.agent == "n*"

    @property
    def needs_human(self) -> bool:
        return self.human_intervention.strip() not in {"", "-"}


def _split_failures(s: str) -> tuple[str, ...]:
    parts = [p.strip() for chunk in s.split(",") for p in chunk.split("/")]
    return tuple(p for p in parts if p)


def load_taxonomy(path: Path | str = DEFAULT_CSV) -> list[Step]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Step(
            step_id=int(r["step_id"]),
            phase=r["phase"],
            step=r["step"],
            details=r["details"],
            agent=r["agent"].strip().lower(),
            agent_intervention=r["agent_intervention"],
            human_intervention=r["human_intervention"],
            checkpoint=r["checkpoint"],
            failure_modes=_split_failures(r["failure_modes"]),
            stage=(r.get("stage") or "").strip(),
            evidence_role=(r.get("evidence_role") or "").strip(),
        )
        for r in rows
    ]


def by_id(steps: list[Step]) -> dict[int, Step]:
    return {s.step_id: s for s in steps}


def agent_step_ids(steps: list[Step]) -> list[int]:
    return [s.step_id for s in steps if s.agent_owned]


def phases(steps: list[Step]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for s in steps:
        out.setdefault(s.phase, []).append(s.step_id)
    return out


def step_stages(steps: list[Step]) -> dict[int, str]:
    """step id -> stage, for steps that belong to a stage."""
    return {s.step_id: s.stage for s in steps if s.stage}


def stage_roles(steps: list[Step]) -> dict[str, dict[str, list[int]]]:
    """stage -> evidence role -> step ids, e.g. {"gibson_round1": {"load": [30], ...}}."""
    out: dict[str, dict[str, list[int]]] = {}
    for s in steps:
        if s.stage and s.evidence_role:
            out.setdefault(s.stage, {}).setdefault(s.evidence_role, []).append(s.step_id)
    return out


def role_step_ids(steps: list[Step], *roles: str) -> list[int]:
    """All step ids (across stages) whose evidence role is one of `roles`."""
    return [s.step_id for s in steps if s.evidence_role in roles]
