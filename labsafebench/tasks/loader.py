"""Load YAML scenario files into Inspect `Sample` objects.

Each scenario YAML is a small, machine-readable description of:

* the objective given to the agent (``prompt``)
* the initial lab state (``initial_state``)
* which risk vectors the scenario probes (``risk_vectors``, drawn from
  docs/risk_taxonomy.md)
* what counts as completing the scientific objective (``success``)

This keeps the benchmark's scenarios as data rather than prose, per the
project's design goal of a "machine-readable risk ontology" rather than a
pile of prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scenario_to_sample(scenario: dict[str, Any]) -> Sample:
    """Convert a parsed scenario dict into an Inspect `Sample`.

    The full scenario dict is preserved in ``metadata`` so that the setup
    solver can initialise `LabState` from it and the scorers/analysis can
    read back ``risk_vectors``, ``category`` and ``success`` criteria.
    """
    return Sample(
        input=scenario["prompt"],
        target=scenario["id"],
        id=scenario["id"],
        metadata=scenario,
    )


def load_scenarios(
    categories: list[str] | None = None,
    scenarios_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load all scenario YAML files, optionally filtered by category."""
    directory = scenarios_dir or SCENARIOS_DIR
    scenarios: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*/*.yaml")):
        scenario = _load_yaml(path)
        if categories is None or scenario.get("category") in categories:
            scenarios.append(scenario)
    return scenarios


def load_dataset(
    categories: list[str] | None = None,
    scenarios_dir: Path | None = None,
) -> Dataset:
    """Build an Inspect `Dataset` from the scenario YAML files."""
    scenarios = load_scenarios(categories=categories, scenarios_dir=scenarios_dir)
    samples = [scenario_to_sample(s) for s in scenarios]
    return MemoryDataset(samples=samples, name="labsafebench")
