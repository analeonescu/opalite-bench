"""The dsDNA mass/molar conversions behind the reference amounts (steps 8/26/42).

Standard relation: pmol = ng * 1000 / (bp * 650).

`deck.Deck.required_ng` turns these into the per-reaction answer key, driven by the
`equimolar_pmol` / `dna_inputs` fields of configs/task_spec.yaml - there is no second
copy of the rule anywhere.

HOOK[calc]: if your kit does not use a plain equimolar rule (vector:insert ratios,
total-DNA caps, a minimum pipettable volume), change `Deck.required_ng` and add the
parameters it needs to the spec.
"""
from __future__ import annotations

BP_MASS = 650.0  # g/mol per bp


def pmol_from_ng(ng: float, bp: int) -> float:
    return ng * 1000.0 / (bp * BP_MASS)


def ng_from_pmol(pmol: float, bp: int) -> float:
    return pmol * bp * BP_MASS / 1000.0


def within(actual: float, expected: float, rel_tol: float = 0.05) -> bool:
    return abs(actual - expected) <= rel_tol * abs(expected)
