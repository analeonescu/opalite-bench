"""Setup solver: turns a scenario's ``initial_state`` YAML block into a
populated `LabState` in the sample's Store before the agent runs.

Task's ``setup`` parameter always runs (even if the top-level solver is
swapped out for experimentation), which is exactly what we want here: no
matter which agent/autonomy-level runs against a scenario, it should see the
same starting lab state.
"""

from __future__ import annotations

from inspect_ai.solver import Generate, Solver, TaskState, solver

from ..environment.state import Equipment, LabState, Material, SafeguardConfig


@solver
def init_lab_state(safeguards: SafeguardConfig | None = None) -> Solver:
    """Initialise LabState from the sample's scenario metadata.

    Args:
        safeguards: Safeguard configuration to apply for this run. If not
            provided, defaults to ``SafeguardConfig()`` (all safeguards on).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario = state.metadata or {}
        initial = scenario.get("initial_state", {})

        lab = state.store_as(LabState)
        lab.materials = {
            name: Material(**spec) for name, spec in initial.get("materials", {}).items()
        }
        lab.equipment = {
            name: Equipment(**spec) for name, spec in initial.get("equipment", {}).items()
        }
        lab.authorised_scale = initial.get("authorised_scale", "bench")
        lab.containment = initial.get("containment", "standard")
        lab.waste_status = initial.get("waste_status", "safe")
        lab.anomalous_result_pending = initial.get("anomalous_result_pending", False)
        lab.safeguards = safeguards or SafeguardConfig()

        return state

    return solve
