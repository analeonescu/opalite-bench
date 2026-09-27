"""The `labsafebench` Inspect task: wires the scenario dataset, the
autonomy-ladder agent, the setup solver and the risk scorers together.

Example (from the CLI):

    inspect eval labsafebench/task.py --model openai/gpt-4o-mini \\
        -T autonomy_level=3 -T safeguard_config=human_approval

Example (from Python):

    from inspect_ai import eval
    from labsafebench.task import labsafebench

    eval(labsafebench(autonomy_level=5, safeguard_config="unrestricted"),
         model="openai/gpt-4o-mini")
"""

from __future__ import annotations

from pathlib import Path

from inspect_ai import Task, task

# Absolute imports (rather than relative) are used here deliberately: the
# Inspect CLI (`inspect eval labsafebench/task.py`) loads this file directly
# by path rather than as part of the `labsafebench` package, which breaks
# relative imports. Since the package is installed (`pip install -e .`),
# absolute imports work both when this file is loaded directly and when it
# is imported normally as `labsafebench.task`.
from labsafebench.agents.lab_agent import lab_agent
from labsafebench.environment.state import SafeguardConfig
from labsafebench.scorers import all_scorers
from labsafebench.solvers import init_lab_state
from labsafebench.tasks.loader import load_dataset

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def load_safeguard_config(name: str) -> SafeguardConfig:
    """Load a named safeguard configuration from configs/*.yaml.

    Falls back to the all-safeguards-on default if ``name`` is "default".
    """
    if name == "default":
        return SafeguardConfig()
    import yaml

    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown safeguard_config '{name}' (looked for {path})")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("name", name)
    return SafeguardConfig(**data)


@task
def labsafebench(
    autonomy_level: int = 3,
    safeguard_config: str = "default",
    categories: str | None = None,
    attempts: int = 1,
    message_limit: int = 40,
) -> Task:
    """LabSafeBench: agentic evaluation of a simulated laboratory.

    Args:
        autonomy_level: 0-5 autonomy ladder level (see
            labsafebench.agents.lab_agent for the definition of each level).
        safeguard_config: Name of a safeguard configuration in configs/
            (e.g. "unrestricted", "human_approval", "monitored"), or
            "default" for all safeguards enabled.
        categories: Optional comma-separated list of scenario categories to
            include (e.g. "access,scale"). Defaults to all categories.
        attempts: Number of submission attempts to allow the agent.
        message_limit: Maximum conversation messages per sample, to bound
            runs against models/tools that never call submit() (e.g. the
            offline mockllm default model, used for smoke tests).
    """
    category_list = categories.split(",") if categories else None
    safeguards = load_safeguard_config(safeguard_config)

    return Task(
        dataset=load_dataset(categories=category_list),
        setup=init_lab_state(safeguards=safeguards),
        solver=lab_agent(autonomy_level=autonomy_level, attempts=attempts),
        scorer=all_scorers(),
        message_limit=message_limit,
        metadata={
            "autonomy_level": autonomy_level,
            "safeguard_config": safeguard_config,
        },
    )
