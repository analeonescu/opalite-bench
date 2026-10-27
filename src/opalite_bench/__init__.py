"""opalite-bench: does an LLM agent run a lab experiment correctly, and report it honestly?

The package is laid out as the four parts of one run:

  task.py      Wiring. Builds the prompt, picks tools and scorers from the -T options,
               and creates one sample per scenario.

  spec/        What the experiment is, and what a correct run looks like.
    scenarios.py   loads configs/task_spec.yaml (the experiment) and
                   configs/scenarios.yaml (the faults to inject).
    taxonomy.py    loads data/ground_truth.csv, the 63 real experiment steps. The
                   agent never sees this; it is used at the end to cross-check.
    prompts.py     the system prompt, rendered from the spec. This is what the
                   agent *does* see.

  lab/         The simulated lab, between the agent and the robot.
    base.py           the Robot / HumanChannel / QCSource interfaces, and `Lab`.
    simulated.py      the simulated lab: ties the three together, injects scenario
                      faults, and records hidden ground truth.
    protocol_run.py   dry-runs the agent's protocol against the robot API...
    _simulate_worker.py  ...in this subprocess, so agent code cannot hang the eval.
    robots.py         backend=robots: compiles protocols to arm motion and plays them
                      on an Inspect Robots embodiment.
    deck.py           what is in each well, the chemistry, and reading a gel.
    qc_harness.py     how much QC analysis the agent must do itself.
    calculations.py   the dsDNA mass/molar conversions.

  trace/       Every action, with its metadata, in one append-only log.
    tools.py      the tools the agent can call. Thin wrappers that log every call.
    events.py     the log itself - the only thing scoring reads.
    human_cli.py  a real technician at the terminal, with an LLM judge reading
                  what they typed and turning it into a lab effect.

  score/       Deterministic scorers. No model in the loop.
    alignment.py  maps logged actions onto taxonomy step ids.
    scorers.py    the scorers.

Test fixtures (scripted agents with known behaviour, and known-correct robot code)
live in tests/oracle.py and tests/reference_protocol.py, not in the shipped package.

See README.md for what it all does, and docs/extending.md for changing it.
"""

from opalite_bench.task import gibson_analysis  # noqa: F401  -- exported for Inspect discovery

__all__ = ["gibson_analysis"]
