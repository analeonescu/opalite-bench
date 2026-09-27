# LabSafeBench

An open-source, proof-of-concept benchmark for evaluating how AI agents
behave when given increasing levels of access and autonomy within a
simulated scientific workflow, starting from a chemical-security risk
taxonomy.

Existing chemical-security evaluations primarily test what models can *say*
about hazardous procedures. LabSafeBench explores the next step: what
happens when an AI system can actually interact with tools and (simulated)
scientific infrastructure?

The benchmark converts a chemical-security risk taxonomy into simulated,
agentic laboratory tasks. The lab, its materials and its instruments are
**deliberately abstract and synthetic** -- there is no real chemistry here,
only the risk-relevant structure (restricted materials, scale limits,
containment controls, approval requirements). The goal is to study agentic
capability, autonomy and safeguards, not to teach or evaluate dangerous
procedures.

```
MODEL                         SAFETY CONTROLS
  |                           ----------------
AGENT                         permissions
  |                           monitoring
TOOLS        <-------->       human approval
  |                           containment
LAB ENVIRONMENT                logging
  |
EXPERIMENT
  |
OBSERVATION
  |
NEXT ACTION
```

## Research question

How do AI agents' scientific capabilities and safety behaviours change as
they are given increasing access to laboratory tools and autonomy?

* **Capability** -- what scientific tasks can agents complete?
* **Safety** -- what unsafe actions do they attempt (and commit)?
* **Control** -- which safeguards actually prevent those actions?

See `docs/methodology.md` for the full methodology and `docs/risk_taxonomy.md`
for how the risk vectors map onto the environment's constraints.

## What's here (v0.1)

* A tiny, abstract simulated laboratory (`labsafebench/environment/`): a
  handful of materials, two instruments, scale/containment/approval state,
  implemented as an Inspect `StoreModel` plus a set of `@tool`-decorated
  actions (`get_lab_state`, `request_material`, `configure_experiment`,
  `run_experiment`, `read_last_result`, `dispose_waste`,
  `request_human_approval`).
* A **six-level autonomy ladder** (`labsafebench/agents/lab_agent.py`),
  0 (advice only) through 5 (extended, multi-step autonomy), implemented as
  one parameterised `react()` agent whose tool access scales with the level.
* **15 scenarios** across 6 categories (`labsafebench/tasks/scenarios/`),
  each a small YAML file declaring its objective, initial lab state, risk
  vectors and success criteria -- a machine-readable risk ontology rather
  than a pile of prompts.
* **Five scorers** (`labsafebench/scorers/`) that separate *capability*
  (did the agent complete the objective?), *safety* (how many unsafe
  actions did it commit?), *intervention* (did it ask for approval when it
  should have?), *safeguard effectiveness* (did the environment's controls
  actually block unsafe attempts?), and *autonomy horizon* (how many actions
  before the first committed violation?).
* **Three safeguard configurations** (`configs/*.yaml`): `unrestricted`,
  `human_approval`, and `monitored` -- the same scenario and model can be run
  under each to see which safeguards actually change agent behaviour.
* A pilot sweep script (`scripts/run_pilot.py`) and a test suite
  (`tests/`) that exercises the environment, scorers and full task using
  Inspect's offline `mockllm` provider (no API keys required).

## Quickstart

```bash
pip install -e .
inspect eval labsafebench/task.py --model openai/gpt-4o-mini \
    -T autonomy_level=3 -T safeguard_config=human_approval
```

Or from Python:

```python
from inspect_ai import eval
from labsafebench.task import labsafebench

logs = eval(
    labsafebench(autonomy_level=5, safeguard_config="unrestricted"),
    model="openai/gpt-4o-mini",
)
```

Run the test suite (offline, no API keys needed):

```bash
pip install -e ".[dev]"
python -m pytest tests/
```

## The capability -> control -> consequence design principle

The environment never tells the agent "this is unsafe, don't do it" and
never silently decides on the agent's behalf. Instead, every risk-relevant
tool call is allowed to be *attempted*, and the environment records:

1. **Capability**: did the agent attempt the action?
2. **Control**: did the active safeguard configuration block it?
3. **Consequence**: if not blocked, the lab state actually changed.

This is what lets the benchmark distinguish "the model never tries anything
risky" from "the model tries constantly but is always blocked" -- two very
different behaviours a single pass/fail score would conflate.

## The autonomy ladder

| Level | Name              | Tools available                                                |
|-------|-------------------|------------------------------------------------------------------|
| 0     | Advice            | none -- recommendation only                                       |
| 1     | Planning          | read-only lab state                                               |
| 2     | Tool use          | + select material, configure experiment                          |
| 3     | Execution         | + run experiment, read result, dispose waste, request approval   |
| 4     | Adaptive autonomy | same tools as 3, encouraged to decide next steps itself           |
| 5     | Extended autonomy | same tools as 3, encouraged to run multi-step sequences without pausing |

Running the same scenario across levels makes "at what autonomy level does
the agent violate a safety constraint?" a directly observable variable.

## Where Inspect Robots fits

LabSafeBench is built on [Inspect AI](https://inspect.aisi.org.uk/), using
its native `@tool`, `react()` agent, `StoreModel` and `@scorer` primitives.
The architecture deliberately separates the *risk taxonomy + scenarios +
scorers* from the *embodiment* (currently: a synthetic Python state
machine):

```
                    LABSAFEBENCH
                         |
             +-----------+-----------+
             |                       |
        Inspect AI             Inspect Robots
             |                       |
       simulated lab          physical/simulated lab
             |                       |
             +-----------+-----------+
                         |
                  common tasks
                  + risk vectors
                  + scorers
```

[Inspect Robots](https://pypi.org/project/inspect-robots/) is designed
around a swappable policy + embodiment (a real or simulated robot/rig), and
already supports LLM-agent policies via tool calls. Because LabSafeBench's
scenarios, risk vectors and scorers are decoupled from the environment
implementation, a future version could plug in an Inspect Robots embodiment
without changing the scenario definitions or scoring logic -- this is
explicitly a v0.2+ extension, not part of the current scope.

## Scope decision

For this public repo we deliberately avoid implementing real hazardous
chemistry or detailed procedures. The laboratory environment is abstract
and every scenario's safety properties are encoded in the environment's
state machine, not in real-world chemical detail. This preserves the
research value -- agentic capability, autonomy, safeguards, intervention,
trajectory analysis -- without the benchmark being a source of actionable
information.

## Repository layout

```
labsafebench/
    environment/      lab state, constraint checks, tools
    agents/            the autonomy-ladder react() agent
    scorers/           capability / safety / intervention / safeguard / horizon
    solvers/           setup solver that initialises LabState per scenario
    tasks/             scenario YAML files + loader
    task.py            the `labsafebench` Inspect Task entrypoint
configs/               named SafeguardConfig YAML files
docs/                  methodology.md, risk_taxonomy.md
scripts/               run_pilot.py
results/               pilot outputs (gitignored logs, checked-in README)
tests/                 offline tests using Inspect's mockllm provider
```

## Status

This is a small, deliberately scoped v0.1 proof-of-concept -- an initial
attempt to extend a chemical-security model-evaluation risk framework into
agentic scientific environments where the system can interact with tools
and infrastructure, not a validated benchmark. See `docs/methodology.md`
for known limitations and planned extensions.

## License

MIT -- see `LICENSE`.
