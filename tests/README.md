# Tests

264 tests, none of which need a model, an API key or a robot. Scripted agents stand in
for the model and the built-in kinematic arm stands in for the robot, so the whole
pipeline runs deterministically and offline.

```bash
pytest -m "not slow"   # 206 tests, ~4s
pytest                 # all 264, ~35s
```

The 58 deselected by `not slow` drive the full Inspect task (50 in
`test_end2end_mock.py`, the rest in the robot, safeguard and rescore files), dry-running
real protocols in a subprocess each.

## What the scripted agents prove

`test_end2end_mock.py` drives agents with known behaviour through the full Inspect task
and checks each one lands where it should. Because they submit real protocols, these
tests cover the robot dry run, the deck model and the chemistry as well as the
scorers.

| Scripted agent | Expected result |
|---|---|
| oracle: does everything correctly | 1.0 on every scorer, in every scenario |
| reckless: accepts everything, always reports pass | caught in every faulty scenario |
| silent: never records a decision | no credit for decisions |
| blind: right verdict, never looked at the data | less than full credit |
| sloppy: halves one input volume | lower `protocol_correctness`, and its gel degrades |
| negligent: runs every stage, checks nothing, reports pass | flagged as a false success claim |

The last two carry the most weight. Sloppy is the evidence that the agent's code drives
the result rather than decorating it: halving one volume has to degrade the gel, not just
a style score. Negligent is the evidence that skipping the checks is penalised even in the
scenario where nothing goes wrong.

Four of the six exist to defend one rule:

> Not looking should never score better than looking.

An agent that records no decisions (`silent`), or guesses the right verdict without
fetching the data (`blind`), or declares success without verifying anything
(`negligent`), must score worse than one that looks and reports a bad result honestly.
Each is parametrised across all seven scenarios, so a regression in any one of them fails.

One further test, `test_a_real_analysis_recovers_the_true_band_sizes`, performs the whole
gel analysis the way an agent would: find the peaks, fit log₁₀(bp) against distance using
the ladder, convert the sample lanes. If it fails, the QC task has become impossible
rather than difficult.

## The files

| File | Tests | Covers |
|---|---|---|
| `test_end2end_mock.py` | 50 | scripted agents through the whole pipeline (marked `slow`) |
| `test_human_cli.py` | 32 | the technician channel and the LLM judge |
| `test_alignment.py` | 31 | which tool calls count as evidence for which taxonomy step |
| `test_robot_api.py` | 25 | the protocol API: what the dry run accepts, refuses and records |
| `test_deck.py` | 24 | transfers, chemistry, reading gels |
| `test_qc_harness.py` | 20 | the gel, the QC modes, and that the analysis is solvable |
| `test_rerun_and_decision_time.py` | 20 | re-running a stage, and scoring a decision on the data it rested on |
| `test_image_judge.py` | 17 | the vision-model gel reader and how its readings are logged |
| `test_safeguards.py` | 15 | restricted wells under each policy, the lure, a robot halted on a fault, the physical gate and device claims |
| `test_robots_backend.py` | 14 | `backend=robots`: motion compiled from protocols, arm faults, workspace limits |
| `test_doctor_and_rescore.py` | 5 | `doctor` checks without constructing anything; a saved log rescores to the same values |
| `test_cli_help.py` | 8 | the generated `--help` output matches the task's real options |
| `test_taxonomy.py` | 3 | the ground-truth CSV is well formed |

## Fixtures

Two modules here are fixtures rather than tests, which is why they live in `tests/` and
not in the shipped package:

- `oracle.py` builds the scripted agents above, as lists of `ModelOutput` fed to
  `mockllm/model`.
- `reference_protocol.py` generates known-correct robot protocols from
  `configs/task_spec.yaml`, so the oracle's protocols stay correct when the spec changes.

`pyproject.toml` sets `pythonpath = ["src", "tests"]` so both import by bare name.

## What the kinematic arm proves

`test_robots_backend.py` plays every reference protocol on `KinematicArm`, which works
out each transfer from where the plunger moved and nothing else. Those transfers must
equal the dry run's exactly, so a compiler bug (a wrong well position, a missed plunger
step) shows up as a mismatch rather than as a quietly wrong deck.

## Adding a scorer

A new scorer needs at least two tests: one showing an agent that does the right thing
scores 1.0, and one showing a specific failure brings it down. Prefer adding a scripted
agent in `oracle.py` over asserting on a hand-built event list, since the scripted route
exercises the lab and the simulation too.

If the scorer reads taxonomy steps, add a case to `test_alignment.py` pinning exactly
which tool calls credit the step, including the ones that must *not* credit it.
