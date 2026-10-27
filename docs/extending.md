# Extending the eval

Reference for changing the experiment, connecting real hardware, or adding scorers. For
what the repo does and how to run it, see the [README](../README.md).

## Extension points

Every place the simulated scaffold expects to be replaced is marked in the code. To list
them:

```bash
grep -rn "HOOK\[" src configs
```

| Hook | Where | What you plug in |
|---|---|---|
| `HOOK[task-spec]` | `configs/task_spec.yaml` | The whole experiment: brief, deck and labware, substances with their sizes and concentrations, the reactions and their conditions, expected bands per QC stage. The worked example there is **not a real protocol**. |
| `HOOK[calc]` | `lab.deck.Deck.required_ng` | An amount rule other than plain equimolar: vector-to-insert ratios, total-DNA caps, a minimum pipettable volume. |
| `HOOK[robots]` | `lab/robots.py`, `robot:` in `configs/task_spec.yaml` | The arm: how protocols become motion (`ACTION_DIMS`, `to_action`), how an Inspect Robots embodiment is loaded (`-T embodiment=module:factory`), the deck calibration, and the thermocycler device handler. |
| `HOOK[validate]` | `lab/_simulate_worker.py` | What the dry run refuses. Add checks the real robot would make (well capacities, labware heights, liquid classes). |
| `HOOK[safety]` | `lab/robots.py`, `RobotExecutor.execute` | Approval and emergency-stop policy, before anything physical moves. |
| `HOOK[real-lab]` | `task.py`, `build_lab` | Swap in a real human channel and QC source alongside `backend=robots`. |
| `HOOK[human-model]` | `lab/simulated.py` | A simulated technician who is slow, makes mistakes, or refuses. |
| `HOOK[records]` | `trace/tools.py`, `lab/base.py` (`Lab.archive`) | Writes to a LIMS or data system, for taxonomy steps 59-61. |
| `HOOK[alignment]` | `score/alignment.py` | A finer trace-to-step mapping, for example matching individual pipetting commands to steps 9-10, 27-28 and 43-44, or an LLM aligner. |
| `HOOK[judge]` | `score/scorers.py` | LLM-judge scorers. |
| `HumanChannel` | `lab/base.py` | A real technician interface, such as a Slack or web prompt that blocks until they respond. `trace/human_cli.py` is one implementation. |
| `QCSource` | `lab/base.py` | A reader for your QC instrument (gel, fragment analyser, sequencing) returning band sizes in bp. |

## Changing the experiment

Everything specific to this experiment lives in `configs/task_spec.yaml`: the deck layout,
every substance and its concentration, each reaction with its inputs and conditions, the
QC stages and their expected bands, and the prompt text.

`spec/prompts.py` is a skeleton with no Gibson-specific wording, and the stage list is read
from the spec, so a different experiment is a different spec file rather than a code
change. Two things to keep in mind:

- The numbers have to be physically possible. Volumes below the pipette's minimum, or a
  stage that consumes more product than the previous stage can make, will fail in ways
  that look like agent errors.
- `data/ground_truth.csv` needs its `stage` and `evidence_role` columns to match the new
  stage names, or coverage will never credit those steps. `tests/test_taxonomy.py` checks
  that every agent-owned step has both.

## Connecting real hardware

`-T backend=robots` already plays every protocol as arm motion. With the default
`-T embodiment=kinematic` that motion goes to a stand-in arm; to drive a simulator or a
real rig through [Inspect Robots](https://pypi.org/project/inspect-robots/):

1. Write (or install) an Inspect Robots embodiment for your arm whose action is the
   5-D `(x, y, z, plunger_ul, tip)` absolute target in `lab/robots.ACTION_DIMS`, or
   change `to_action` to match the action contract your embodiment declares.
2. Expose a factory and pass it: `-T backend=robots -T embodiment=my_rig.embodiments:make_arm`.
   Inspect Robots asks for hardware-free constructors that connect in `reset()`; the
   executor calls `reset()` once per lab, then `step()` once per waypoint.
3. Calibrate the `robot:` section of `configs/task_spec.yaml` (deck origin, slot pitch,
   safe height, tip length, workspace) and the `labware:` geometry to your deck.
4. Hand thermocycler commands to the real instrument via `RobotExecutor(devices=...)`.
5. Implement a human channel and a QC source for your instruments, and wire them in
   `task.py` (`HOOK[real-lab]`).
6. Replace `configs/task_spec.yaml` with the real experiment, and adjust
   `Deck.required_ng` if your amount rule is not plain equimolar.

Run `python -m opalite_bench doctor --embodiment my_rig.embodiments:make_arm` first: it
checks the spec, that every well is inside the workspace, and that the factory imports,
without constructing it.

What is built:

- **A physical gate.** An embodiment is treated as hardware unless it declares
  `physical = False`, and is refused without `-T allow_physical=true`. That flag is the
  operator's say-so for this host, not an authorisation from the facility.
- **Device claims.** Each run takes an exclusive lock on the device (`DeviceClaim`, keyed
  by the embodiment reference), so a second eval on the same host gets "device busy"
  instead of driving the same arm. POSIX only.
- **Halt on fault.** A fault on the arm halts it. Every later run is refused until the
  agent asks the technician to `review` it, which clears the fault and resets the
  embodiment.

**Before an embodiment touches hardware**: `RobotExecutor` calls `step()` directly, so
Inspect Robots' own approver chain (bounds clamp, per-step delta limit) does not run.
The workspace check in `RobotExecutor.check` and the embodiment's own hard clamp are the
only motion guards. It still needs per-step delta limits, deck-state checks (is the
plate really there?), and an emergency-stop path (`HOOK[safety]`). None of that is built.

## Known gaps

Roughly in priority order.

1. **Real ground truth for QC.** In the simulation, QC outcomes are computed from well
   contents. On real hardware you need a human label (pass / fail / ambiguous per QC
   stage) stored with the sample, so `decision_integrity` has something to score against.
2. **Plan and record judges.** Taxonomy steps 7, 25 and 41 ask whether the
   fragment-to-assembly plan is right; step 62 asks whether the final record is faithful
   to what happened. Both need an LLM judge reading against the event log. Neither is
   scored today.
3. **Intervention-type scoring.** The `human_intervention` column distinguishes approval
   from verification (steps 40, 57, 58). The agent should arguably have to request
   `approve` before committing to a result. Only `review` is scored at present.
   `-T human=cli` now records intervention types and timings, so the data exists.
4. **A technician who can fail.** The default simulated technician always succeeds,
   instantly. Making it slow, error-prone or occasionally non-compliant
   (`HOOK[human-model]`) would test whether the agent notices.
5. **The `n*` taxonomy steps** (barcode scanning). Add a `scan_barcode` tool and a scanner
   channel, then flip those rows to `y`; coverage picks them up once the CSV lists them.
6. **A technician who cannot photograph the gel.** The default technician always takes
   the picture when asked. `-T human=cli` already allows a real refusal, and the three
   outcomes (never asked / asked and refused / provided) are recorded, but no scenario
   injects a missing photograph automatically.
7. **A more realistic gel.** The rendered gel has sharp bands and mild noise. Real gels
   have smiling, overloading and uneven illumination. This matters most for `image_vlm`,
   where a model might read the synthetic gel well and a photograph badly.
8. **`qc_tools=internet`.** Wired up but refuses to run: it would execute agent-written
   code with network access, which needs a sandbox.
9. **Sequence screening.** The restricted insert is restricted because the spec says so.
   A real lab screens sequences before synthesis or use. A screening step (for example,
   Common Mechanism, as Inspect Labs' `commec` plugin does) would make "is this material
   cleared?" something the agent has to find out, not something it is told.
10. **Tamper-evident evidence.** Rescoring trusts the `.eval` file. Inspect Labs writes a
    companion file with hashes binding the log to instrument records and robot logs; the
    same would let a rescore detect an edited log.
