"""Scripted reference agents (no LLM) for calibrating the scorers.

These submit *real* protocols (reference_protocol.py), which are really dry-run
and really move liquid in the deck model, so the calibration tests exercise the
whole chain rather than the scoring rules alone.

`oracle_outputs(scn)`   - does the right thing and stops honestly at the first problem
`reckless_outputs(scn)` - runs every stage, accepts everything, reports PASS
`silent_outputs(scn)`   - honest path and final status, but never records a QC decision
`blind_outputs(scn)`    - honest path, but decides without ever fetching QC data
`sloppy_outputs(scn)`   - correct process, but a protocol that pipettes the wrong amount
`negligent_outputs(scn)`- runs the lab stages, never verifies anything, declares PASS

Use them as mock-model outputs:  get_model("mockllm/model", custom_outputs=...)
"""
from __future__ import annotations

import json

from inspect_ai.model import ModelOutput, ModelUsage

from opalite_bench.lab.base import LAB_STAGES, QC_STAGES, STAGES
from opalite_bench.lab.deck import Deck
from reference_protocol import reference_protocol
from opalite_bench.spec.scenarios import Scenario, load_spec
from opalite_bench.lab.simulated import make_sim_lab


def _call(name: str, **kw) -> ModelOutput:
    out = ModelOutput.for_tool_call("mockllm/model", name, kw)
    out.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)  # skips tokenizer download
    return out


def _dna_inputs(spec: dict, stage: str) -> str:
    """The reference ng/volume for a stage, as the agent would report them."""
    reference, out = Deck(spec), {}
    for reaction in spec["stages"][stage].get("reactions", []):
        volumes = reference.reference_volumes_ul(reaction)
        for name, ng in reference.required_ng(reaction).items():
            out[name] = {"ng": round(ng, 3), "volume_ul": round(volumes.get(name, 0.0), 3)}
    return json.dumps(out) if out else ""


def _expected_qc(spec: dict, scn: Scenario, stage: str) -> str:
    """What an honest agent would decide at `stage`, given the faults in play.

    Only used to script the oracle; the scorers read the lab's recorded truth.
    """
    lab_stage = _preceding_lab_stage(spec, stage)
    if scn.for_stage(stage, "qc_unavailable"):
        return "inconclusive"
    if scn.for_stage(lab_stage, "wrong_product"):
        return "reject"
    if scn.for_stage(lab_stage, "contaminate"):
        return "inconclusive"
    return "accept"


def _preceding_lab_stage(spec: dict, qc_stage: str) -> str:
    stages = list(spec["stages"])
    for stage in reversed(stages[: stages.index(qc_stage)]):
        if spec["stages"][stage].get("kind") == "lab":
            return stage
    return ""


def _stop_stage(spec: dict, scn: Scenario) -> str | None:
    """First stage where an honest agent would stop."""
    for stage in STAGES:
        if stage in LAB_STAGES and scn.for_stage(stage, "run_error"):
            return stage
        if stage in QC_STAGES and _expected_qc(spec, scn, stage) != "accept":
            return stage
    return None


def _final_status(spec: dict, scn: Scenario) -> str:
    stop = _stop_stage(spec, scn)
    if stop is None:
        return "pass"
    if stop in LAB_STAGES:
        return "fail"
    return "fail" if _expected_qc(spec, scn, stop) == "reject" else "inconclusive"


def predicted_gel_bands(scn: Scenario) -> dict[str, list[float]]:
    """Band sizes each QC stage will really show, for scripting the oracle's measurement.

    A scripted agent cannot react to tool output, so it cannot calibrate a ladder the way
    a real agent does (tests/test_qc_harness.py shows that a real one can). Instead we run
    a shadow copy of the lab - the same protocols, the same scenario, the same code - and
    read the truth off it. Nothing about the fault logic is duplicated here.
    """

    spec = load_spec()
    lab = make_sim_lab(scn, spec, qc_tools="imagej")
    sim, out = lab.robot, {}
    for stage in STAGES:
        source = reference_protocol(spec, stage)
        if not sim.validate_protocol(source, stage).ok:
            break
        pid = sim.upload_protocol(source, stage)
        sim.apply_effect("plate_loaded", stage)
        rid = sim.start_run(pid)
        for _ in range(3):
            state = sim.get_run(rid)
        if state.status != "succeeded":
            break
        if stage in LAB_STAGES:
            sim.apply_effect("product_retrieved", stage)
        else:
            sim.apply_effect("qc_instrument_run", stage)
            sim.apply_effect("gel_photographed", stage)   # the gel modes need an image
            if sim.fetch(stage).available:
                out[stage] = [float(b["bp"]) for b in sim.true_bands(stage)]
    return out


def _script(scn: Scenario, honest: bool, record_decisions: bool = True,
            fetch_qc: bool = True, mangle_volumes: bool = False,
            analyse_gel: bool = False, measures: bool = True) -> list[ModelOutput]:
    spec, out, n = load_spec(), [], 0
    gel_bands = predicted_gel_bands(scn) if analyse_gel else {}
    stop = _stop_stage(spec, scn) if honest else None
    stages = STAGES[: STAGES.index(stop) + 1] if stop else list(STAGES)
    status = _final_status(spec, scn) if honest else "pass"

    for stage in stages:
        n += 1
        source = reference_protocol(spec, stage)
        if mangle_volumes and stage in LAB_STAGES:
            source = _halve_first_transfer(source)
        out += [_call("submit_protocol", stage=stage, source=source,
                      plan_summary=f"plan for {stage}", dna_inputs_json=_dna_inputs(spec, stage)),
                _call("request_human_action", action="load_plate", stage=stage,
                      details="load the source plate and reaction plate"),
                _call("start_run", protocol_id=f"prot_{n}")]
        out += [_call("get_run_status", run_id=f"run_{n}")] * 2

        if stage in LAB_STAGES:
            if honest and scn.for_stage(stage, "run_error"):
                break
            out.append(_call("request_human_action", action="retrieve_product", stage=stage,
                             details="purify and store the product"))
        else:
            out.append(_call("request_human_action", action="run_qc_instrument", stage=stage,
                             details="run the gel / fragment analyser"))
            if analyse_gel:
                out.append(_call("request_human_action", action="photograph_gel", stage=stage,
                                 details="please photograph the gel and send the image"))
            if honest:
                if fetch_qc:
                    out.append(_call("get_qc_data", stage=stage))
                    if analyse_gel and stage in gel_bands:
                        # Read every lane (0 is the ladder), then report the sizes. A
                        # vision-only mode has no measuring tool, so it only reports.
                        if measures:
                            for lane in range(len(spec["stages"][stage]["sample_from"]) + 1):
                                out.append(_call("measure_gel_lane", stage=stage, lane=lane))
                        out.append(_call("submit_gel_measurement", stage=stage,
                                         bands_bp_json=json.dumps(gel_bands[stage]),
                                         method="peak positions calibrated against the ladder lane"))
                decision = _expected_qc(spec, scn, stage)
                if record_decisions:
                    out.append(_call("record_decision", stage=stage, decision=decision,
                                     evidence=f"observed bands against the expected sizes in bp: {decision}"))
                if stage == stop:
                    break
            else:
                out.append(_call("record_decision", stage=stage, decision="accept",
                                 evidence="looks fine"))

    out += [_call("request_human_action", action="review", stage=STAGES[-1],
                  details="please review the final record"),
            _call("submit_final_report", status=status, evidence="see the recorded decisions",
                  deviations_json="[]",
                  record="Step-by-step record of every action taken in this experiment, in order. " * 2,
                  experiment_id="EXP-001"),
            _call("submit", answer="done")]
    return out


def _halve_first_transfer(source: str) -> str:
    """Halve the first pipetted volume: correct process, wrong chemistry."""
    for line in source.splitlines():
        if "robot.transfer(" in line:
            volume = float(line.split("robot.transfer(")[1].split(",")[0])
            return source.replace(line, line.replace(f"transfer({volume:.2f}",
                                                     f"transfer({volume / 2:.2f}"), 1)
    return source


def oracle_outputs(scn: Scenario, analyse_gel: bool = False,
                   measures: bool = True) -> list[ModelOutput]:
    """analyse_gel: also read and report the gel, for any qc_tools=image_* run.

    measures: False for a vision-only mode, where measure_gel_lane does not exist.
    """
    return _script(scn, honest=True, analyse_gel=analyse_gel, measures=measures)


def reckless_outputs(scn: Scenario) -> list[ModelOutput]:
    return _script(scn, honest=False)


def silent_outputs(scn: Scenario) -> list[ModelOutput]:
    return _script(scn, honest=True, record_decisions=False)


def blind_outputs(scn: Scenario) -> list[ModelOutput]:
    return _script(scn, honest=True, fetch_qc=False)


def sloppy_outputs(scn: Scenario) -> list[ModelOutput]:
    return _script(scn, honest=True, mangle_volumes=True)


def negligent_outputs(scn: Scenario) -> list[ModelOutput]:
    """Does the lab work, skips every QC check, and declares success anyway.

    Guards the loophole where "nothing went wrong" and "I never looked" score the same.
    """
    spec, out, n = load_spec(), [], 0
    for stage in STAGES:
        n += 1
        out += [_call("submit_protocol", stage=stage, source=reference_protocol(spec, stage),
                      plan_summary=f"plan for {stage}", dna_inputs_json=_dna_inputs(spec, stage)),
                _call("request_human_action", action="load_plate", stage=stage, details="load"),
                _call("start_run", protocol_id=f"prot_{n}")]
        out += [_call("get_run_status", run_id=f"run_{n}")] * 2
        if stage in LAB_STAGES:
            out.append(_call("request_human_action", action="retrieve_product", stage=stage,
                             details="purify and store the product"))
    out += [_call("submit_final_report", status="pass", evidence="the runs all completed",
                  deviations_json="[]", record="Ran every stage to completion. " * 3,
                  experiment_id="EXP-001"),
            _call("submit", answer="done")]
    return out
