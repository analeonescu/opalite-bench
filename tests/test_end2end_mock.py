"""End-to-end through Inspect with scripted (mock) models: calibrates the scorers.

These run the whole chain - real robot dry run, real deck model, real chemistry -
so they also catch breakage in the lab itself, not only in the scoring rules.
"""
import re

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model

from oracle import (blind_outputs, oracle_outputs, reckless_outputs, silent_outputs,
                             sloppy_outputs)
from opalite_bench.spec.scenarios import load_scenarios
from opalite_bench.task import gibson_analysis
import asyncio
from oracle import negligent_outputs
from opalite_bench.lab.base import HumanResponse
from opalite_bench.lab.simulated import AutoTechnician, make_sim_lab
from opalite_bench.spec.scenarios import load_spec

pytestmark = pytest.mark.slow     # these drive the full Inspect pipeline

SCNS = load_scenarios()
FAULTY = [k for k in SCNS if k != "clean_success"]


def run_sample(sid, outputs, tmp_path):
    model = get_model("mockllm/model", custom_outputs=outputs)
    log = inspect_eval(gibson_analysis(scenarios=sid), model=model, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error
    return log.samples[0]


def run(sid, outputs, tmp_path):
    return {k: v.value for k, v in run_sample(sid, outputs, tmp_path).scores.items()}


# ------------------------------------------------------------------ the honest agent

@pytest.mark.parametrize("sid", list(SCNS))
def test_oracle_scores_perfectly(sid, tmp_path):
    """The reference agent writes real protocols: everything must come out clean."""
    s = run(sid, oracle_outputs(SCNS[sid]), tmp_path)
    assert all(v == 1.0 for v in s["decision_integrity"].values()), s["decision_integrity"]
    assert s["taxonomy_coverage"]["step_coverage"] == 1.0, s["taxonomy_coverage"]
    assert s["handoff_behaviour"]["no_precondition_violation"] == 1.0
    assert s["calculation_accuracy"]["dna_inputs_correct"] == 1.0
    assert s["protocol_quality"]["valid_protocol_for_each_needed_stage"] == 1.0


@pytest.mark.parametrize("sid", list(SCNS))
def test_oracle_protocols_do_the_right_thing_on_the_deck(sid, tmp_path):
    """Reference protocols must deliver the right amounts under the right conditions."""
    s = run(sid, oracle_outputs(SCNS[sid]), tmp_path)["protocol_correctness"]
    assert s["inputs_delivered_correctly"] == 1.0, s
    assert s["reagents_added"] == 1.0 and s["conditions_met"] == 1.0, s


# ------------------------------------------------------------------ dishonest agents

@pytest.mark.parametrize("sid", FAULTY)
def test_reckless_is_caught(sid, tmp_path):
    s = run(sid, reckless_outputs(SCNS[sid]), tmp_path)["decision_integrity"]
    assert s["no_false_success_claim"] == 0.0
    assert s["final_status_correct"] == 0.0
    assert s["decision_accuracy"] < 1.0


@pytest.mark.parametrize("sid", list(SCNS))
def test_recording_no_decisions_earns_no_decision_credit(sid, tmp_path):
    """Staying silent must not beat recording decisions (a missing decision counts as wrong)."""
    s = run(sid, silent_outputs(SCNS[sid]), tmp_path)["decision_integrity"]
    assert s["decision_accuracy"] == 0.0, s


@pytest.mark.parametrize("sid", list(SCNS))
def test_decisions_are_judged_on_what_the_agent_saw(sid, tmp_path):
    """Guessing the right verdict without ever fetching QC data is not a correct decision."""
    s = run(sid, blind_outputs(SCNS[sid]), tmp_path)["decision_integrity"]
    assert s["decision_accuracy"] < 1.0, s


# --------------------------------------------------- the agent whose chemistry is wrong

def test_a_wrong_volume_is_caught_and_changes_the_result(tmp_path):
    """The point of simulating for real: bad pipetting must show up in the data, not just
    in a style score. Halving one input should cost protocol_correctness *and* spoil QC."""
    sample = run_sample("clean_success", sloppy_outputs(SCNS["clean_success"]), tmp_path)
    scores = {k: v.value for k, v in sample.scores.items()}

    assert scores["protocol_correctness"]["inputs_delivered_correctly"] < 1.0
    assert scores["protocol_correctness"]["reaction_complete"] < 1.0

    verdicts = [e["data"]["verdict"] for e in sample.store["events"]
                if e["kind"] == "sim" and e["name"] == "qc_truth"]
    assert verdicts and any(v != "pass" for v in verdicts), \
        f"wrong amounts should show up in the QC data, got {verdicts}"


def test_protocol_source_is_logged(tmp_path):
    """The code the agent wrote has to be in the trace to be auditable."""
    sample = run_sample("clean_success", oracle_outputs(SCNS["clean_success"]), tmp_path)
    sources = [e["data"].get("source") for e in sample.store["events"]
               if e["name"] == "submit_protocol"]
    assert sources and all(s and "def run(" in s for s in sources)


# ------------------------------------------------------------------------- evidence

def test_coverage_evidence_points_at_real_events(tmp_path):
    sample = run_sample("clean_success", oracle_outputs(SCNS["clean_success"]), tmp_path)
    cov = sample.scores["taxonomy_coverage"].metadata
    events = sample.store["events"]
    assert cov["evidence"], "credited steps should be listed with their reasons"
    for step, why in cov["evidence"].items():
        m = re.search(r"\[event (\d+): (\w+)\]$", why)
        assert m, (step, why)
        assert events[int(m.group(1))]["name"] == m.group(2), (step, why)


def test_every_step_carries_a_reason_on_both_sides_of_the_ledger(tmp_path):
    """Coverage has to be readable as prose, not as a set of ids: every expected step
    says why it was expected, and every uncredited one says what would have credited it."""
    sample = run_sample("round2_run_error", reckless_outputs(SCNS["round2_run_error"]), tmp_path)
    score = sample.scores["taxonomy_coverage"]
    cov = score.metadata
    # metadata round-trips through JSON, so the step-id keys come back as strings
    assert {str(i) for i in cov["missing"]} == set(cov["missing_because"]), \
        "every gap needs an explanation"
    assert set(cov["evidence"]) | set(cov["missing_because"]) >= set(cov["expected_because"])
    assert all(r.startswith("needs ") or "evidence_role" in r
               for r in cov["missing_because"].values())
    assert "NOT EVIDENCED" in score.explanation and "EVIDENCED" in score.explanation


def test_claiming_pass_without_verifying_is_a_false_success_claim(tmp_path):
    """Running every stage but checking nothing must not score as a clean success,
    even in the scenario where nothing actually goes wrong."""
    s = run("clean_success", negligent_outputs(SCNS["clean_success"]), tmp_path)["decision_integrity"]
    assert s["no_false_success_claim"] == 0.0, s
    assert s["decision_accuracy"] == 0.0, s


# ---------------------------------------------------- QC analysis (qc_tools=image_*)

def run_gel(sid, outputs, tmp_path, qc_tools="image_manual"):
    model = get_model("mockllm/model", custom_outputs=outputs)
    log = inspect_eval(gibson_analysis(scenarios=sid, qc_tools=qc_tools), model=model,
                       log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error
    return {k: v.value for k, v in log.samples[0].scores.items()}


@pytest.mark.parametrize("sid", ["clean_success", "fragment_qc_fail", "final_wrong_size"])
def test_oracle_analyses_the_gel_correctly(sid, tmp_path):
    s = run_gel(sid, oracle_outputs(SCNS[sid], analyse_gel=True), tmp_path)
    assert s["qc_analysis"]["analysed_every_stage"] == 1.0, s["qc_analysis"]
    assert s["qc_analysis"]["measurement_accuracy"] == 1.0, s["qc_analysis"]
    assert all(v == 1.0 for v in s["decision_integrity"].values()), s["decision_integrity"]


def test_skipping_the_analysis_is_penalised(tmp_path):
    """Deciding from a gel without ever measuring it earns no analysis credit."""
    s = run_gel("clean_success", oracle_outputs(SCNS["clean_success"], analyse_gel=False),
                   tmp_path)
    assert s["qc_analysis"]["analysed_every_stage"] == 0.0, s["qc_analysis"]
    assert s["qc_analysis"]["measurement_accuracy"] == 0.0, s["qc_analysis"]


def test_analysis_is_required_for_the_analyse_taxonomy_steps(tmp_path):
    """With a gel, steps 23/39/55 need a measurement, not just a fetch."""
    measured = run_gel("clean_success", oracle_outputs(SCNS["clean_success"], analyse_gel=True),
                          tmp_path)
    assert measured["taxonomy_coverage"]["step_coverage"] == 1.0


def test_a_vision_mode_puts_the_gel_in_front_of_the_model(tmp_path):
    """image_vlm has no measuring tool, so the picture has to reach the model itself."""
    model = get_model("mockllm/model",
                      custom_outputs=oracle_outputs(SCNS["clean_success"], analyse_gel=True,
                                                    measures=False))
    log = inspect_eval(gibson_analysis(scenarios="clean_success", qc_tools="image_vlm"),
                       model=model, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error

    sample = log.samples[0]
    images = [c for m in sample.messages for c in (m.content if isinstance(m.content, list) else [])
              if getattr(c, "type", None) == "image"]
    assert images, "the gel image should be attached to the QC tool result"

    # Inspect stores large content out of line, so the reference is either the data URI
    # itself or an attachment id pointing at it.
    reference = images[0].image
    resolved = (sample.attachments or {}).get(reference.removeprefix("attachment://"), reference)
    assert resolved.startswith("data:image/png;base64,"), resolved[:60]

    scores = {k: v.value for k, v in log.samples[0].scores.items()}
    assert scores["qc_analysis"]["analysed_every_stage"] == 1.0, scores["qc_analysis"]
    assert scores["qc_analysis"]["measurement_accuracy"] == 1.0, scores["qc_analysis"]


def test_the_measuring_mode_does_not_send_an_image(tmp_path):
    """image_manual is the numeric task: no picture, so a text-only model can do it."""
    model = get_model("mockllm/model",
                      custom_outputs=oracle_outputs(SCNS["clean_success"], analyse_gel=True))
    log = inspect_eval(gibson_analysis(scenarios="clean_success", qc_tools="image_manual"),
                       model=model, log_dir=str(tmp_path), display="none")[0]
    images = [c for m in log.samples[0].messages for c in (m.content if isinstance(m.content, list) else [])
              if getattr(c, "type", None) == "image"]
    assert not images


# ------------------------------------------------- asking for the gel photograph

def test_there_is_no_image_until_the_technician_photographs_the_gel(tmp_path):
    """Running the gel is not the same as having a picture of it."""

    lab = make_sim_lab(SCNS["clean_success"], load_spec(), qc_tools="image_manual")
    sim = lab.robot
    sim.apply_effect("qc_instrument_run", "fragment_qc")

    before = sim.fetch("fragment_qc")
    assert not before.available and "photograph_gel" in before.message

    sim.apply_effect("gel_photographed", "fragment_qc")
    assert sim.fetch("fragment_qc").available

    # Both outcomes are recorded, so "we never got a picture" is visible afterwards.
    records = [e.data for e in lab.log.events if e.name == "gel_image"]
    assert [r["provided"] for r in records] == [False, True]
    assert all(r["analysis"] == "image_manual" for r in records)


def test_a_gel_cannot_be_photographed_before_it_is_run(tmp_path):

    sim = make_sim_lab(SCNS["clean_success"], load_spec(), qc_tools="image_manual").robot
    response = sim.apply_effect("gel_photographed", "fragment_qc")
    assert not response.ok and "no gel to photograph" in response.message


def test_never_asking_for_the_image_counts_against_the_agent(tmp_path):
    """An agent that skips the request gets no analysis credit: that is its own doing."""
    s = run_gel("clean_success", oracle_outputs(SCNS["clean_success"], analyse_gel=False),
                tmp_path)
    assert s["qc_analysis"]["analysed_every_stage"] == 0.0, s["qc_analysis"]


def test_asking_and_being_refused_is_recorded_differently_from_not_asking():
    """Only one of these is the agent's fault, so the scorer has to tell them apart."""


    class NoCamera(AutoTechnician):
        async def handle(self, action, stage, details, lab):
            if action == "photograph_gel":
                return HumanResponse(False, "the gel doc camera is broken")
            return await super().handle(action, stage, details, lab)

    def gel_record(technician, ask):
        lab = make_sim_lab(SCNS["clean_success"], load_spec(), qc_tools="image_manual",
                           technician=technician)
        asyncio.run(lab.robot.request("run_qc_instrument", "fragment_qc", ""))
        if ask:
            asyncio.run(lab.robot.request("photograph_gel", "fragment_qc", ""))
        lab.robot.fetch("fragment_qc")
        return [e.data for e in lab.log.events if e.name == "gel_image"][-1]

    refused = gel_record(NoCamera(), ask=True)
    assert (refused["requested"], refused["provided"]) == (True, False)

    never_asked = gel_record(None, ask=False)
    assert (never_asked["requested"], never_asked["provided"]) == (False, False)
