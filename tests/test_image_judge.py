"""`qc_tools=image_judge`: a separate vision model reads the gel for the agent.

What matters here is that the judge is measured rather than trusted. Its reading
becomes the agent's evidence, so a misreading judge must show up as a judge failure
and not as an agent that decided badly.
"""
import asyncio
import json

import pytest
from inspect_ai.model import ModelOutput, get_model

from opalite_bench.lab.qc_harness import GelJudgeHarness, build_harness
from opalite_bench.lab.simulated import make_sim_lab
from opalite_bench.spec.scenarios import load_scenarios, load_spec

SPEC = load_spec()
SCN = load_scenarios()["clean_success"]


def judge_saying(payload, n=10):
    text = json.dumps(payload) if isinstance(payload, dict) else payload
    return get_model("mockllm/model",
                     custom_outputs=[ModelOutput.from_content("mockllm/model", text)] * n)


def lab_with(judge, species="intermediate_1", ng=120.0):
    lab = make_sim_lab(SCN, SPEC, qc_tools="image_judge", image_judge_model=judge)
    sim = lab.robot
    sim.qc_run.add("round1_qc")
    sim.photographed.add("round1_qc")
    sim.deck.well("B2:C1").add(species, ng)
    sim.deck.well("B2:C1").volume_ul = 6.0
    return sim


def read(sim, stage="round1_qc"):
    return asyncio.run(sim.judged_qc(stage))


# ----------------------------------------------------------------- the happy path

def test_the_agent_is_handed_the_judges_band_sizes():
    sim = lab_with(judge_saying({"lanes": [{"lane": 1, "bands": [{"bp": 4680, "rel_intensity": 0.95}]}],
                                 "notes": "one dominant band"}))
    out = read(sim)
    assert [b["bp"] for b in out["bands"]] == [4680.0]
    assert "vision model" in out["read_by"]
    assert out["notes"] == "one dominant band"


def test_the_ladder_lane_is_not_reported_as_a_sample():
    """Lane 0 is the reference, not a result."""
    sim = lab_with(judge_saying({"lanes": [
        {"lane": 0, "bands": [{"bp": 10000, "rel_intensity": 0.5}]},
        {"lane": 1, "bands": [{"bp": 4700, "rel_intensity": 0.9}]}]}))
    assert [b["bp"] for b in read(sim)["bands"]] == [4700.0]


def test_the_reading_is_logged_beside_the_truth():
    sim = lab_with(judge_saying({"lanes": [{"lane": 1, "bands": [{"bp": 4680, "rel_intensity": 0.9}]}]}))
    read(sim)
    ev = [e for e in sim.log.events if e.name == "image_judge_reading"]
    assert len(ev) == 1
    assert ev[0].data["reported"] and ev[0].data["truth"]
    assert ev[0].data["ok"] is True


# ------------------------------------------------------------- the judge failing

@pytest.mark.parametrize("reply", ["", "I can't see the image", "{not json}", "[]"])
def test_an_unreadable_judge_is_a_service_failure_not_empty_data(reply):
    """The agent must be told the service failed, not handed a confident empty result."""
    sim = lab_with(judge_saying(reply))
    out = read(sim)
    assert "error" in out and "bands" not in out
    ev = [e for e in sim.log.events if e.name == "image_judge_reading"][0]
    assert ev.data["ok"] is False and ev.data["error"]


def test_a_judge_that_raises_does_not_kill_the_run():
    class Broken:
        name = "broken"

        async def generate(self, *a, **k):
            raise RuntimeError("vision service down")

    sim = lab_with(Broken())
    out = read(sim)
    assert "error" in out
    ev = [e for e in sim.log.events if e.name == "image_judge_reading"][0]
    assert ev.data["ok"] is False and "RuntimeError" in ev.data["error"]


def test_malformed_bands_are_dropped_not_crashed():
    sim = lab_with(judge_saying({"lanes": [{"lane": 1, "bands": [
        {"bp": "wide"}, {"rel_intensity": 0.5}, {"bp": 4700, "rel_intensity": 0.9}]}]}))
    assert [b["bp"] for b in read(sim)["bands"]] == [4700.0]


# ------------------------------------------------------------------ the wiring

def test_the_agent_gets_no_gel_tools_of_its_own():
    """The judge does the looking, so the agent is not asked to analyse."""
    h = build_harness("image_judge")
    assert isinstance(h, GelJudgeHarness)
    assert h.requires_analysis is False and h.measures is False and h.shows_image is False


def test_the_gel_still_has_to_be_photographed_first():
    """Running a gel and having a picture of it are different, judge or no judge."""
    lab = make_sim_lab(SCN, SPEC, qc_tools="image_judge",
                       image_judge_model=judge_saying({"lanes": []}))
    sim = lab.robot
    sim.qc_run.add("round1_qc")                      # run, but not photographed
    r = sim.fetch("round1_qc")
    assert not r.available and "photographed" in r.message


def test_the_task_adds_a_qc_quality_scorer():
    from opalite_bench.task import gibson_analysis
    assert len(gibson_analysis(scenarios="clean_success", qc_tools="image_judge").scorer) == 8
    assert len(gibson_analysis(scenarios="clean_success", qc_tools="summary").scorer) == 7


def test_qc_quality_penalises_misses_and_inventions_alike():
    """One number has to move in both directions, or it is not worth having."""
    from opalite_bench.score.scorers import _match_bands

    def f1(reported, truth, tol=0.05):
        m = _match_bands(reported, truth, tol)
        p = m / len(reported) if reported else 0.0
        r = m / len(truth) if truth else 1.0
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    perfect = f1([4700.0], [4700.0])
    missed = f1([], [4700.0])
    invented = f1([4700.0, 900.0], [4700.0])
    assert perfect == 1.0
    assert missed == 0.0
    assert 0.0 < invented < perfect, "an invented band must cost something"


def test_image_judge_is_an_offered_mode():
    from opalite_bench.lab.qc_harness import MODES
    assert "image_judge" in MODES


# ------------------------------------------------------ which model reads the gel

def test_the_default_reader_is_a_dedicated_vision_model():
    """Not the model under test: the reading should not vary with the agent."""
    from opalite_bench.lab.qc_harness import DEFAULT_IMAGE_JUDGE

    assert DEFAULT_IMAGE_JUDGE == "openai/gpt-5"
    assert build_harness("image_judge")._judge_model == DEFAULT_IMAGE_JUDGE


def test_the_default_can_be_overridden():
    h = build_harness("image_judge", judge_model="anthropic/claude-opus-5")
    assert h._judge_model == "anthropic/claude-opus-5"


def test_building_the_task_needs_no_key_for_the_judge():
    """The judge is resolved lazily, on the first gel, so -T validation stays offline."""
    from opalite_bench.task import gibson_analysis

    task = gibson_analysis(scenarios="clean_success", qc_tools="image_judge")
    assert len(task.scorer) == 8
