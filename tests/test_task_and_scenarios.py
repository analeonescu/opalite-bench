"""Tests for the scenario loader and the full `labsafebench` task."""

from __future__ import annotations

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput

from labsafebench.agents.lab_agent import tools_for_level
from labsafebench.task import labsafebench
from labsafebench.tasks.loader import load_scenarios


def test_all_scenarios_load_and_have_required_fields():
    scenarios = load_scenarios()
    assert len(scenarios) == 15
    categories = {s["category"] for s in scenarios}
    assert categories == {"access", "scale", "containment", "oversight", "anomaly", "adaptive"}
    for scenario in scenarios:
        assert scenario["id"]
        assert scenario["prompt"]
        assert isinstance(scenario["risk_vectors"], list) and scenario["risk_vectors"]
        assert "success" in scenario


def test_autonomy_ladder_restricts_tools():
    assert tools_for_level(0) == []
    assert len(tools_for_level(1)) == 1
    assert len(tools_for_level(2)) == 3
    assert len(tools_for_level(3)) == 7
    assert len(tools_for_level(5)) == 7


def test_labsafebench_task_runs_end_to_end_with_mockllm():
    def scripted(input, tools, tool_choice, config):
        # immediately submit -- level 0 (advice) has no tools anyway.
        return ModelOutput.for_tool_call(
            model="mockllm/model", tool_name="submit", tool_arguments={"answer": "advice"}
        )

    logs = inspect_eval(
        labsafebench(autonomy_level=0, categories="access", safeguard_config="unrestricted"),
        model="mockllm/model",
        model_args={"custom_outputs": scripted},
        display="none",
        limit=1,
    )
    log = logs[0]
    assert log.status == "success"
    assert log.samples is not None and len(log.samples) == 1
    sample = log.samples[0]
    # At autonomy level 0 no tools are available, so no experiments run.
    assert sample.scores["capability"].value == "I"
