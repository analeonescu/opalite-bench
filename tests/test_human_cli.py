"""Tests for the human-in-the-loop channel: the judge, the timing, and the log.

The terminal reader is injected, so none of these touch stdin. What matters most is that
the judge fails *closed*: an unreadable or ambiguous report must never let the lab act as
though a physical step had happened.
"""
import asyncio
import json
from datetime import datetime, timezone

import pytest
from inspect_ai.model import ModelOutput, get_model

from opalite_bench.lab.simulated import SimulatedLab, make_sim_lab
from opalite_bench.trace.human_cli import CliTechnician, Interpretation, InterventionLog, parse_judgement
from opalite_bench.spec.scenarios import load_scenarios, load_spec
import io
import sys
import threading
from opalite_bench.paths import ROOT
from opalite_bench.trace.human_cli import intervention_log_path, read_report_from_terminal

EFFECTS = SimulatedLab.EFFECTS
SPEC = load_spec()
SCN = load_scenarios()["clean_success"]


def run(coro):
    """Run one coroutine. Avoids needing an async pytest plugin."""
    return asyncio.run(coro)


def judge_returning(payload) -> object:
    """A mock model whose reply is `payload` (a dict is sent as JSON, a string verbatim)."""
    text = json.dumps(payload) if isinstance(payload, dict) else payload
    return get_model("mockllm/model", custom_outputs=[
        ModelOutput.from_content("mockllm/model", text) for _ in range(20)])


def technician(payload, report="Loaded both plates, nothing unusual.", tmp_path=None):
    log = InterventionLog(path=(tmp_path / "human.jsonl") if tmp_path else "/dev/null",
                          run_id="test-run")
    return CliTechnician(judge_model=judge_returning(payload), log=log,
                         reader=lambda action, stage, details: report)


# --------------------------------------------------------------------- parsing

def test_a_clean_judgement_is_read_back():
    result = parse_judgement(json.dumps({
        "effect": "plate_loaded", "confident": True, "summary": "both plates on the deck",
        "deviations": ["used slot C2 instead of B2"], "observations": [], "not_checked": ["volumes"],
    }), EFFECTS)
    assert result.effect == "plate_loaded" and result.confident
    assert result.deviations == ["used slot C2 instead of B2"]
    assert result.not_checked == ["volumes"]


def test_json_wrapped_in_prose_is_still_read():
    result = parse_judgement('Sure! {"effect": "qc_instrument_run", "summary": "gel run"} Hope that helps.',
                             EFFECTS)
    assert result.effect == "qc_instrument_run"


@pytest.mark.parametrize("reply", ["", "I think the plate is loaded?", "{not json}",
                                   '{"effect": "launch_rocket", "summary": "x"}'])
def test_anything_unreadable_fails_closed(reply):
    """An unparseable or out-of-range judgement must not advance the lab."""
    result = parse_judgement(reply, EFFECTS)
    assert result.effect == "not_done"
    assert result.confident is False and result.judge_error


def test_scalar_fields_are_tolerated():
    """Judges sometimes return a string where a list was asked for."""
    result = parse_judgement('{"effect": "acknowledged", "summary": "ok", "deviations": "none"}',
                             EFFECTS)
    assert result.deviations == ["none"]


# ------------------------------------------------------------------ interpreting

def test_an_empty_report_is_never_treated_as_done():
    tech = CliTechnician(judge_model=judge_returning({"effect": "plate_loaded", "summary": "x"}),
                         log=InterventionLog(path="/dev/null"), reader=lambda *a: "")
    result = run(tech.interpret("load_plate", "gibson_round1", "", "", EFFECTS))
    assert result.effect == "not_done"


def test_a_judge_that_errors_fails_closed():
    class Broken:
        async def generate(self, *a, **k):
            raise RuntimeError("judge is down")
    tech = CliTechnician(judge_model=Broken(), log=InterventionLog(path="/dev/null"))
    result = run(tech.interpret("load_plate", "gibson_round1", "", "loaded it", EFFECTS))
    assert result.effect == "not_done" and "judge is down" in result.judge_error


# ----------------------------------------------------- the report drives the lab

def test_a_positive_report_changes_the_lab():
    lab = make_sim_lab(SCN, SPEC)
    tech = technician({"effect": "plate_loaded", "summary": "both plates on the deck"})
    response = run(tech.handle("load_plate", "gibson_round1", "load them", lab.robot))
    assert response.ok
    assert "gibson_round1" in lab.robot.loaded


def test_the_technician_can_overrule_the_agents_request():
    """The agent asked for a plate to be loaded; the technician says it did not happen."""
    lab = make_sim_lab(SCN, SPEC)
    tech = technician({"effect": "not_done", "confident": False,
                       "summary": "plate was cracked, could not load it"},
                      report="The round-1 plate arrived cracked. I did not load anything.")
    response = run(tech.handle("load_plate", "gibson_round1", "load them", lab.robot))
    assert not response.ok
    assert "gibson_round1" not in lab.robot.loaded, "an unperformed step must not unblock the run"
    assert "cracked" in response.message


def test_what_the_agent_is_told_carries_the_caveats():
    lab = make_sim_lab(SCN, SPEC)
    tech = technician({"effect": "plate_loaded", "confident": True, "summary": "plates loaded",
                       "deviations": ["used 9 uL, the well ran low"],
                       "not_checked": ["I did not verify well volumes"]})
    response = run(tech.handle("load_plate", "fragment_generation", "", lab.robot))
    assert "9 uL" in response.message and "NOT checked" in response.message


# -------------------------------------------------------------- logging and timing

def test_every_intervention_is_logged_with_its_timing(tmp_path):
    lab = make_sim_lab(SCN, SPEC)
    tech = technician({"effect": "plate_loaded", "summary": "done"},
                      report="Loaded both plates. Took about two minutes.", tmp_path=tmp_path)
    run(tech.handle("load_plate", "gibson_round1", "load the source plate", lab.robot))

    rows = [json.loads(line) for line in (tmp_path / "human.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == "test-run" and row["stage"] == "gibson_round1"
    assert row["requested_action"] == "load_plate" and row["applied"] == "plate_loaded"
    assert "Loaded both plates" in row["report"]
    assert isinstance(row["seconds"], float) and row["logged_at"]


def test_interventions_also_reach_the_eval_event_log(tmp_path):
    """So intervention types and times can be scored alongside everything else."""
    lab = make_sim_lab(SCN, SPEC)
    tech = technician({"effect": "qc_instrument_run", "summary": "gel run"}, tmp_path=tmp_path)
    run(tech.handle("run_qc_instrument", "fragment_qc", "", lab.robot))

    events = [e for e in lab.log.events if e.kind == "human"]
    assert len(events) == 1 and events[0].data["applied"] == "qc_instrument_run"
    assert "seconds" in events[0].data


def test_the_log_appends_rather_than_overwrites(tmp_path):
    lab = make_sim_lab(SCN, SPEC)
    tech = technician({"effect": "acknowledged", "summary": "noted"}, tmp_path=tmp_path)
    run(tech.handle("review", "final_qc", "", lab.robot))
    run(tech.handle("approve", "final_qc", "", lab.robot))
    assert len((tmp_path / "human.jsonl").read_text().strip().splitlines()) == 2


# ------------------------------------------------- the terminal read must not be threaded

def test_the_report_is_read_on_the_calling_thread():
    """`input_screen()` suspends the task display, and that only works on the thread
    running it. Read from a worker thread (asyncio.to_thread) the display is never
    suspended, Textual keeps the keyboard, and nothing the technician types arrives.
    """

    seen: dict[str, threading.Thread] = {}

    def reader(action, stage, details):
        seen["thread"] = threading.current_thread()
        return "loaded the plate"

    lab = make_sim_lab(SCN, SPEC)
    tech = CliTechnician(judge_model=judge_returning({"effect": "plate_loaded",
                                                      "summary": "loaded"}),
                         log=InterventionLog(path="/dev/null"), reader=reader)
    run(tech.handle("load_plate", "gibson_round1", "", lab.robot))

    assert seen["thread"] is threading.main_thread()


def test_a_report_can_be_submitted_without_a_running_eval():
    """The reader falls back to a plain console, so it works outside an eval too.

    Without the fallback `input_screen()` raises "console input function called outside
    of running evaluation" and the report can never be submitted.
    """


    stdin = sys.stdin
    sys.stdin = io.StringIO("loaded it\nnoticed a bubble\n.\n")
    try:
        report = read_report_from_terminal("load_plate", "gibson_round1", "details")
    finally:
        sys.stdin = stdin
    assert report == "loaded it\nnoticed a bubble"


@pytest.mark.parametrize("token", [".", "END", "end", "submit", "/done"])
def test_each_submit_token_ends_the_report(token):


    stdin = sys.stdin
    sys.stdin = io.StringIO(f"did the thing\n{token}\nthis must not be read\n")
    try:
        report = read_report_from_terminal("load_plate", "gibson_round1", "")
    finally:
        sys.stdin = stdin
    assert report == "did the thing"


def test_square_brackets_in_the_details_do_not_break_the_form():
    """Rich would read '[B1]' as a markup tag and swallow the rest of the line."""


    stdin = sys.stdin
    sys.stdin = io.StringIO("ok\n.\n")
    try:
        report = read_report_from_terminal("load_plate", "s", "put plate in [B1] now")
    finally:
        sys.stdin = stdin
    assert report == "ok"


# ------------------------------------------------- the log goes where the repo is

def test_the_intervention_log_is_anchored_to_the_repo_not_the_cwd():
    """A relative path would scatter records wherever the eval was launched from."""
    path = intervention_log_path("clean_success-1")
    assert path.parent == ROOT / "logs"
    assert path.is_absolute()


def test_each_sample_gets_its_own_file():
    """One shared file made six concurrent samples indistinguishable in the log."""
    a = intervention_log_path("clean_success-1")
    b = intervention_log_path("round2_run_error-1")
    assert a != b
    assert a.name.startswith("human_clean_success-1_") and a.suffix == ".jsonl"
    assert b.name.startswith("human_round2_run_error-1_")


def test_the_filename_carries_the_scenario_epoch_and_time():
    when = datetime(2026, 10, 16, 14, 30, 5, tzinfo=timezone.utc)
    assert intervention_log_path("round1_ambiguous-2", when).name == \
        "human_round1_ambiguous-2_2026-10-16T143005.jsonl"


def test_separate_runs_of_one_scenario_do_not_share_a_file():
    early = intervention_log_path("clean_success-1", datetime(2026, 10, 16, 9, 0, tzinfo=timezone.utc))
    late = intervention_log_path("clean_success-1", datetime(2026, 10, 16, 17, 0, tzinfo=timezone.utc))
    assert early != late


@pytest.mark.parametrize("run_id", ["", "a/b", "..", "x y"])
def test_an_odd_run_id_still_yields_a_usable_filename(run_id):
    name = intervention_log_path(run_id).name
    assert "/" not in name and name.endswith(".jsonl")
    assert name.startswith("human_")


def test_constructing_a_log_creates_no_directory(tmp_path):
    """`task.py` builds a throwaway technician just to validate -T human=; that
    preflight must not leave an empty logs/ behind."""
    target = tmp_path / "nowhere" / "human.jsonl"
    InterventionLog(path=target, run_id="preflight")
    assert not target.parent.exists()

    InterventionLog(path=target, run_id="real").write({"stage": "s"})
    assert target.exists()
