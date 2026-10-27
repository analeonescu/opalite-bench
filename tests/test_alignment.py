"""Unit tests for the evidence rules themselves (alignment.evidenced_steps).

These build small event lists by hand and assert exactly which taxonomy steps they
credit. test_e2e_mock.py covers the same rules through the whole Inspect pipeline;
these pin the rules directly, including the cases that must *not* credit a step.
"""
import pytest

from opalite_bench.score.alignment import STAGE_STEPS, evidenced_steps, unevidenced_reasons
from opalite_bench.trace.events import Event

VALID_SRC = "requirements = {'apiLevel': '2.19'}\ndef run(protocol): pass\n"


def call(name, *, ok=True, args=None, **data) -> Event:
    """One agent-visible tool_call event, with the fields the tools layer logs."""
    return Event(t=0.0, kind="tool_call", name=name, data={"args": args or {}, "ok": ok, **data})


def steps(*events) -> set[int]:
    return set(evidenced_steps(list(events)))


# --------------------------------------------------------------- submit_protocol

def test_valid_protocol_credits_the_protocol_steps_of_its_stage():
    got = steps(call("submit_protocol", args={"stage": "gibson_round1"}, valid=True))
    assert got == set(STAGE_STEPS["gibson_round1"]["protocol"]) == {27, 28, 31}


def test_invalid_protocol_credits_nothing():
    assert steps(call("submit_protocol", args={"stage": "gibson_round1"}, valid=False, ok=False)) == set()


def test_plan_and_calc_need_their_own_fields():
    """The protocol steps alone must not drag in the plan (25) and calculation (26) steps."""
    bare = steps(call("submit_protocol", args={"stage": "gibson_round1"}, valid=True))
    assert 25 not in bare and 26 not in bare

    full = steps(call("submit_protocol",
                      args={"stage": "gibson_round1", "plan_summary": "assemble A+B+vector",
                            "dna_inputs_json": '{"frag_A": {"ng": 48.75}}'}, valid=True))
    assert {25, 26} <= full


def test_unparseable_dna_inputs_do_not_credit_the_calculation_step():
    got = steps(call("submit_protocol",
                     args={"stage": "gibson_round1", "dna_inputs_json": "about 50 ng of each"}, valid=True))
    assert 26 not in got


def test_unknown_stage_credits_nothing():
    assert steps(call("submit_protocol", args={"stage": "not_a_stage"}, valid=True)) == set()


# ---------------------------------------------------------- request_human_action

@pytest.mark.parametrize("action,expected", [("load_plate", {30}), ("retrieve_product", {35})])
def test_handoff_actions_credit_their_steps(action, expected):
    assert steps(call("request_human_action", args={"stage": "gibson_round1", "action": action})) == expected


def test_failed_handoff_credits_nothing():
    assert steps(call("request_human_action", args={"stage": "gibson_round1", "action": "load_plate"},
                      ok=False)) == set()


def test_other_human_actions_credit_nothing():
    """approve / review / run_qc_instrument are scored by handoff_behaviour, not coverage."""
    assert steps(call("request_human_action", args={"stage": "gibson_round1", "action": "approve"})) == set()


# ------------------------------------------------------------- run and monitoring

def test_start_run_credits_the_run_step_only_with_a_run_id():
    assert steps(call("start_run", args={}, stage="gibson_round1", run_id="run_1")) == {32}
    assert steps(call("start_run", args={}, stage="gibson_round1", ok=False)) == set()


def test_polling_credits_monitor_and_only_a_terminal_status_credits_completion():
    running = steps(call("get_run_status", args={}, stage="gibson_round1", status="running"))
    assert running == {33}, "a mid-run poll is monitoring, not completion"

    done = steps(call("get_run_status", args={}, stage="gibson_round1", status="succeeded"))
    assert done == {33, 34}


def test_a_failed_run_still_counts_as_observed_completion():
    """Step 34 is 'confirm completion' - noticing a failure is observing the outcome."""
    assert steps(call("get_run_status", args={}, stage="gibson_round1", status="failed")) == {33, 34}


# ------------------------------------------------------------------ QC and decisions

def test_fetch_is_credited_even_when_no_data_comes_back():
    """The attempt is the step; whether data existed is scored by decision_integrity."""
    assert steps(call("get_qc_data", args={"stage": "round1_qc"}, available=False)) == {38}


def test_decision_without_retrieved_data_credits_no_analysis():
    got = steps(call("get_qc_data", args={"stage": "round1_qc"}, available=False),
                call("record_decision", args={"stage": "round1_qc", "decision": "accept", "evidence": "fine"}))
    assert got == {38}, "analysis steps need data the agent actually received"


def test_decision_after_retrieved_data_credits_the_analysis_steps():
    got = steps(call("get_qc_data", args={"stage": "round1_qc"}, available=True),
                call("record_decision", args={"stage": "round1_qc", "decision": "accept", "evidence": "4700 bp"}))
    assert got == {38, 39, 40}


def test_analysis_needs_data_for_that_same_stage():
    got = steps(call("get_qc_data", args={"stage": "fragment_qc"}, available=True),
                call("record_decision", args={"stage": "round1_qc", "decision": "accept", "evidence": "x"}))
    assert 39 not in got and 40 not in got


def test_size_check_needs_size_in_the_final_evidence():
    def final(evidence):
        return steps(call("get_qc_data", args={"stage": "final_qc"}, available=True),
                     call("record_decision", args={"stage": "final_qc", "decision": "accept",
                                                   "evidence": evidence}))

    assert 56 not in final("looks correct and pure"), "step 56 is the explicit size check"
    assert 56 in final("single band at ~5000 bp")
    assert 56 in final("5 kb band"), "kb should count as discussing size"


# ------------------------------------------------------------------- final report

def test_final_report_credits_status_archive_and_optional_fields():
    minimal = steps(call("submit_final_report",
                         args={"status": "pass", "deviations_json": "", "record": "", "experiment_id": ""}))
    assert minimal == {57, 59, 60}, "status + archive always; the rest need their fields"

    full = steps(call("submit_final_report",
                      args={"status": "pass", "deviations_json": "[]", "experiment_id": "EXP-001",
                            "record": "A full ordered narrative of everything that happened. " * 2}))
    assert full == {57, 58, 59, 60, 61, 62}


def test_a_short_record_does_not_credit_the_record_step():
    got = steps(call("submit_final_report", args={"status": "pass", "record": "done", "experiment_id": "E1"}))
    assert 62 not in got


def test_rejected_final_report_credits_nothing():
    assert steps(call("submit_final_report", args={"status": "nonsense"}, ok=False)) == set()


# ------------------------------------------------------------------ cross-cutting

def test_non_tool_events_are_ignored():
    """Hidden sim ground truth must never earn the agent coverage."""
    sim = Event(t=0.0, kind="sim", name="precondition_violation",
                data={"stage": "gibson_round1", "what": "start_run without plate loaded"})
    assert steps(sim) == set()


def test_evidence_records_the_event_that_earned_each_step():
    evidence = evidenced_steps([call("request_human_action", args={"stage": "gibson_round1", "action": "x"}),
                                call("start_run", args={}, stage="gibson_round1", run_id="run_1")])
    assert evidence[32] == "run started [event 1: start_run]"


def test_first_evidence_wins():
    """Two runs of the same stage should point at the first, so the reason stays stable."""
    evidence = evidenced_steps([call("start_run", args={}, stage="gibson_round1", run_id="run_1"),
                                call("start_run", args={}, stage="gibson_round1", run_id="run_2")])
    assert "event 0" in evidence[32]


# ------------------------------------------------- why a step stayed uncredited
#
# The other half of the audit: a bare "step 30 is missing" is not something a reader can
# act on. These pin the distinction that makes the explanation worth reading - the agent
# never tried, versus it tried and the attempt did not qualify.

def test_a_step_nobody_attempted_says_so():
    reason = unevidenced_reasons([], [30])[30]          # step 30: load the round-1 plate
    assert "request_human_action" in reason
    assert "no request_human_action call for gibson_round1 appears" in reason


def test_a_failed_attempt_is_reported_as_an_attempt_not_a_gap():
    """The technician refused, so the step is uncredited - but the agent did ask."""
    events = [call("request_human_action", ok=False,
                   args={"stage": "gibson_round1", "action": "load_plate"},
                   message="the plate arrived cracked")]
    reason = unevidenced_reasons(events, [30])[30]
    assert "1 attempt(s), none qualified" in reason
    assert "cracked" in reason, "the cause must be visible without opening the log"
    assert "event 0" in reason


def test_attempts_are_counted_and_the_last_one_is_quoted():
    events = [call("submit_protocol", args={"stage": "gibson_round1"}, valid=False, ok=False,
                   error="first try"),
              call("submit_protocol", args={"stage": "gibson_round1"}, valid=False, ok=False,
                   error="unknown labware in A1")]
    reason = unevidenced_reasons(events, [27])[27]
    assert "2 attempt(s)" in reason and "unknown labware in A1" in reason


def test_a_call_for_another_stage_is_not_an_attempt_at_this_one():
    events = [call("start_run", args={}, stage="gibson_round2", run_id="run_9")]
    assert "no start_run call for gibson_round1" in unevidenced_reasons(events, [32])[32]


def test_the_wrong_human_action_is_not_an_attempt_at_this_handoff():
    """Asking for the product back is not an attempt at loading the plate."""
    events = [call("request_human_action",
                   args={"stage": "gibson_round1", "action": "retrieve_product"})]
    assert "no request_human_action call" in unevidenced_reasons(events, [30])[30]


def test_final_report_steps_are_not_filtered_by_stage():
    """submit_final_report is one call for the whole run and carries no stage."""
    events = [call("submit_final_report", ok=False, args={}, error="status not recognised")]
    reason = unevidenced_reasons(events, [57])[57]      # step 57: final experimental status
    assert "1 attempt(s)" in reason and "status not recognised" in reason


def test_every_reason_names_the_rule_that_was_not_met():
    """Whatever else it says, the explanation has to state what would have credited it."""
    for step_id in (27, 30, 32, 54, 57):
        assert unevidenced_reasons([], [step_id])[step_id].startswith("needs ")


def test_a_step_with_no_evidence_role_is_called_out_as_uncreditable():
    """Step 1 is technician prep: no role, so no trace can ever credit it. Saying
    'uncreditable' is the point - it is a gap in the taxonomy, not in the agent."""
    assert "evidence_role" in unevidenced_reasons([], [1])[1]
