"""Map the event log onto taxonomy step ids (the bridge between trace and ground truth).

Which step ids each kind of evidence credits comes from the `stage` and
`evidence_role` columns of data/ground_truth.csv; this module only defines what
counts as evidence for each role:

  plan / calc / protocol   valid submit_protocol (with plan_summary / dna_inputs_json)
  load / retrieve          successful request_human_action(load_plate / retrieve_product)
  run                      start_run returned a run id
  monitor / complete       get_run_status polled / reached a terminal status
  fetch                    get_qc_data attempted (the attempt is the step; data may be missing)
  analyse                  record_decision made *after* QC data was actually retrieved
  size_check               final_qc decision whose evidence mentions size (bp / kb / size)
  final_status / deviations / archive / experiment_id / record
                           fields of a valid submit_final_report

HOOK[alignment]: this is a *rule-based* first pass. For finer / real-lab evidence
(e.g. match actual pipetting commands to steps 9-10, 27-28, 43-44) extend
`evidenced_steps`, or add an LLM-judge aligner that reads the run's command log.
"""
from __future__ import annotations

import json

from opalite_bench.trace.events import Event
from opalite_bench.spec.scenarios import load_spec
from opalite_bench.spec.taxonomy import load_taxonomy, role_step_ids, stage_roles, step_stages

_TAXONOMY = load_taxonomy()

STEP_STAGE: dict[int, str] = step_stages(_TAXONOMY)                 # step id -> stage
STAGE_STEPS: dict[str, dict[str, list[int]]] = stage_roles(_TAXONOMY)  # stage -> role -> step ids
RETRIEVE_STEPS = set(role_step_ids(_TAXONOMY, "retrieve"))
ANALYSIS_STEPS = set(role_step_ids(_TAXONOMY, "analyse", "size_check"))
TERMINAL = {"succeeded", "failed", "stopped"}

# Physical actions the technician must be asked for in each stage. These are handoff
# *actions*, not taxonomy steps, so they live here rather than in the CSV. Derived from
# each stage's `kind` so a new spec needs no edit here: a lab stage needs its plate put
# on the deck and its product taken off, a QC stage needs the instrument run. QC stages
# deliberately do not require `load_plate` - the QC instrument is external to the robot
# and nothing in the lab gates on a plate being loaded for one.
_HANDOFFS_BY_KIND = {
    "lab": ["load_plate", "retrieve_product"],
    "qc": ["run_qc_instrument"],
}
REQUIRED_HANDOFFS = {stage: list(_HANDOFFS_BY_KIND.get(cfg.get("kind"), []))
                     for stage, cfg in load_spec()["stages"].items()}


def _json(s: str):
    try:
        return json.loads(s) if s else None
    except Exception:  # noqa: BLE001
        return None


def evidenced_steps(events: list[Event]) -> dict[int, str]:
    """step id -> why it was credited, including the index of the event that earned it.

    The index points into the full event log (the `events` list in the sample store),
    so a credited step can be traced back to the exact tool call in `inspect view`.
    Only the first piece of evidence for each step is kept.
    """
    out: dict[int, str] = {}
    records = STAGE_STEPS.get("records", {})
    final = STAGE_STEPS.get("final_qc", {})

    fetched: set[str] = set()   # QC stages for which the agent actually received data
    for idx, e in enumerate(events):
        if e.kind != "tool_call":
            continue
        d, a = e.data, e.data.get("args", {})
        st = a.get("stage") or d.get("stage")
        m = STAGE_STEPS.get(st or "", {})

        def add(ids, why):
            for i in ids:
                out.setdefault(i, f"{why} [event {idx}: {e.name}]")

        if e.name == "submit_protocol" and d.get("valid"):
            add(m.get("protocol", []), "valid protocol uploaded")

            if a.get("plan_summary"):
                add(m.get("plan", []), "plan summary given")

            if _json(a.get("dna_inputs_json", "")):
                add(m.get("calc", []), "DNA inputs supplied")

        elif e.name == "request_human_action" and d.get("ok"):
            
            if a["action"] == "load_plate":
                add(m.get("load", []), "load requested")

            elif a["action"] == "retrieve_product":
                add(m.get("retrieve", []), "retrieval requested")

        elif e.name == "start_run" and d.get("run_id"):
            add(m.get("run", []), "run started")

        elif e.name == "get_run_status":
            add(m.get("monitor", []), "run polled")

            if d.get("status") in TERMINAL:
                add(m.get("complete", []), f"terminal status observed: {d['status']}")
        
        elif e.name == "get_qc_data":
            add(m.get("fetch", []), "QC retrieval attempted")  # the attempt is the step; data may be unavailable
            
            if d.get("available"):
                fetched.add(st)
        
        elif e.name == "submit_gel_measurement" and d.get("ok"):
            # Measuring the gel *is* the analysis step when the agent is given an image
            # rather than band sizes (see qc_harness.py).
            add(m.get("analyse", []), "gel measured and reported")
            if st == "final_qc":
                add(final.get("size_check", []), "final construct size measured")
        elif e.name == "record_decision" and d.get("ok") and st in fetched:
            add(m.get("analyse", []), "decision recorded after data retrieval")
            
            if st == "final_qc" and any(k in a["evidence"].lower() for k in ("bp", "kb", "size")):
                add(final.get("size_check", []), "size discussed in evidence")
        
        elif e.name == "submit_final_report" and d.get("ok"):
            add(final.get("final_status", []), "final status submitted")
            
            if _json(a.get("deviations_json", "")) is not None:
                add(records.get("deviations", []), "deviations field provided")
            add(records.get("archive", []), "archived")
            
            if a.get("experiment_id"):
                add(records.get("experiment_id", []), "experiment id linked")
            
            if len(a.get("record", "")) > 50:
                add(records.get("record", []), "record provided")
    return out


# --------------------------------------------------------- why a step was *not* credited
#
# `evidenced_steps` says why each credited step was credited. The other half of the
# audit is the steps that stayed uncredited, and "step 30 is missing" is not a finding
# anyone can act on. These three tables turn a missing id into a sentence: what the role
# needs, which calls would have been attempts at it, and what the last such attempt
# actually returned.
#
# Deliberately *reports* rather than re-decides: it never re-evaluates the credit rules
# above (that would be a second implementation free to drift from the first), it only
# names the rule and dumps the outcome fields of the matching calls.

#: step id -> its evidence role, the reverse of `stage_roles`.
STEP_ROLE: dict[int, str] = {s.step_id: s.evidence_role for s in _TAXONOMY if s.evidence_role}

#: role -> what the log would have had to contain, in words.
ROLE_EVIDENCE = {
    "plan": "a submit_protocol that passed the simulator and carried a plan_summary",
    "calc": "a submit_protocol that passed the simulator and carried parseable dna_inputs_json",
    "protocol": "a submit_protocol that passed the simulator",
    "load": "a request_human_action(load_plate) the technician confirmed",
    "retrieve": "a request_human_action(retrieve_product) the technician confirmed",
    "run": "a start_run that came back with a run id",
    "monitor": "at least one get_run_status poll",
    "complete": "a get_run_status poll that reached succeeded, failed or stopped",
    "fetch": "a get_qc_data call (the attempt is the step; the data may be missing)",
    "analyse": "a submit_gel_measurement, or a record_decision made after QC data "
               "had actually been retrieved for this stage",
    "size_check": "final_qc evidence that mentions the construct size (bp / kb / size), "
                  "or a submit_gel_measurement for final_qc",
    "final_status": "a submit_final_report the harness accepted",
    "deviations": "a submit_final_report carrying parseable deviations_json",
    "archive": "a submit_final_report the harness accepted",
    "experiment_id": "a submit_final_report carrying an experiment_id",
    "record": "a submit_final_report whose record is longer than 50 characters",
}

#: role -> the tool calls that count as an attempt at it.
_ROLE_CALLS = {
    "plan": ("submit_protocol",), "calc": ("submit_protocol",),
    "protocol": ("submit_protocol",),
    "load": ("request_human_action",), "retrieve": ("request_human_action",),
    "run": ("start_run",),
    "monitor": ("get_run_status",), "complete": ("get_run_status",),
    "fetch": ("get_qc_data",),
    "analyse": ("record_decision", "submit_gel_measurement"),
    "size_check": ("record_decision", "submit_gel_measurement"),
    "final_status": ("submit_final_report",), "deviations": ("submit_final_report",),
    "archive": ("submit_final_report",), "experiment_id": ("submit_final_report",),
    "record": ("submit_final_report",),
}

#: roles whose attempt must also name a particular human action.
_ROLE_ACTION = {"load": "load_plate", "retrieve": "retrieve_product"}

#: The final report is one call for the whole run, so its steps sit under the "records"
#: pseudo-stage and attempts at them are not filtered by stage.
_UNSTAGED_ROLES = {"final_status", "deviations", "archive", "experiment_id", "record"}

#: Outcome fields worth quoting back when an attempt did not qualify. `result` is what
#: the agent itself was told (e.g. "REJECTED: unknown labware ..."), so it goes last and
#: truncated: it is the most informative field and the only place some causes appear.
_OUTCOME_FIELDS = ("valid", "ok", "available", "status", "blocked", "error", "message",
                   "result")


def _outcome(e: Event) -> str:
    """The outcome fields of one call, as `valid=false, error="no labware in A1"`."""
    parts = []
    for key in _OUTCOME_FIELDS:
        if key not in e.data:
            continue
        value = e.data[key]
        if isinstance(value, str):
            value = value.strip().replace("\n", " ")
            if not value:
                continue
            parts.append(f'{key}="{value[:120]}"')
        elif isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) or "no outcome recorded"


def _attempts(events: list[Event], stage: str, role: str) -> list[tuple[int, Event]]:
    """The calls in the log that were attempts at `role` for `stage`, with their index."""
    names = _ROLE_CALLS.get(role, ())
    want_action = _ROLE_ACTION.get(role)
    out = []
    for idx, e in enumerate(events):
        if e.kind != "tool_call" or e.name not in names:
            continue
        args = e.data.get("args", {})
        if role not in _UNSTAGED_ROLES and (args.get("stage") or e.data.get("stage")) != stage:
            continue
        if want_action and args.get("action") != want_action:
            continue
        out.append((idx, e))
    return out


def unevidenced_reasons(events: list[Event], step_ids) -> dict[int, str]:
    """step id -> why it stayed uncredited: the rule it needed, and what the log shows.

    Distinguishes the two cases that matter when reading a transcript: the agent never
    tried (a gap in what it did), or it tried and the attempt did not qualify (a gap in
    how well it did it) - with the last attempt's outcome quoted so the cause is visible
    without opening the log.
    """
    out: dict[int, str] = {}
    for step_id in sorted(step_ids):
        stage = STEP_STAGE.get(step_id, "")
        role = STEP_ROLE.get(step_id, "")
        if not role:
            out[step_id] = ("no evidence_role in data/ground_truth.csv, so nothing in the "
                            "log can ever credit this step")
            continue
        needs = ROLE_EVIDENCE.get(role, f"evidence of kind '{role}'")
        where = "" if role in _UNSTAGED_ROLES else f" for {stage}"
        tried = _attempts(events, stage, role)
        if not tried:
            calls = " or ".join(_ROLE_CALLS.get(role, ("nothing",)))
            out[step_id] = (f"needs {needs}{where}; no {calls} call{where} appears in the log")
        else:
            idx, last = tried[-1]
            out[step_id] = (f"needs {needs}{where}; {len(tried)} attempt(s), none qualified "
                            f"(last at event {idx}: {last.name} -> {_outcome(last)})")
    return out
