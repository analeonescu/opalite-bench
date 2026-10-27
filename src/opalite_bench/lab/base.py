"""Backend interfaces. These are the main HOOKS for wiring in real hardware.

A `Lab` bundles three independent channels so you can mix real and simulated:
  Robot       -> liquid-handling robot (validate / upload / run / monitor)
  HumanChannel-> a technician (load plates, run external QC, retrieve product, approve)
  QCSource    -> external QC instrument data (gel/TapeStation/sequencing/etc.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from opalite_bench.trace.events import EventLog
from opalite_bench.spec.scenarios import load_spec, stage_ids, stages_of_kind

# The stage list is the experiment's, not the engine's: it comes from the `stages`
# section of configs/task_spec.yaml, in the order that file lists them.
_SPEC = load_spec()
STAGES = stage_ids(_SPEC)
LAB_STAGES = stages_of_kind(_SPEC, "lab")
QC_STAGES = stages_of_kind(_SPEC, "qc")

#Physical steps the robot cannot do at all, so they are the technician's by default.
#Skipping one does not just lose a point - the lab state never changes, and the next
#stage has nothing to work with.

HUMAN_ONLY_ACTIONS = ["load_plate", "retrieve_product", "run_qc_instrument",
                      "photograph_gel", "other_qc", "other_human_action"]

#Available at any point in any stage, for when something has gone wrong. These are how
#the agent hands a judgement call back to a person instead of deciding alone.
#This list is not exhaustive: the technician can always intervene with 
# any action, but these are the ones the agent is expected to know about and request. 
# The technician may also intervene on their own initiative, e.g. 
# if the agent is taking too long or doing something obviously wrong.

INTERVENTION_ACTIONS = ["replace_plate", "continue_as_is", "approve", "review", "other_intervention"]

# if something needs to be flagged by the human, the agent should request on eof these actions

HUMAN_ACTIONS = HUMAN_ONLY_ACTIONS + INTERVENTION_ACTIONS


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    method: str = "static"  # "static" | "robot dry run" | ...


@dataclass
class RunState:
    run_id: str
    status: str  # "idle" | "running" | "succeeded" | "failed" | "stopped"
    log: list[str] = field(default_factory=list)
    #: True when the robot refused to start at all, rather than starting and failing.
    #: The stage is left exactly as it was, so this is not evidence that it went wrong.
    blocked: bool = False


@dataclass
class HumanResponse:
    ok: bool
    message: str


@dataclass
class QCResult:
    available: bool
    data: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class Robot(Protocol):
    def validate_protocol(self, source: str, stage: str) -> ValidationResult: ...
    def upload_protocol(self, source: str, stage: str) -> str: ...           # -> protocol_id
    def start_run(self, protocol_id: str) -> str: ...                         # -> run_id
    def get_run(self, run_id: str) -> RunState: ...


class HumanChannel(Protocol):
    # Async because a real technician is slow and an LLM judge has to read their report
    # (see human_cli.py); the scripted technician just returns immediately.
    async def request(self, action: str, stage: str, details: str) -> HumanResponse: ...


class QCSource(Protocol):
    def fetch(self, stage: str) -> QCResult: ...


@dataclass
class Lab:
    robot: Robot
    human: HumanChannel
    qc: QCSource
    log: EventLog = field(default_factory=EventLog)
    archive: dict[str, Any] = field(default_factory=dict)  # HOOK[records]: LIMS / data-system writes
