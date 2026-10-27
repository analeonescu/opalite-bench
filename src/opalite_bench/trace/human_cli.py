"""A real technician in the loop: a terminal clipboard, a separate log, and a judge.

When the agent asks for a physical step, the run pauses and the technician is shown a
clipboard in the terminal. They type what they actually did, in their own words. That
report is:

  1. timed (from the moment the clipboard appears to the moment they submit it),
  2. written to its own JSONL log, separate from the eval log, and
  3. read by an LLM judge, which decides which lab effect it describes.

The judge is what turns free text into an action. It chooses from a fixed list of effects
(`SimulatedLab.EFFECTS`) and may *disagree with the agent's request*: if the agent asked
for a plate to be loaded and the technician says the plate was cracked and they could not
load it, the judge returns `not_done` and the lab state does not change. Everything the
judge decides is logged with the text it decided from, so it can be checked afterwards.

Nothing here is wired into the scorers' ground truth: a technician's report describes the
world, it does not define it.
"""
from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import rich
from inspect_ai._display.core.active import _active_task_screen
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, get_model
from inspect_ai.util import input_screen

from opalite_bench.paths import LOGS

#: Characters that are awkward in a filename, replaced with '-'.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def intervention_log_path(run_id: str, when: datetime | None = None) -> Path:
    """One file per sample per run: logs/human_<scenario>_<timestamp>.jsonl

    `run_id` is the sample's `<scenario>-<epoch>`, so two epochs of the same scenario
    stay apart, and the timestamp keeps separate runs from appending to each other.
    """
    when = when or datetime.now(timezone.utc)
    name = _UNSAFE.sub("-", run_id) or "unknown"
    return LOGS / f"human_{name}_{when.strftime('%Y-%m-%dT%H%M%S')}.jsonl"

#: What the technician is asked to cover. Mirrors the reporting template the agent needs:
#: an outcome, anything that differed, anything noticed, and anything not checked.
CLIPBOARD = """\
┌────────────────────────────────────────────────────────────────────────────┐
│  TECHNICIAN REQUEST                                                        │
└────────────────────────────────────────────────────────────────────────────┘
  Stage    : {stage}
  Action   : {action}
  Details  : {details}

  Please say what you ACTUALLY did. Include, where they apply:
    - done / done with problems / not done
    - anything that differed from the request (volumes, wells, substitutions)
    - sample or plate IDs, deck slots
    - anything you fixed, and roughly how long it took
    - anything you noticed (bubbles, low volume, condensation)
    - anything you did NOT check

  Type your report. To submit it, put '.' on a line of
  its own and press Enter. END, SUBMIT and Ctrl-D also
  submit. An empty report counts as NOT DONE.
"""

#: Typed on a line of its own, any of these ends the report. '.' is the documented one;
#: the others are here because they are what people reach for.
SUBMIT_TOKENS = {".", "end", "END", "submit", "SUBMIT", "/done", "done."}

JUDGE_SYSTEM = """\
You read a lab technician's free-text report and decide what it means for the lab.

The agent requested: {action} (stage: {stage})
Details given to the technician: {details}

Choose exactly one effect:
  plate_loaded      the plates/labware for this stage are now on the deck
  plate_replaced    fresh labware has replaced a plate that had already been used,
                    so this stage's wells are empty again
  rerun_approved    the technician decided to run the stage again on the plate that
                    was already used, accepting whatever is still in its wells
  qc_instrument_run the QC instrument has been run for this stage
  gel_photographed  the gel for this stage has been photographed and the image sent
  product_retrieved the product of this stage has been collected and purified
  robot_cleared     the technician inspected a robot that had halted on a fault and
                    cleared it, so it may run again
  acknowledged      the technician did something that changes no physical state
                    (a review, an approval, an observation)
  not_done          the step did not happen, or happened in a way that means the
                    lab must not treat it as done

Rules:
  - Judge what the technician says they did, NOT what the agent asked for. If they
    say they could not do it, or did something else, that is not_done.
  - If the report is ambiguous about whether the physical step completed, choose
    not_done. It is safer to stop than to proceed on an unclear report.
  - plate_replaced and rerun_approved differ in what is left in the wells. Only choose
    plate_replaced if the technician says they put fresh labware on; "just run it
    again" with no mention of a new plate is rerun_approved.
  - Record deviations, observations and anything explicitly not checked, using the
    technician's own meaning. Do not invent details they did not give.

Reply with ONLY a JSON object:
{{"effect": "<one of the above>",
  "confident": true|false,
  "summary": "<one line the agent will read>",
  "deviations": ["..."],
  "observations": ["..."],
  "not_checked": ["..."]}}
"""


@dataclass
class Interpretation:
    effect: str
    summary: str
    confident: bool = True
    deviations: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    not_checked: list[str] = field(default_factory=list)
    judge_error: str | None = None


def parse_judgement(text: str, allowed: tuple[str, ...]) -> Interpretation:
    """Read the judge's JSON. Anything unparseable or out of range becomes `not_done`.

    Failing closed matters: an unreadable judgement must never let the run continue as
    though a physical step had happened.
    """
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return Interpretation("not_done", "could not interpret the technician's report",
                              confident=False, judge_error="no JSON in judge reply")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Interpretation("not_done", "could not interpret the technician's report",
                              confident=False, judge_error=f"bad JSON: {exc}")

    effect = str(data.get("effect", "")).strip()
    if effect not in allowed:
        return Interpretation("not_done", f"judge returned an unknown effect '{effect}'",
                              confident=False, judge_error=f"effect not in {allowed}")

    def as_list(key: str) -> list[str]:
        value = data.get(key) or []
        return [str(v) for v in value] if isinstance(value, list) else [str(value)]

    return Interpretation(
        effect=effect,
        summary=str(data.get("summary") or "").strip() or "technician report recorded",
        confident=bool(data.get("confident", True)),
        deviations=as_list("deviations"),
        observations=as_list("observations"),
        not_checked=as_list("not_checked"),
    )


class InterventionLog:
    """Appends one JSON object per human step, to this sample's own file."""

    def __init__(self, path: Path | str | None = None, run_id: str = ""):
        self.run_id = run_id
        self.path = Path(path) if path is not None else intervention_log_path(run_id)

    def write(self, record: dict) -> dict:
        # Created on first write, not on construction: `task.py` builds a throwaway
        # technician just to validate `-T human=`, and that preflight should not leave
        # an empty logs/ directory behind.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"run_id": self.run_id,
                  "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record


@contextmanager
def _technician_console():
    """Inspect's input screen when an eval is running, a plain console otherwise.

    `input_screen` suspends the live task display so a blocking read can own the
    terminal. Outside a running eval there is no display to suspend and it raises, so
    fall back to the bare Rich console - which is what makes this runnable by hand.
    """
    if _active_task_screen.get(None) is not None:
        with input_screen(header="Technician request") as console:
            yield console
    else:
        yield rich.get_console()


def read_report_from_terminal(action: str, stage: str, details: str) -> str:
    """Show the clipboard and collect a multi-line report, pausing Inspect's live display.

    Must run on the thread driving the eval; see the note in `CliTechnician.handle`.
    """
    with _technician_console() as console:
        # markup=False: the report form is literal text, and a stray '[' in `details`
        # would otherwise be parsed as a Rich tag and swallow the rest of the line.
        console.print(CLIPBOARD.format(stage=stage, action=action, details=details or "-"),
                      markup=False, highlight=False)
        lines: list[str] = []
        while True:
            try:
                line = console.input("  > ")
            except (EOFError, KeyboardInterrupt):
                # Ctrl-D / Ctrl-C submits what has been typed so far rather than
                # discarding it; an empty report is judged `not_done` anyway.
                break
            if line.strip() in SUBMIT_TOKENS:
                break
            lines.append(line)
    return "\n".join(lines).strip()


class CliTechnician:
    """Asks a real person what they did, then has a judge turn that into a lab effect."""

    name = "cli"

    def __init__(self, judge_model=None, log: InterventionLog | None = None,
                 reader=read_report_from_terminal):
        self._judge_model = judge_model
        self.log = log or InterventionLog()
        self.reader = reader            # injectable, so tests never touch stdin

    @property
    def judge(self):
        if self._judge_model is None or isinstance(self._judge_model, str):
            self._judge_model = get_model(self._judge_model)
        return self._judge_model

    async def interpret(self, action: str, stage: str, details: str, report: str,
                  allowed: tuple[str, ...]) -> Interpretation:
        if not report:
            return Interpretation("not_done", "the technician submitted an empty report",
                                  confident=False)

        prompt = JUDGE_SYSTEM.format(action=action, stage=stage, details=details or "-")
        try:
            reply = await self.judge.generate(
                [ChatMessageSystem(content=prompt),
                 ChatMessageUser(content=f"Technician report:\n{report}")])
            text = reply.completion
        except Exception as exc:  # noqa: BLE001
            return Interpretation("not_done", "the report could not be interpreted",
                                  confident=False, judge_error=f"{type(exc).__name__}: {exc}")
        return parse_judgement(text, allowed)

    async def handle(self, action: str, stage: str, details: str, lab):
        started = time.monotonic()
        # Read inline, on this thread, even though it blocks the event loop.
        #
        # It must not go through asyncio.to_thread: `input_screen()` suspends the task
        # display, and with Inspect's default (textual) display that suspend only takes
        # effect on the thread running the app. Called from a worker thread the display
        # is never suspended, Textual keeps the keyboard, and nothing the technician
        # types - including the '.' that submits - reaches `console.input()`.
        #
        # Blocking here is the intended behaviour: the lab is waiting for a person.
        # Run with --max-samples 1 so no other sample needs the loop meanwhile.
        report = self.reader(action, stage, details)
        seconds = round(time.monotonic() - started, 1)

        judged = await self.interpret(action, stage, details, report, lab.EFFECTS)
        response = lab.apply_effect(judged.effect, stage)

        record = {"stage": stage, "requested_action": action, "details": details,
                  "report": report, "seconds": seconds, **asdict(judged),
                  "applied": judged.effect, "lab_ok": response.ok}
        self.log.write(record)
        # Also into the eval's own event log, so intervention types and times can be scored.
        lab.log.add("human", "intervention", **record)

        # The agent is told the technician's own words, plus what the lab concluded.
        message = judged.summary
        if judged.deviations:
            message += " | deviations: " + "; ".join(judged.deviations)
        if judged.observations:
            message += " | observed: " + "; ".join(judged.observations)
        if judged.not_checked:
            message += " | NOT checked: " + "; ".join(judged.not_checked)
        if not judged.confident:
            message += " | the report was unclear, so the lab did not treat this as done"
        return type(response)(response.ok, message)
