"""Run an agent-written robot protocol and turn the result into structured facts.

This is the bridge the rest of the eval scores against: instead of asking "does this
code parse?", it asks "what would this code actually do on the deck?". The answer
(`ProtocolRun`) feeds three things:

  * `deck.py`      applies the transfers to well contents, so QC results follow from
                   what the agent really pipetted;
  * `scorers.py`   compares the transfers against the reference for that stage;
  * `robots.py`    compiles the commands into motion for an Inspect Robots embodiment.

The dry run happens in a subprocess (`_simulate_worker.py`) with a timeout, because
the protocol is agent-written code.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

WORKER = Path(__file__).with_name("_simulate_worker.py")
DEFAULT_TIMEOUT_S = 60
METHOD = "robot dry run"


@dataclass(frozen=True)
class Transfer:
    """One aspirate->dispense pair, in '<slot>:<well>' terms (e.g. 'B2:A1')."""
    source: str
    dest: str
    volume_ul: float


@dataclass
class ProtocolRun:
    ok: bool
    errors: list[str] = field(default_factory=list)
    transfers: list[Transfer] = field(default_factory=list)
    thermal: list[dict] = field(default_factory=list)
    commands: list[dict] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    method: str = METHOD

    @property
    def summary(self) -> str:
        if not self.ok:
            return "; ".join(self.errors) or "dry run failed"
        return f"{len(self.transfers)} transfers, {len(self.thermal)} thermal steps"

    def volume_into(self, well: str) -> float:
        return sum(t.volume_ul for t in self.transfers if t.dest == well)

    def touches(self, wells) -> list[str]:
        """Which of `wells` any command aspirates from, dispenses into or mixes in."""
        return sorted({c["location"] for c in self.commands if c.get("location") in wells})

    def stopped_before(self, wells) -> ProtocolRun:
        """This run cut off just before its first command at one of `wells`, as a robot
        that halts there would leave it: everything earlier happened, nothing after."""
        cut = next((i for i, c in enumerate(self.commands) if c.get("location") in wells), None)
        if cut is None:
            return self
        commands = self.commands[:cut]
        # Each dispense made one transfer, and each hold/cycle one thermal step, in order.
        transfers = sum(c["op"] == "dispense" for c in commands)
        thermal = sum(c["op"] in ("hold", "thermocycle") for c in commands)
        return ProtocolRun(ok=True, transfers=self.transfers[:transfers],
                           thermal=self.thermal[:thermal], commands=commands,
                           log=self.log[:cut], method=self.method)

    def runs_cycles(self, repetitions: int, denature_c: float = 90.0) -> bool:
        """True if a thermocycling profile runs at least `repetitions` cycles including a
        denaturation step at or above `denature_c` (i.e. a real PCR, not a warm hold).
        """
        return any(step["kind"] == "cycle" and step["cycles"] >= repetitions
                   and any(s["celsius"] >= denature_c for s in step["steps"])
                   for step in self.thermal)

    def holds_temperature(self, celsius: float, minutes: float, tol_c: float = 2.0) -> bool:
        """True if some hold stays within tol_c of `celsius` for >= `minutes`."""
        return any(step["kind"] == "hold" and abs(step["celsius"] - celsius) <= tol_c
                   and step["minutes"] + 1e-9 >= minutes
                   for step in self.thermal)


# Dry runs are deterministic and the same source is often submitted more than once (a
# retry, or several scenarios running the same reference protocol), so results are
# memoised per process.
_CACHE: dict[str, "ProtocolRun"] = {}


def worker_input(spec: dict) -> dict:
    """The parts of the spec the robot needs to check a protocol: deck, pipette, labware."""
    return {k: spec.get(k) or {} for k in ("deck", "pipette", "labware")}


def simulate_protocol(source: str, spec: dict | None = None,
                      timeout_s: int = DEFAULT_TIMEOUT_S) -> ProtocolRun:
    """Dry-run `source` against the spec's deck. Never raises: a broken protocol comes
    back as ok=False."""
    if spec is None:
        from opalite_bench.spec.scenarios import load_spec
        spec = load_spec()
    job = {"source": source, **worker_input(spec)}
    key = json.dumps(job, sort_keys=True)
    if (cached := _CACHE.get(key)) is not None:
        return cached
    run = _simulate_uncached(key, timeout_s)
    _CACHE[key] = run
    return run


def _simulate_uncached(job_json: str, timeout_s: int) -> ProtocolRun:
    try:
        proc = subprocess.run([sys.executable, "-I", str(WORKER)], input=job_json,
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return ProtocolRun(ok=False, errors=[f"protocol did not finish within {timeout_s}s "
                                             "(infinite loop, or waiting on input?)"])
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        return ProtocolRun(ok=False, errors=[f"dry run produced no result: {tail}"])

    return ProtocolRun(
        ok=bool(data.get("ok")),
        errors=list(data.get("errors", [])),
        transfers=[Transfer(t["source"], t["dest"], t["volume_ul"]) for t in data.get("transfers", [])],
        thermal=list(data.get("thermal", [])),
        commands=list(data.get("commands", [])),
        log=list(data.get("log", [])),
    )
