"""The robot behind `-T backend=robots`: protocols played as arm motion.

The agent's protocol is dry-run first (protocol_run), so by the time a run starts its
commands are known and valid. Here they become motion:

    commands --compile_plan--> waypoints (x, y, z, plunger_ul, tip) + device commands
             --RobotExecutor--> embodiment.reset() / embodiment.step(action), one per waypoint

The embodiment is an Inspect Robots embodiment (a simulator or a real arm with a pipette
end effector), loaded from a `module:factory` reference, or `KinematicArm` below, a
stand-in that needs no hardware and re-derives every transfer from the arm's motion
alone. That second derivation is what the tests check the compiler against.

Thermocycler commands (lid, hold, cycle) are not arm motion; they go to a separate
device handler. Well contents stay with deck.py either way: on real hardware the liquid
is real, and the deck model is how the eval knows what should be in each well.

HOOK[robots]: the action layout (ACTION_DIMS / to_action), how an embodiment is loaded,
and the deck calibration in the spec's `robot:` section are the parts to adapt to a rig.
Driving an embodiment through `step()` directly bypasses Inspect Robots' own approver
chain, so the workspace check here, and the embodiment's own hard clamp, are the guards.
"""
from __future__ import annotations

import importlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from opalite_bench.lab.protocol_run import ProtocolRun

#: One action per waypoint: end-effector position (mm), plunger volume (uL) and whether
#: a tip is on. An absolute-target layout, so it maps onto an `eef_abs_pose`-style
#: contract plus two tool dimensions.
ACTION_DIMS = ("x", "y", "z", "plunger_ul", "tip")

DEVICE_OPS = {"open_lid", "close_lid", "hold", "thermocycle", "comment"}


@dataclass
class Waypoint:
    x: float
    y: float
    z: float
    plunger_ul: float
    tip: int
    note: str = ""
    #: Set on the step that finishes a dispense, so a run that faults partway knows how
    #: many transfers actually reached a well.
    completes_transfer: bool = False


@dataclass
class DeviceCommand:
    op: str
    args: dict = field(default_factory=dict)


# -- geometry -----------------------------------------------------------------------


class Geometry:
    """Where every well is in the arm's frame, from the spec's labware and robot sections."""

    def __init__(self, spec: dict):
        self.robot = spec.get("robot") or {}
        self.labware = spec.get("labware") or {}
        deck = spec.get("deck") or {}
        self.slots: dict[str, dict] = {}
        for key, cfg in deck.items():
            if key == "trash" or not isinstance(cfg, dict):
                continue
            slot = cfg["slot"] if key == "thermocycler" else key
            self.slots[slot] = self.labware.get(cfg["labware"], {})
        self.trash = deck.get("trash")
        self.safe_z = float(self.robot.get("safe_z_mm", 120.0))
        self.tip_length = float(self.robot.get("tip_length_mm", 50.0))

    def slot_corner(self, slot: str) -> tuple[float, float]:
        """Front-left corner of a slot. Rows A-D run back to front, columns 1-3 left to right."""
        ox, oy = self.robot.get("origin_mm", [0.0, 321.0])
        px, py = self.robot.get("slot_pitch_mm", [164.0, 107.0])
        row, col = ord(slot[0].upper()) - ord("A"), int(slot[1:]) - 1
        return ox + col * px, oy - row * py

    def slot_centre(self, slot: str) -> tuple[float, float]:
        x, y = self.slot_corner(slot)
        return x + 64.0, y + 43.0

    def well_xy(self, key: str) -> tuple[float, float]:
        slot, well = key.split(":", 1)
        lw = self.slots[slot]
        rows, pitch = int(lw.get("rows", 8)), float(lw.get("pitch_mm", 9.0))
        fx, fy = lw.get("first_well_mm", [14.4, 11.2])
        r, c = ord(well[0]) - ord("A"), int(well[1:]) - 1
        x, y = self.slot_corner(slot)
        return x + fx + c * pitch, y + fy + (rows - 1 - r) * pitch

    def well_bottom_z(self, key: str) -> float:
        lw = self.slots[key.split(":", 1)[0]]
        return float(lw.get("top_mm", 16.0)) - float(lw.get("depth_mm", 0.0))

    def tip_top_z(self, key: str) -> float:
        return float(self.slots[key.split(":", 1)[0]].get("top_mm", 60.0))

    def nearest_well(self, x: float, y: float) -> str | None:
        """The well whose centre is within half a pitch of (x, y), if any."""
        for slot, lw in self.slots.items():
            rows, cols = int(lw.get("rows", 8)), int(lw.get("cols", 12))
            half = float(lw.get("pitch_mm", 9.0)) / 2
            for r in range(rows):
                for c in range(cols):
                    key = f"{slot}:{chr(65 + r)}{c + 1}"
                    wx, wy = self.well_xy(key)
                    if abs(wx - x) < half and abs(wy - y) < half:
                        return key
        return None


# -- compiling ----------------------------------------------------------------------


def compile_plan(commands: list[dict], spec: dict) -> list[Waypoint | DeviceCommand]:
    """Turn a dry run's commands into arm waypoints and device commands, in order.

    Every liquid step is: travel at safe height above the well, descend, act on the
    plunger, rise again. Travel height keeps a mounted tip clear of the tallest labware.
    """
    geo = Geometry(spec)
    plan: list[Waypoint | DeviceCommand] = []
    plunger, tip = 0.0, 0

    def at(x, y, z, note="", completes=False):
        plan.append(Waypoint(round(x, 3), round(y, 3), round(z, 3), round(plunger, 4), tip,
                             note, completes))

    def visit(key: str, z: float, act: Callable[[], None], note: str) -> None:
        x, y = geo.well_xy(key)
        at(x, y, geo.safe_z, f"above {key}")
        at(x, y, z, f"down to {key}")
        act()
        at(x, y, geo.safe_z, f"up from {key}")

    for cmd in commands:
        op, where = cmd["op"], cmd.get("location")
        if op in DEVICE_OPS:
            plan.append(DeviceCommand(op, {k: v for k, v in cmd.items() if k != "op"}))
        elif op == "pick_up_tip":
            def act(where=where):
                nonlocal tip
                tip = 1
                at(*geo.well_xy(where), geo.tip_top_z(where), f"tip on at {where}")
            visit(where, geo.tip_top_z(where), act, "pick_up_tip")
        elif op == "drop_tip":
            x, y = geo.slot_centre(geo.trash)
            at(x, y, geo.safe_z, "above trash")
            tip, plunger = 0, 0.0
            at(x, y, geo.safe_z, "tip ejected")
        elif op in ("aspirate", "dispense", "mix"):
            z = geo.well_bottom_z(where) + 1.0 + geo.tip_length * tip
            volume = float(cmd["volume_ul"])

            def act(op=op, where=where, volume=volume, z=z, cmd=cmd):
                nonlocal plunger
                x, y = geo.well_xy(where)
                if op == "aspirate":
                    plunger += volume
                    at(x, y, z, f"aspirate {volume:g} uL")
                elif op == "dispense":
                    plunger -= volume
                    at(x, y, z, f"dispense {volume:g} uL", completes=True)
                else:
                    for _ in range(int(cmd["repetitions"])):
                        plunger += volume
                        at(x, y, z, "mix up")
                        plunger -= volume
                        at(x, y, z, "mix down")
            visit(where, z, act, op)
        else:
            raise ValueError(f"cannot compile robot command {op!r}")
    return plan


def to_action(wp: Waypoint):
    """A waypoint as the embodiment's action vector, in ACTION_DIMS order."""
    import numpy as np
    return np.array([wp.x, wp.y, wp.z, wp.plunger_ul, float(wp.tip)], dtype=np.float64)


# -- executing ----------------------------------------------------------------------


@dataclass
class ExecutionReport:
    ok: bool
    message: str = ""
    steps: int = 0
    transfers_completed: int = 0
    log: list[str] = field(default_factory=list)
    #: True when the robot never started (busy, or halted), so nothing on the deck changed.
    blocked: bool = False


def is_physical(embodiment: Any) -> bool:
    """Real hardware unless the embodiment says otherwise with `physical = False`."""
    return bool(getattr(embodiment, "physical", True))


class DeviceBusy(RuntimeError):
    pass


class DeviceClaim:
    """An exclusive claim on one physical device, held for as long as it moves.

    A lock file per device under the temp directory, so two evals on the same host (or
    two processes of one) can never drive the same arm at once. The second one gets
    DeviceBusy rather than waiting: the agent sees a robot that is busy.
    """

    DIR = Path(tempfile.gettempdir()) / "opalite_bench_claims"

    def __init__(self, device_id: str):
        self.device_id = device_id
        self.path = self.DIR / (re.sub(r"[^A-Za-z0-9_.-]", "_", device_id) + ".lock")
        self._fd: int | None = None

    def __enter__(self) -> "DeviceClaim":
        import fcntl   # POSIX only; physical runs are refused where it is missing
        self.DIR.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise DeviceBusy(f"device '{self.device_id}' is claimed by another run") from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return self

    def __exit__(self, *exc) -> None:
        import fcntl
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


class RobotExecutor:
    """Plays dry-run protocols on an embodiment, one waypoint per `step()`.

    A fault halts the robot: `halted` holds what went wrong, and every later run is
    refused until a person clears it (`clear()`, via the technician's `review`). A
    physical embodiment is claimed (DeviceClaim) for the duration of each run.
    """

    def __init__(self, embodiment: Any, spec: dict,
                 action_fn: Callable[[Waypoint], Any] = to_action,
                 devices: Callable[[DeviceCommand], str | None] | None = None,
                 device_id: str | None = None):
        self.embodiment, self.spec, self.action_fn = embodiment, spec, action_fn
        self.devices = devices or (lambda cmd: None)   # HOOK[robots]: a real thermocycler
        self.workspace = (spec.get("robot") or {}).get("workspace_mm") or {}
        self.device_id = device_id or type(embodiment).__name__
        self.halted: str | None = None
        self._ready = False

    def clear(self) -> str | None:
        """A person has inspected the robot after a fault. It resets before moving again."""
        fault, self.halted, self._ready = self.halted, None, False
        return fault

    def check(self, run: ProtocolRun) -> list[str]:
        """Problems that would stop the arm, found before it moves: anything off the workspace."""
        problems = []
        for wp in compile_plan(run.commands, self.spec):
            if not isinstance(wp, Waypoint):
                continue
            for axis in ("x", "y", "z"):
                lo, hi = self.workspace.get(axis, (float("-inf"), float("inf")))
                if not lo <= getattr(wp, axis) <= hi:
                    problems.append(f"{wp.note}: {axis}={getattr(wp, axis):g} mm is outside the "
                                    f"arm's workspace [{lo:g}, {hi:g}]")
                    break
        return problems[:5]

    def execute(self, stage: str, run: ProtocolRun) -> ExecutionReport:
        if self.halted:
            return ExecutionReport(False, f"robot halted after a fault ({self.halted})", blocked=True)
        if not is_physical(self.embodiment):
            return self._play(stage, run)
        try:
            with DeviceClaim(self.device_id):
                return self._play(stage, run)
        except DeviceBusy as exc:
            return ExecutionReport(False, str(exc), blocked=True)

    def _play(self, stage: str, run: ProtocolRun) -> ExecutionReport:
        report = ExecutionReport(ok=True)
        # HOOK[safety]: on real hardware, gate this on human approval and an e-stop path.
        try:
            if not self._ready:
                self.embodiment.reset()
                self._ready = True
            for item in compile_plan(run.commands, self.spec):
                if isinstance(item, DeviceCommand):
                    if (line := self.devices(item)):
                        report.log.append(line)
                    continue
                self.embodiment.step(self.action_fn(item))
                report.steps += 1
                report.transfers_completed += item.completes_transfer
        except Exception as exc:  # noqa: BLE001
            # Any fault on the arm is a failed run, not a crashed eval: the agent has to
            # notice it and decide what to do, the way it would with a real robot.
            report.ok = False
            report.message = f"robot fault during {stage}: {type(exc).__name__}: {exc}"
            self.halted = report.message
        report.log.append(f"arm executed {report.steps} waypoints, "
                          f"{report.transfers_completed} transfers")
        return report


class KinematicArm:
    """Stand-in embodiment: a gantry with a pipette and no physics.

    Takes the same actions a real embodiment would (ACTION_DIMS), refuses anything
    outside the workspace, and independently works out each transfer from where the
    plunger moved: down at a well with liquid in the tip is a dispense into that well.
    `transfers` is therefore derived from motion alone, not from the protocol.
    """

    physical = False

    def __init__(self, spec: dict):
        self.geo = Geometry(spec)
        self.workspace = (spec.get("robot") or {}).get("workspace_mm") or {}
        self.transfers: list[tuple[str, str, float]] = []
        self.reset()

    def reset(self) -> None:
        self.state = [0.0, 0.0, self.geo.safe_z, 0.0, 0.0]
        self.source: str | None = None

    def step(self, action) -> None:
        x, y, z, plunger, tip = (float(v) for v in action)
        for axis, value in zip("xyz", (x, y, z)):
            lo, hi = self.workspace.get(axis, (float("-inf"), float("inf")))
            if not lo <= value <= hi:
                raise RuntimeError(f"{axis}={value:g} mm outside workspace [{lo:g}, {hi:g}]")
        delta = plunger - self.state[3]
        if abs(delta) > 1e-6 and tip:
            well = self.geo.nearest_well(x, y)
            if delta > 0 and self.source is None:
                self.source = well
            elif delta < 0 and well != self.source:
                self.transfers.append((self.source, well, round(-delta, 4)))
            if plunger <= 1e-6:
                self.source = None
        if not tip:
            self.source = None
        self.state = [x, y, z, plunger, tip]


def resolve_factory(ref: str) -> Callable[[], Any]:
    """The factory a "pkg.module:factory" reference names, imported but not called."""
    module, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"embodiment must be 'kinematic' or 'module:factory', not {ref!r}")
    factory = getattr(importlib.import_module(module), attr)
    if not callable(factory):
        raise TypeError(f"{ref} is not callable")
    return factory


def load_embodiment(ref: str | None, spec: dict, allow_physical: bool = False):
    """`None`/"kinematic" -> KinematicArm; "pkg.module:factory" -> factory().

    The factory returns an Inspect Robots embodiment (anything with reset() and
    step(action)) whose action space follows ACTION_DIMS. Inspect Robots asks for
    hardware-free constructors, so building one to ask whether it is physical is safe.
    A physical one is refused unless `allow_physical` - which is the operator saying
    this host may drive it, not an authorisation from the facility.
    """
    if ref in (None, "", "kinematic"):
        return KinematicArm(spec)
    embodiment = resolve_factory(ref)()
    if is_physical(embodiment) and not allow_physical:
        raise PermissionError(
            f"embodiment {ref} is physical (it does not declare physical = False); "
            "refusing to drive it without -T allow_physical=true")
    return embodiment
