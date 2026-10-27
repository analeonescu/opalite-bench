"""Subprocess worker: dry-run one agent-written robot protocol.

Runs as its own process (see protocol_run.simulate_protocol) because the protocol is
code the *agent* wrote: a hang, a crash or an `exit()` must not take the eval with it.

Reads {"source": ..., "deck": ..., "pipette": ..., "labware": ...} as JSON on stdin and
writes one JSON object to stdout:
  {ok, errors, transfers, thermal, commands, log}
Anything the protocol itself prints goes to stderr, so stdout stays parseable.

Stdlib only, on purpose: the worker must run under any interpreter the eval does, and
must not import the eval package (the protocol could reach into it).

The protocol API
----------------
A protocol is a module defining `run(robot)`. Wells are addressed as "<slot>:<well>",
e.g. "B2:A1"; the thermocycler's plate is addressed by the thermocycler's slot.

    robot.pick_up_tip()                       next unused tip from the tip rack
    robot.drop_tip()                          into the trash
    robot.aspirate(volume_ul, well)
    robot.dispense(volume_ul, well)
    robot.mix(repetitions, volume_ul, well)
    robot.transfer(volume_ul, source, dest, new_tip="always")   # "always" | "once" | "never"
    robot.open_lid() / robot.close_lid()      thermocycler lid
    robot.hold(celsius, minutes)              hold the thermocycler block
    robot.thermocycle(steps, cycles)          steps: [{"celsius": 98, "seconds": 10}, ...]
    robot.comment(text)

Every call is checked the way the physical robot would refuse it: unknown slots or
wells, volumes outside the pipette's range, aspirating without a tip, pipetting into
the thermocycler with its lid closed, heating with the lid open, running out of tips.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys

_WELL = re.compile(r"^([A-Z])(\d{1,2})$")


class ProtocolError(Exception):
    """Something the robot would refuse to do."""


class Robot:
    """Records what a protocol does, refusing anything the hardware could not do.

    HOOK[validate]: add the checks your robot makes (well capacity, labware height,
    liquid classes). Keep the API in step with prompts.PROTOCOL_API.
    """

    def __init__(self, deck: dict, pipette: dict, labware: dict):
        tc = deck.get("thermocycler")
        self._tc_slot = tc["slot"] if isinstance(tc, dict) else None
        self._labware_of: dict[str, dict] = {}     # slot -> labware geometry
        for key, cfg in deck.items():
            if key == "trash" or not isinstance(cfg, dict):
                continue
            slot = cfg["slot"] if key == "thermocycler" else key
            self._labware_of[slot] = {"name": cfg["labware"], **labware.get(cfg["labware"], {})}
        self._trash = deck.get("trash")
        self._min = float(pipette.get("min_volume_ul", 1.0))
        self._max = float(pipette.get("max_volume_ul", 50.0))
        self._tips = [f"{slot}:{w}" for slot, lw in self._labware_of.items()
                      if lw.get("tip_volume_ul") for w in self._wells(lw)]

        self.lid_open = not self._tc_slot
        self.tip: str | None = None                  # the tip on the pipette, if any
        self.tip_source: str | None = None           # where the liquid in it came from
        self.tip_volume = 0.0
        self.transfers: list[dict] = []
        self.thermal: list[dict] = []
        self.commands: list[dict] = []
        self.log: list[str] = []

    # -- geometry -------------------------------------------------------------------

    @staticmethod
    def _wells(lw: dict) -> list[str]:
        rows, cols = int(lw.get("rows", 8)), int(lw.get("cols", 12))
        return [f"{chr(65 + r)}{c + 1}" for c in range(cols) for r in range(rows)]

    def _check_well(self, key: str) -> str:
        if not isinstance(key, str) or ":" not in key:
            raise ProtocolError(f"well {key!r} must be written '<slot>:<well>', e.g. 'B2:A1'")
        slot, well = key.split(":", 1)
        lw = self._labware_of.get(slot)
        if lw is None:
            raise ProtocolError(f"no labware in slot {slot} (deck has {sorted(self._labware_of)})")
        m = _WELL.match(well)
        rows, cols = int(lw.get("rows", 8)), int(lw.get("cols", 12))
        if not m or ord(m.group(1)) - 65 >= rows or not 1 <= int(m.group(2)) <= cols:
            raise ProtocolError(f"{lw['name']} in slot {slot} has no well {well}")
        if lw.get("tip_volume_ul"):
            raise ProtocolError(f"slot {slot} holds tips, not liquid: cannot pipette at {key}")
        if slot == self._tc_slot and not self.lid_open:
            raise ProtocolError(f"the thermocycler lid is closed: open it before pipetting at {key}")
        return key

    def _record(self, op: str, **args) -> None:
        self.commands.append({"op": op, **args})
        text = f"{op} " + ", ".join(f"{k}={v}" for k, v in args.items())
        self.log.append(text.strip())

    # -- tips -----------------------------------------------------------------------

    def pick_up_tip(self) -> None:
        if self.tip:
            raise ProtocolError("pick_up_tip: the pipette already has a tip; drop it first")
        if not self._tips:
            raise ProtocolError("pick_up_tip: out of tips")
        self.tip = self._tips.pop(0)
        self._record("pick_up_tip", location=self.tip)

    def drop_tip(self) -> None:
        if not self.tip:
            raise ProtocolError("drop_tip: there is no tip on the pipette")
        if not self._trash:
            raise ProtocolError("drop_tip: the deck has no trash")
        self.tip, self.tip_source, self.tip_volume = None, None, 0.0
        self._record("drop_tip", location=self._trash)

    # -- liquid ---------------------------------------------------------------------

    def _check_volume(self, what: str, volume_ul: float) -> float:
        volume = float(volume_ul)
        if volume < self._min - 1e-9:
            raise ProtocolError(f"{what}: {volume:g} uL is below the pipette minimum of {self._min:g} uL")
        return volume

    def aspirate(self, volume_ul: float, well: str) -> None:
        volume = self._check_volume("aspirate", volume_ul)
        key = self._check_well(well)
        if not self.tip:
            raise ProtocolError(f"aspirate at {key}: no tip on the pipette")
        if self.tip_source not in (None, key):
            raise ProtocolError(f"aspirate at {key}: the tip still holds liquid from "
                                f"{self.tip_source}; dispense it or drop the tip first")
        if self.tip_volume + volume > self._max + 1e-9:
            raise ProtocolError(f"aspirate at {key}: {self.tip_volume + volume:g} uL exceeds the "
                                f"{self._max:g} uL tip")
        self.tip_source, self.tip_volume = key, self.tip_volume + volume
        self._record("aspirate", volume_ul=round(volume, 4), location=key)

    def dispense(self, volume_ul: float, well: str) -> None:
        volume = float(volume_ul)
        key = self._check_well(well)
        if not self.tip or self.tip_volume <= 1e-9:
            raise ProtocolError(f"dispense at {key}: nothing in the tip")
        if volume > self.tip_volume + 1e-9:
            raise ProtocolError(f"dispense at {key}: {volume:g} uL requested, the tip holds "
                                f"{self.tip_volume:g} uL")
        self.transfers.append({"source": self.tip_source, "dest": key, "volume_ul": volume})
        self.tip_volume -= volume
        if self.tip_volume <= 1e-9:
            self.tip_source, self.tip_volume = None, 0.0
        self._record("dispense", volume_ul=round(volume, 4), location=key)

    def mix(self, repetitions: int, volume_ul: float, well: str) -> None:
        """Aspirate and dispense in place. Moves no liquid between wells."""
        volume = self._check_volume("mix", volume_ul)
        key = self._check_well(well)
        if not self.tip:
            raise ProtocolError(f"mix at {key}: no tip on the pipette")
        if volume > self._max + 1e-9:
            raise ProtocolError(f"mix at {key}: {volume:g} uL exceeds the {self._max:g} uL tip")
        self._record("mix", repetitions=int(repetitions), volume_ul=round(volume, 4), location=key)

    def transfer(self, volume_ul: float, source: str, dest: str, new_tip: str = "always") -> None:
        """Move `volume_ul` from source to dest, splitting it if it exceeds one tipful."""
        if new_tip not in ("always", "once", "never"):
            raise ProtocolError(f"transfer: new_tip must be 'always', 'once' or 'never', not {new_tip!r}")
        volume = self._check_volume("transfer", volume_ul)
        chunks = []
        while volume > 1e-9:
            chunks.append(min(volume, self._max))
            volume -= chunks[-1]
        for i, chunk in enumerate(chunks):
            if new_tip == "always" or (new_tip == "once" and i == 0):
                if self.tip:
                    self.drop_tip()
                self.pick_up_tip()
            self.aspirate(chunk, source)
            self.dispense(chunk, dest)
        if new_tip != "never" and self.tip:
            self.drop_tip()

    # -- thermocycler ---------------------------------------------------------------

    def _need_thermocycler(self, what: str) -> None:
        if not self._tc_slot:
            raise ProtocolError(f"{what}: the deck has no thermocycler")

    def open_lid(self) -> None:
        self._need_thermocycler("open_lid")
        self.lid_open = True
        self._record("open_lid")

    def close_lid(self) -> None:
        self._need_thermocycler("close_lid")
        self.lid_open = False
        self._record("close_lid")

    def hold(self, celsius: float, minutes: float) -> None:
        self._need_thermocycler("hold")
        if self.lid_open:
            raise ProtocolError("hold: close the thermocycler lid before heating the block")
        step = {"kind": "hold", "celsius": float(celsius), "minutes": float(minutes)}
        self.thermal.append(step)
        self._record("hold", celsius=float(celsius), minutes=float(minutes))

    def thermocycle(self, steps: list[dict], cycles: int) -> None:
        self._need_thermocycler("thermocycle")
        if self.lid_open:
            raise ProtocolError("thermocycle: close the thermocycler lid before cycling")
        try:
            clean = [{"celsius": float(s["celsius"]), "seconds": float(s["seconds"])} for s in steps]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("thermocycle: each step needs numeric 'celsius' and 'seconds'") from exc
        if not clean or int(cycles) < 1:
            raise ProtocolError("thermocycle: needs at least one step and one cycle")
        self.thermal.append({"kind": "cycle", "cycles": int(cycles), "steps": clean})
        self._record("thermocycle", cycles=int(cycles), steps=clean)

    def comment(self, text: str) -> None:
        self._record("comment", text=str(text)[:200])


def main() -> int:
    result: dict = {"ok": False, "errors": [], "transfers": [], "thermal": [],
                    "commands": [], "log": []}
    try:
        job = json.loads(sys.stdin.read())
        robot = Robot(job.get("deck") or {}, job.get("pipette") or {}, job.get("labware") or {})
    except Exception as exc:  # noqa: BLE001
        result["errors"] = [f"bad worker input: {exc}"]
        json.dump(result, sys.stdout)
        return 0

    try:
        namespace: dict = {"__name__": "protocol"}
        # The protocol's own stdout would corrupt our JSON, so send it to stderr.
        with contextlib.redirect_stdout(sys.stderr):
            exec(compile(job["source"], "protocol.py", "exec"), namespace)
            run = namespace.get("run")
            if not callable(run):
                raise ProtocolError("the protocol must define run(robot)")
            run(robot)
        if robot.tip:
            robot.log.append("warning: protocol ended with a tip still on the pipette")
        result.update(ok=True)
    except ProtocolError as exc:
        n = len(robot.commands)
        result["errors"] = [f"ProtocolError after {n} command(s): {exc}"]
    except Exception as exc:  # noqa: BLE001
        # A bug in the protocol itself (NameError, a typo'd method) is as useful a signal
        # as a refused command, so report rather than raise.
        result["errors"] = [f"{type(exc).__name__}: {str(exc)[:1500]}"]

    result.update(transfers=robot.transfers, thermal=robot.thermal,
                  commands=robot.commands, log=robot.log)
    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
