"""`python -m opalite_bench doctor`: check a setup before running anything.

Nothing here constructs an embodiment, starts a model or moves a robot. It checks that
the spec is self-consistent, that every well the deck declares is inside the arm's
workspace, and that what `-T backend=robots` would need is importable.
"""
from __future__ import annotations

import importlib.util
from importlib import metadata

from opalite_bench.lab.robots import Geometry, resolve_factory
from opalite_bench.spec.scenarios import load_scenarios, load_spec


def _deck_wells(geo: Geometry) -> list[str]:
    return [f"{slot}:{chr(65 + r)}{c + 1}" for slot, lw in geo.slots.items()
            for r in range(int(lw.get("rows", 8))) for c in range(int(lw.get("cols", 12)))]


def checks(embodiment: str | None = None) -> list[tuple[str, str, str]]:
    """(status, what, detail) for each check; status is ok, warn or FAIL."""
    out: list[tuple[str, str, str]] = []
    try:
        spec, scenarios = load_spec(), load_scenarios()
        out.append(("ok", "spec", f"{len(spec['stages'])} stages, {len(scenarios)} scenarios"))
    except Exception as exc:  # noqa: BLE001
        return [("FAIL", "spec", f"{type(exc).__name__}: {exc}")]

    geo = Geometry(spec)
    missing = [lw for lw in {c.get("labware") for c in spec["deck"].values() if isinstance(c, dict)}
               if lw not in (spec.get("labware") or {})]
    out.append(("FAIL", "labware", f"no geometry for {missing}") if missing
               else ("ok", "labware", f"geometry for all {len(geo.slots)} slots"))
    if missing:
        return out

    wells = set(_deck_wells(geo))
    stray = [w for w in [*spec.get("initial_contents", {}), *spec.get("restricted_wells", {})]
             if w not in wells]
    out.append(("FAIL", "wells", f"not on the deck: {stray}") if stray
               else ("ok", "wells", "starting material and restricted wells are on the deck"))

    workspace = (spec.get("robot") or {}).get("workspace_mm") or {}
    off = []
    for key in sorted(wells):
        x, y = geo.well_xy(key)
        for axis, value in (("x", x), ("y", y)):
            lo, hi = workspace.get(axis, (float("-inf"), float("inf")))
            if not lo <= value <= hi:
                off.append(f"{key} {axis}={value:g}")
    out.append(("FAIL", "reach", f"outside the workspace: {off[:5]}") if off
               else ("ok", "reach", f"all {len(wells)} wells inside the arm's workspace"))

    try:
        out.append(("ok", "inspect-robots", metadata.version("inspect-robots")))
    except metadata.PackageNotFoundError:
        out.append(("warn", "inspect-robots", "not installed; needed only for a real "
                    "embodiment (pip install -e '.[robots]')"))

    out.append(("ok", "device claims", "fcntl available") if importlib.util.find_spec("fcntl")
               else ("warn", "device claims", "no fcntl on this platform: physical "
                     "embodiments cannot be claimed, so run one eval at a time"))

    if embodiment and embodiment != "kinematic":
        try:
            resolve_factory(embodiment)
            out.append(("ok", "embodiment", f"{embodiment} importable (not constructed)"))
        except Exception as exc:  # noqa: BLE001
            out.append(("FAIL", "embodiment", f"{embodiment}: {type(exc).__name__}: {exc}"))
    else:
        out.append(("ok", "embodiment", "kinematic (built-in, no hardware)"))
    return out


def main(argv: list[str]) -> int:
    embodiment = None
    if argv[:1] == ["--embodiment"] and len(argv) == 2:
        embodiment = argv[1]
    elif argv:
        print("usage: python -m opalite_bench doctor [--embodiment pkg.module:factory]")
        return 2
    results = checks(embodiment)
    for status, what, detail in results:
        print(f"  {status:<4}  {what:<15} {detail}")
    return 1 if any(status == "FAIL" for status, _, _ in results) else 0
