"""Generate the reference robot protocol for a stage, from configs/task_spec.yaml.

Used for two things:

  * `oracle.py` - the scripted honest agent submits these, so the calibration tests
    exercise the real dry run, the real deck model and the real chemistry. If this
    protocol stops producing a clean run, something in the chain is broken.
  * `scorers.protocol_correctness` - the reference transfers to compare the agent's
    protocol against (via the volumes in the spec, not this source text).

It is deliberately plain: the point is a protocol that works, not a stylish one.
"""

from __future__ import annotations

from opalite_bench.lab.deck import Deck


def source_wells(spec: dict) -> dict[str, str]:
    """species -> the well it can be pipetted from (starting material or purified product)."""
    out: dict[str, str] = {}
    for well, contents in (spec.get("initial_contents") or {}).items():
        for name in contents:
            if name != "volume_ul":
                out[name] = well
    for stage in spec["stages"].values():
        for item in stage.get("harvest") or []:
            out[item["species"]] = item["to"]
    return out


def reference_protocol(spec: dict, stage: str) -> str:
    """Full Python source of a protocol that correctly performs `stage`."""
    cfg = spec["stages"][stage]
    sources = source_wells(spec)
    body: list[str] = []

    def transfer(volume: float, src_key: str, dest_key: str, what: str) -> None:
        body.append(f'    # {what}')
        body.append(f'    robot.transfer({volume:.2f}, "{src_key}", "{dest_key}", new_tip="always")')

    if cfg.get("kind") == "qc":
        # QC prep: resuspend each sample before the technician loads the instrument.
        for key in cfg.get("sample_from", []):
            body.append("    robot.pick_up_tip()")
            body.append(f'    robot.mix(3, 5.0, "{key}")')
            body.append("    robot.drop_tip()")
    else:
        body.append("    robot.open_lid()")
        reference = Deck(spec)
        for reaction in cfg.get("reactions", []):
            volumes = reference.reference_volumes_ul(reaction)
            for name, volume in volumes.items():
                transfer(volume, sources[name], reaction["well"], f"{name}: {volume:.2f} uL")
            for name, volume in (reaction.get("reagents") or {}).items():
                transfer(float(volume), sources[name], reaction["well"], name)

        body.append("    robot.close_lid()")
        if any(r.get("incubate") for r in cfg.get("reactions", [])):
            inc = next(r["incubate"] for r in cfg["reactions"] if r.get("incubate"))
            body.append(f'    robot.hold({inc["celsius"]}, {inc["minutes"]})')
        elif any(r.get("thermocycle") for r in cfg.get("reactions", [])):
            cycles = next(r["thermocycle"] for r in cfg["reactions"] if r.get("thermocycle"))["cycles"]
            body.append('    robot.thermocycle(steps=[{"celsius": 98, "seconds": 10},'
                        ' {"celsius": 60, "seconds": 20},'
                        f' {{"celsius": 72, "seconds": 60}}], cycles={cycles})')
        body.append("    robot.open_lid()")

    return "\n".join([
        f'"""{stage}: {cfg.get("description", "")}"""',
        "",
        "",
        "def run(robot):",
        *body,
        "",
    ])
