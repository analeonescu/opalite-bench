"""The robot protocol API: what the dry run accepts, refuses, and records.

The dry run stands in for the robot's own checks, so each refusal here is one the
hardware would make. Each case is a few lines of protocol through the real subprocess.
"""
import pytest
from reference_protocol import reference_protocol

from opalite_bench.lab.protocol_run import simulate_protocol
from opalite_bench.spec.scenarios import load_spec

SPEC = load_spec()


def dry_run(body: str):
    return simulate_protocol("def run(robot):\n" + "\n".join(f"    {line}" for line in body.splitlines()),
                             SPEC)


@pytest.mark.parametrize("stage", list(SPEC["stages"]))
def test_reference_protocols_pass(stage):
    run = simulate_protocol(reference_protocol(SPEC, stage), SPEC)
    assert run.ok, run.errors


def test_transfer_is_recorded_with_its_volume():
    run = dry_run('robot.open_lid()\nrobot.transfer(2.5, "B2:A1", "B1:A1")')
    assert run.ok, run.errors
    assert [(t.source, t.dest, t.volume_ul) for t in run.transfers] == [("B2:A1", "B1:A1", 2.5)]


def test_large_transfer_is_split_into_tipfuls():
    run = dry_run('robot.transfer(120, "D2:A1", "B2:H12")')
    assert run.ok, run.errors
    assert [t.volume_ul for t in run.transfers] == [50.0, 50.0, 20.0]


def test_mix_moves_no_liquid():
    run = dry_run('robot.pick_up_tip()\nrobot.mix(3, 5, "B2:A1")\nrobot.drop_tip()')
    assert run.ok and run.transfers == []


def test_thermal_steps_are_structured():
    run = dry_run('robot.close_lid()\nrobot.hold(50, 15)\n'
                  'robot.thermocycle([{"celsius": 98, "seconds": 10}, {"celsius": 60, "seconds": 20}], 30)')
    assert run.ok, run.errors
    assert run.holds_temperature(50, 15) and not run.holds_temperature(50, 20)
    assert run.runs_cycles(30) and not run.runs_cycles(31)


@pytest.mark.parametrize("body, refusal", [
    ('robot.transfer(5, "B2:A1", "Z9:A1")', "no labware in slot Z9"),
    ('robot.transfer(5, "B2:A1", "B2:J1")', "has no well J1"),
    ('robot.transfer(5, "B2:A1", "B2-A1")', "must be written"),
    ('robot.transfer(0.5, "B2:A1", "B2:A2")', "below the pipette minimum"),
    ('robot.aspirate(5, "B2:A1")', "no tip"),
    ('robot.transfer(5, "B2:A1", "B1:A1")', "lid is closed"),
    ('robot.open_lid()\nrobot.hold(50, 15)', "close the thermocycler lid"),
    ('robot.pick_up_tip()\nrobot.aspirate(60, "D2:A1")', "exceeds the 50 uL tip"),
    ('robot.pick_up_tip()\nrobot.aspirate(5, "B2:A1")\nrobot.aspirate(5, "B2:A2")', "still holds liquid"),
    ('robot.pick_up_tip()\nrobot.dispense(5, "B2:A1")', "nothing in the tip"),
    ('robot.transfer(5, "C1:A1", "B2:A1")', "holds tips"),
    ('for _ in range(97):\n    robot.pick_up_tip()\n    robot.drop_tip()', "out of tips"),
    ('robot.fly()', "AttributeError"),
])
def test_refusals(body, refusal):
    run = dry_run(body)
    assert not run.ok
    assert refusal in run.errors[0], run.errors


def test_protocol_without_run_is_rejected():
    run = simulate_protocol("x = 1\n", SPEC)
    assert not run.ok and "run(robot)" in run.errors[0]


def test_hanging_protocol_times_out():
    run = simulate_protocol("def run(robot):\n    while True:\n        pass\n", SPEC, timeout_s=2)
    assert not run.ok and "did not finish" in run.errors[0]
