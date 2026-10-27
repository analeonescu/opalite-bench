"""backend=robots: dry-run protocols compiled to motion and played on an embodiment.

The kinematic arm works out every transfer from where the plunger moved, without
seeing the protocol. If the compiler is right, that matches the dry run exactly.
"""
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.model import get_model
from oracle import oracle_outputs
from reference_protocol import reference_protocol

from opalite_bench.lab.protocol_run import simulate_protocol
from opalite_bench.lab.robots import (
    ACTION_DIMS,
    DeviceCommand,
    KinematicArm,
    RobotExecutor,
    Waypoint,
    compile_plan,
    load_embodiment,
)
from opalite_bench.lab.simulated import make_sim_lab
from opalite_bench.spec.scenarios import load_scenarios, load_spec
from opalite_bench.task import gibson_analysis

SPEC = load_spec()
SCNS = load_scenarios()
LAB_STAGES = [s for s, c in SPEC["stages"].items() if c.get("kind") == "lab"]


@pytest.mark.parametrize("stage", list(SPEC["stages"]))
def test_motion_reproduces_the_dry_run_transfers(stage):
    run = simulate_protocol(reference_protocol(SPEC, stage), SPEC)
    arm = KinematicArm(SPEC)
    report = RobotExecutor(arm, SPEC).execute(stage, run)
    assert report.ok, report.message
    assert arm.transfers == [(t.source, t.dest, round(t.volume_ul, 4)) for t in run.transfers]
    assert report.transfers_completed == len(run.transfers)


def test_plan_keeps_thermal_steps_off_the_arm():
    run = simulate_protocol(reference_protocol(SPEC, "gibson_round1"), SPEC)
    plan = compile_plan(run.commands, SPEC)
    devices = [p.op for p in plan if isinstance(p, DeviceCommand)]
    assert devices == ["open_lid", "close_lid", "hold", "open_lid"]
    assert all(len(ACTION_DIMS) == 5 for p in plan if isinstance(p, Waypoint))


def test_travel_happens_at_safe_height():
    run = simulate_protocol(reference_protocol(SPEC, "fragment_generation"), SPEC)
    plan = [p for p in compile_plan(run.commands, SPEC) if isinstance(p, Waypoint)]
    safe = SPEC["robot"]["safe_z_mm"]
    for before, after in zip(plan, plan[1:]):
        if (before.x, before.y) != (after.x, after.y):
            assert before.z == after.z == safe, (before, after)


class FaultyArm(KinematicArm):
    """Stalls after a fixed number of steps, like a gripper or axis fault."""

    def __init__(self, spec, fail_after):
        super().__init__(spec)
        self.fail_after, self.steps = fail_after, 0

    def step(self, action):
        self.steps += 1
        if self.steps > self.fail_after:
            raise RuntimeError("axis stall")
        super().step(action)


def _start(lab, stage):
    sim = lab.robot
    pid = sim.upload_protocol(reference_protocol(SPEC, stage), stage)
    sim.apply_effect("plate_loaded", stage)
    rid = sim.start_run(pid)
    return [sim.get_run(rid) for _ in range(3)][-1]


def test_arm_fault_fails_the_run_and_only_finished_transfers_land():
    arm = FaultyArm(SPEC, fail_after=40)
    lab = make_sim_lab(SCNS["clean_success"], SPEC, executor=RobotExecutor(arm, SPEC))
    state = _start(lab, "fragment_generation")
    assert state.status == "failed" and "axis stall" in state.log[-1]
    event = next(e for e in lab.log.events if e.name == "robot_execution")
    assert not event.data["ok"]
    landed = event.data["transfers_completed"]
    assert 0 < landed < len(simulate_protocol(reference_protocol(SPEC, "fragment_generation"), SPEC).transfers)
    assert len(arm.transfers) == landed


def test_protocol_off_the_workspace_is_rejected():
    spec = {**SPEC, "robot": {**SPEC["robot"], "workspace_mm": {"x": [0, 100]}}}
    lab = make_sim_lab(SCNS["clean_success"], spec, executor=RobotExecutor(KinematicArm(spec), spec))
    v = lab.robot.validate_protocol(reference_protocol(spec, "fragment_generation"), "fragment_generation")
    assert not v.ok and "outside the arm's workspace" in v.errors[0]


def test_load_embodiment():
    assert isinstance(load_embodiment(None, SPEC), KinematicArm)
    assert isinstance(load_embodiment("kinematic", SPEC), KinematicArm)
    with pytest.raises(ValueError):
        load_embodiment("not-a-reference", SPEC)


def test_unknown_backend_fails_fast():
    with pytest.raises(ValueError, match="unknown backend"):
        gibson_analysis(backend="opentrons")


@pytest.mark.slow
@pytest.mark.parametrize("sid", ["clean_success", "final_wrong_size"])
def test_oracle_scores_the_same_on_the_robot(sid, tmp_path):
    """Playing protocols on the arm must not change what the honest agent achieves."""
    model = get_model("mockllm/model", custom_outputs=oracle_outputs(SCNS[sid]))
    log = inspect_eval(gibson_analysis(scenarios=sid, backend="robots"), model=model,
                       log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success", log.error
    scores = {k: v.value for k, v in log.samples[0].scores.items()}
    assert all(v == 1.0 for v in scores["decision_integrity"].values()), scores
    assert scores["protocol_correctness"]["inputs_delivered_correctly"] == 1.0
