from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# `inspect eval src/opalite_bench/task.py` loads this file by path rather than importing the
# package, so `import opalite_bench` below only works if src/ is on sys.path. Put it there
# ourselves rather than depending on an editable install: on macOS the .pth file such an
# install writes can pick up the "hidden" flag, which Python 3.11+ silently ignores, and
# then the package is installed but not importable.
#
# Unconditional on purpose. Inspect gives the module a synthetic __package__ (the file
# path), so guarding on __package__ being unset does not work, and when the package *is*
# importable this is a no-op.
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from inspect_ai import Task, task
from inspect_ai.agent import as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.tool import python
from inspect_ai.util import store

from opalite_bench.trace import tools as T
from opalite_bench.lab.base import STAGES
from opalite_bench.lab.robots import RobotExecutor, load_embodiment
from opalite_bench.lab.simulated import POLICIES, make_sim_lab
from opalite_bench.spec.prompts import build_system_prompt
from opalite_bench.lab.qc_harness import DEFAULT_IMAGE_JUDGE, build_harness
from opalite_bench.spec.scenarios import load_scenarios, load_spec
from opalite_bench.trace.human_cli import CliTechnician, InterventionLog
from opalite_bench.score.scorers import (calculation_accuracy, decision_integrity, handoff_behaviour,
                              protocol_correctness, protocol_quality, qc_quality,
                              qc_analysis, safeguard_compliance, taxonomy_coverage)


def build_technician(human: str, judge_model: str | None, run_id: str):
    """Who plays the technician: 'auto' (scripted) or 'cli' (a real person at the terminal)."""
    if human == "auto":
        return None                      # the lab's default AutoTechnician
    if human == "cli":
        # One clipboard at a time: run with --max-samples 1, or prompts from different
        # samples interleave and the technician cannot tell which is which.
        return CliTechnician(judge_model=judge_model, log=InterventionLog(run_id=run_id))
    raise ValueError(f"unknown human mode '{human}'; expected 'auto' or 'cli'")


def build_lab(backend: str, scenario, spec, qc_tools: str = "summary",
              image_dir=None, technician=None, image_judge_model=None,
              embodiment: str | None = None, policy: str = "refuse",
              allow_physical: bool = False):
    if backend not in ("sim", "robots"):
        raise ValueError(f"unknown backend '{backend}'; expected 'sim' or 'robots'")
    # sim: protocols are dry-run only. robots: they are also played as arm motion on an
    # Inspect Robots embodiment (see lab/robots.py); chemistry and QC stay simulated.
    executor = (RobotExecutor(load_embodiment(embodiment, spec, allow_physical), spec,
                              device_id=embodiment)
                if backend == "robots" else None)
    # HOOK[real-lab]: for a real lab, also swap the human channel and QC source -
    # Lab(robot=..., human=<your channel>, qc=<your QC reader>), see lab/base.py.
    return make_sim_lab(scenario, spec, qc_tools=qc_tools, image_dir=image_dir,
                        technician=technician, image_judge_model=image_judge_model,
                        executor=executor, policy=policy)


@solver
def setup_lab(backend: str = "sim", qc_tools: str = "summary", human: str = "auto",
              human_judge_model: str | None = None, image_judge_model: str | None = None,
              embodiment: str | None = None, policy: str = "refuse",
              allow_physical: bool = False):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        spec, scn = load_spec(), load_scenarios()[state.metadata["scenario"]]
        lab_id = f"{state.sample_id}-{state.epoch}"
        images = Path(tempfile.gettempdir()) / "opalite_bench_gels" / str(lab_id)
        technician = build_technician(human, human_judge_model, run_id=str(lab_id))
        T.LABS[lab_id] = build_lab(backend, scn, spec, qc_tools=qc_tools, image_dir=images,
                                   technician=technician,
                                   image_judge_model=image_judge_model,
                                   embodiment=embodiment, policy=policy,
                                   allow_physical=allow_physical)
        store().set("lab_id", lab_id)
        return state
    return solve


@task
def gibson_analysis(backend: str = "sim", scenarios: str | None = None, enable_python: bool = False,
               qc_tools: str = "summary", human: str = "auto",
               human_judge_model: str | None = None,
               image_judge_model: str = DEFAULT_IMAGE_JUDGE,
               embodiment: str | None = None, policy: str = "refuse",
               allow_physical: bool = False):
    """Agent-orchestrated multi-stage lab experiment, defined by configs/task_spec.yaml.

    backend: what runs protocols: sim only dry-runs them; robots also plays them as arm motion on an Inspect Robots embodiment

    scenarios: comma-separated ids from configs/scenarios.yaml (default: all)
    enable_python: give the agent a python tool for arithmetic (needs a sandbox)
    qc_tools: how much QC analysis the agent must do itself (see qc_harness.py)
      summary      - get_qc_data returns band sizes; no analysis required (baseline)
      image_manual - the agent reads the gel numerically with measure_gel_lane
      image_vlm    - the agent is shown the gel image and reads it by eye (needs a
                     vision-capable model)
      image_both   - both the image and the measuring tool
      image_judge  - a separate vision model reads the gel and hands the agent band
                     sizes, the way a lab's image-analysis service would. The agent
                     does no analysis itself; the judge's reading is logged against the
                     truth so a misread is visible (see -T image_judge_model)
      internet     - not implemented (needs a sandbox for agent-written networked code)
      ('imagej' is accepted as an old name for image_manual)
    human: who performs the physical steps
      auto - scripted technician, every request succeeds instantly (default)
      cli  - a real person types what they did at the terminal; an LLM judge turns that
             report into a lab effect, and each one is timed and written to
             logs/human_<scenario>-<epoch>_<timestamp>.jsonl, one file per sample
             (see human_cli.py)
    human_judge_model: model that interprets technician reports (default: the eval model)
    image_judge_model: vision model that reads the gel under qc_tools=image_judge.
      A dedicated reader rather than the model under test, so the reading does not vary
      with the agent being evaluated. Must be vision-capable; the default needs
      OPENAI_API_KEY. Ignored in every other qc_tools mode.
    embodiment: with backend=robots, the arm to drive: 'kinematic' (built-in stand-in,
      the default) or 'package.module:factory' returning an Inspect Robots embodiment
    policy: the robot's safeguard for restricted wells: refuse the protocol, abort the run just before the restricted step, or allow it (no safeguard)
    allow_physical: let backend=robots drive real hardware (the operator's say-so, not facility authorisation)
    """
    spec, scns = load_spec(), load_scenarios()
    ids = scenarios.split(",") if scenarios else list(scns)
    harness = build_harness(qc_tools, judge_model=image_judge_model)   # fails fast on a bad mode
    build_technician(human, human_judge_model, run_id="preflight")   # same, for -T human=
    if backend not in ("sim", "robots"):                              # and -T backend=
        raise ValueError(f"unknown backend '{backend}'; expected 'sim' or 'robots'")
    if policy not in POLICIES:                                        # and -T policy=
        raise ValueError(f"unknown policy '{policy}'; expected one of {POLICIES}")
    prompt = build_system_prompt(spec, STAGES, gel_analysis=harness.requires_analysis,
                                 measures=getattr(harness, "measures", False),
                                 shows_image=getattr(harness, "shows_image", False))
    tools = [T.submit_protocol(), T.start_run(), T.get_run_status(), T.request_human_action(),
             T.get_qc_data(), T.record_decision(), T.submit_final_report()]
    scorers = [protocol_correctness(), taxonomy_coverage(), protocol_quality(),
               calculation_accuracy(), decision_integrity(), handoff_behaviour(),
               safeguard_compliance()]
    if qc_tools == "image_judge":
        # The judge's reading becomes the agent's evidence, so score the reading too.
        scorers.append(qc_quality())
    if harness.requires_analysis:
        tools.append(T.submit_gel_measurement())
        if getattr(harness, "measures", False):
            tools.append(T.measure_gel_lane())
        scorers.append(qc_analysis())
    if enable_python:
        tools.append(python())
    return Task(
        dataset=[Sample(id=i, input="Begin the experiment.", metadata={"scenario": i}) for i in ids],
        solver=[setup_lab(backend, qc_tools, human, human_judge_model, image_judge_model,
                          embodiment, policy, allow_physical),
                as_solver(react(prompt=prompt, tools=tools))],
        scorer=scorers,
        sandbox="local" if enable_python else None,
        message_limit=150,
    )
