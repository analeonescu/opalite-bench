"""The -T help text is generated from the task, so it cannot go stale. These tests
check that it stays generated rather than drifting into a hand-written list."""
import inspect as _inspect

from opalite_bench.__main__ import main, render_help
from opalite_bench.lab.qc_harness import MODES
from opalite_bench.spec.scenarios import load_scenarios
from opalite_bench.task import gibson_analysis
import click
from inspect_ai._cli.eval import eval_command
from opalite_bench.__main__ import render_inspect_flags

HELP = render_help()


def test_every_task_parameter_is_listed():
    for name in _inspect.signature(gibson_analysis).parameters:
        assert f"-T {name}=" in HELP, f"{name} is missing from the help"


def test_every_qc_mode_and_scenario_is_listed():
    for mode in MODES:
        assert mode in HELP, mode
    for scenario in load_scenarios():
        assert scenario in HELP, scenario


def test_defaults_come_from_the_signature():
    assert "default: summary" in HELP          # qc_tools
    assert "default: auto" in HELP             # human
    assert "default: all 7 scenarios" in HELP


def test_it_points_at_the_inspect_flags_it_does_not_own():
    assert "inspect eval --help" in HELP
    assert "--max-samples" in HELP and "--model" in HELP


def test_help_exits_cleanly_and_anything_else_does_not(capsys):
    assert main(["--help"]) == 0
    assert main([]) == 0
    assert main(["--bogus"]) == 2
    assert "Task options" in capsys.readouterr().out


# ------------------------------------------------------------- --all (Inspect flags)

def test_all_mode_lists_inspect_flags_read_from_inspect_itself():
    """Read off Inspect's own click command, so the list cannot go stale."""


    text = render_inspect_flags()
    visible = [p for p in eval_command.params
               if isinstance(p, click.Option) and not p.hidden]
    assert len(visible) > 50, "sanity: inspect eval has many options"
    for option in visible:
        longest = max(option.opts, key=len)
        assert longest in text, f"{longest} missing from --all"


def test_all_mode_groups_the_flags():

    text = render_inspect_flags()
    for group in ("Model", "Task and samples", "Limits and retries", "Logging and display"):
        assert f"  {group}" in text, group


def test_default_help_stays_short_and_points_at_all(capsys):
    main(["--help"])
    out = capsys.readouterr().out
    assert "python -m opalite_bench --all" in out
    assert len(out.splitlines()) < 100, "the default help should stay readable"
