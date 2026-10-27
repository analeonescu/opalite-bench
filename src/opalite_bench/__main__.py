"""`python -m opalite_bench --help`: list the task's -T options.

Inspect's own `--help` covers its CLI flags but not a task's parameters, so this reads
them straight off `gibson_analysis`: its signature gives the names and defaults, its
docstring gives the descriptions, and the allowed values come from the modules that
define them. Nothing here is a hand-written copy, so it cannot drift from the code.
"""
from __future__ import annotations

import inspect as _inspect
import sys
import textwrap

from opalite_bench.lab.qc_harness import MODES
from opalite_bench.lab.simulated import POLICIES
from opalite_bench.spec.scenarios import load_scenarios
from opalite_bench.task import gibson_analysis

TASK_FILE = "src/opalite_bench/task.py"


def _docstring_options(func) -> dict[str, str]:
    """Pull `name: description` blocks out of the task docstring, keeping indented lines."""
    doc = _inspect.getdoc(func) or ""
    names = set(_inspect.signature(func).parameters)
    options: dict[str, list[str]] = {}
    current: str | None = None
    for line in doc.splitlines()[1:]:
        head = line.split(":", 1)
        if line and not line.startswith((" ", "\t")) and len(head) == 2 and head[0] in names:
            current = head[0]
            options[current] = [head[1].strip()]
        elif current and line.strip():
            # Sub-lines (the per-value explanations) are already laid out in the
            # docstring, so keep their indentation instead of re-flowing them.
            options[current].append("  " + line.rstrip())
        elif not line.strip():
            current = None
    return {name: "\n".join(lines) for name, lines in options.items()}


def _choices() -> dict[str, list[str]]:
    """Allowed values, read from wherever they are actually defined."""
    return {
        "backend": ["sim", "robots"],
        "policy": list(POLICIES),
        "allow_physical": ["true", "false"],
        "qc_tools": list(MODES),
        "human": ["auto", "cli"],
        "scenarios": list(load_scenarios()),
        "enable_python": ["true", "false"],
    }


def render_help() -> str:
    signature = _inspect.signature(gibson_analysis)
    descriptions = _docstring_options(gibson_analysis)
    choices = _choices()
    width = 78

    out = [
        "opalite-bench — options for the gibson_analysis task",
        "",
        "Run an eval:",
        f"  inspect eval {TASK_FILE} --model <provider>/<model> [-T name=value ...]",
        "",
        "Task options (-T):",
    ]
    for name, param in signature.parameters.items():
        default = _default(name, param.default)
        out.append(f"\n  -T {name}={_hint(name, choices)}")
        out.append(f"     default: {default}")
        if name in choices and name not in ("enable_python", "allow_physical"):
            out.append(f"     allowed: {', '.join(choices[name])}")
        if body := descriptions.get(name):
            first, *rest = body.splitlines()
            out += textwrap.wrap(first, width=width, initial_indent="     ",
                                 subsequent_indent="     ")
            out += [f"     {line}" for line in rest]

    out += [
        "",
        "Commonly used Inspect flags (full list: inspect eval --help):",
        "  --model <provider>/<model>   the model under test (needs that provider's API key)",
        "  --max-samples N              samples in flight at once; use 1 with -T human=cli",
        "  --limit N                    stop after N samples",
        "  --epochs N                   repeat every sample N times",
        "  --log-dir DIR                where to write the eval log (default: ./logs)",
        "",
        "Examples:",
        f"  inspect eval {TASK_FILE} --model anthropic/<model>",
        f"  inspect eval {TASK_FILE} --model anthropic/<model> \\",
        "     -T scenarios=round1_ambiguous,final_wrong_size",
        f"  inspect eval {TASK_FILE} --model anthropic/<model> -T qc_tools=image_vlm",
        f"  inspect eval {TASK_FILE} --model anthropic/<model> \\",
        "     -T human=cli -T human_judge_model=anthropic/<model> --max-samples 1",
        "",
        "Tests need no model or API key:  pytest -m \"not slow\"",
    ]
    return "\n".join(out)


def _default(name: str, value) -> str:
    if value is None:
        return (f"all {len(load_scenarios())} scenarios" if name == "scenarios" else "not set")
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _hint(name: str, choices: dict[str, list[str]]) -> str:
    if name == "scenarios":
        return "<id>[,<id>...]"
    if name.endswith("_model"):
        return "<provider>/<model>"
    if name in choices:
        return "|".join(c.split(" ")[0] for c in choices[name])
    return "<value>"


#: How Inspect's own eval flags get grouped for `--all`. First match wins; anything
#: unmatched falls into "other".
_FLAG_GROUPS = [
    ("Model", ("model", "-M", "temperature", "top", "max-tokens", "seed", "reasoning",
               "logit", "frequency", "presence", "stop", "cache", "response-schema",
               "best-of", "num-choices", "internal-tools", "parallel-tool")),
    ("Task and samples", ("-T", "task", "sample", "epoch", "limit", "solver", "approval",
                          "tags", "metadata", "env", "dataset")),
    ("Limits and retries", ("retry", "timeout", "attempts", "working", "token-limit",
                            "message-limit", "time-limit", "fail-on", "max-")),
    ("Sandbox", ("sandbox", "docker", "no-sandbox")),
    ("Logging and display", ("log", "display", "trace", "debug", "json", "no-score",
                             "score", "verbose", "quiet")),
]


def _group_of(name: str) -> str:
    for group, keys in _FLAG_GROUPS:
        if any(k in name for k in keys):
            return group
    return "Other"


def render_inspect_flags() -> str:
    """Every `inspect eval` option, read off its click command so the list stays current."""
    try:
        import click

        from inspect_ai._cli.eval import eval_command
    except Exception as exc:  # noqa: BLE001
        return ("Could not read Inspect's own flags "
                f"({type(exc).__name__}); run `inspect eval --help` for them.")

    grouped: dict[str, list[tuple[str, str]]] = {}
    for param in eval_command.params:
        if not isinstance(param, click.Option) or param.hidden:
            continue
        flags = ", ".join(sorted(param.opts, key=len, reverse=True))
        if param.metavar or param.type.name not in ("boolean",):
            flags += f" {param.metavar or param.type.name.upper()}"
        grouped.setdefault(_group_of(" ".join(param.opts)), []).append(
            (flags, (param.help or "").strip()))

    out = ["", "All `inspect eval` flags:"]
    for group, _ in [*_FLAG_GROUPS, ("Other", ())]:
        if not (rows := sorted(grouped.get(group, []))):
            continue
        out += ["", f"  {group}"]
        for flags, help_text in rows:
            out.append(f"    {flags}")
            if help_text:
                out += textwrap.wrap(help_text, width=74, initial_indent="        ",
                                     subsequent_indent="        ")
    return "\n".join(out)


USAGE = """\
usage: python -m opalite_bench [--help | --all | doctor [--embodiment REF]]

  --help   the task's -T options, plus the Inspect flags you will usually want
  --all    the above, plus every `inspect eval` flag
  doctor   check the spec, the deck's reach and the robot setup, without running anything
"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["doctor"]:
        from opalite_bench.doctor import main as doctor
        return doctor(argv[1:])
    if argv and argv[0] not in ("-h", "--help", "help", "-a", "--all", "all"):
        print(f"unknown argument {argv[0]!r}\n{USAGE}", file=sys.stderr)
        return 2
    print(render_help())
    if argv and argv[0] in ("-a", "--all", "all"):
        print(render_inspect_flags())
    else:
        print("\nFor every Inspect flag:  python -m opalite_bench --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
