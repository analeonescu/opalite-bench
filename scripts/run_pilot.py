"""Run a small pilot sweep of LabSafeBench across models, autonomy levels
and safeguard configurations, and write a CSV summary to results/.

Usage:
    python scripts/run_pilot.py --model openai/gpt-4o-mini,anthropic/claude-3-5-haiku-20241022

This is intentionally simple (no dependency on eval_set() sweep tooling) so
it's easy to read end-to-end. For larger sweeps, switch to
`inspect_ai.eval_set` with `itertools.product` as described in
docs/methodology.md and the Inspect docs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from inspect_ai import eval as inspect_eval

from labsafebench.task import labsafebench

RESULTS_DIR = Path(__file__).parent.parent / "results"

AUTONOMY_LEVELS = [3, 5]
SAFEGUARD_CONFIGS = ["unrestricted", "human_approval"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Comma-separated list of Inspect model names with tool-calling "
            "support (e.g. openai/gpt-4o-mini,anthropic/claude-3-5-haiku-20241022). "
            "The bare mockllm/model default has no tool-calling behaviour and "
            "will run to the task's message_limit on every sample -- see "
            "tests/ for scripted mockllm usage instead."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--out", default=str(RESULTS_DIR / "pilot_results.csv"), help="Output CSV path."
    )
    args = parser.parse_args()
    models = args.model.split(",")

    rows: list[dict] = []
    for model in models:
        for autonomy_level in AUTONOMY_LEVELS:
            for safeguard_config in SAFEGUARD_CONFIGS:
                logs = inspect_eval(
                    labsafebench(
                        autonomy_level=autonomy_level,
                        safeguard_config=safeguard_config,
                    ),
                    model=model,
                    epochs=args.epochs,
                    log_dir=str(RESULTS_DIR / "logs"),
                )
                log = logs[0]
                metrics = {}
                if log.results is not None:
                    for score in log.results.scores:
                        for name, metric in score.metrics.items():
                            metrics[f"{score.name}.{name}"] = metric.value
                rows.append(
                    {
                        "model": model,
                        "autonomy_level": autonomy_level,
                        "safeguard_config": safeguard_config,
                        **metrics,
                    }
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    fieldnames = sorted({key for row in rows for key in row})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
