# Results

This directory holds pilot sweep outputs.

* `logs/` -- raw Inspect `.eval` logs (viewable with `inspect view`).
  Gitignored by default; regenerate with `scripts/run_pilot.py`.
* `pilot_results.csv` -- flattened per-run metrics (model, autonomy_level,
  safeguard_config, and each scorer's metrics), produced by
  `scripts/run_pilot.py`.

To reproduce a pilot run against real models:

To run against real models:

```bash
python scripts/run_pilot.py --model openai/gpt-4o-mini,anthropic/claude-3-5-haiku-20241022
```

No pilot results are currently checked in -- this is a v0.1 scaffold. See
docs/methodology.md for the planned 3 models x 2 autonomy levels x 2
safeguard configs x 5 repeats x 15 scenarios pilot design.
