"""What the experiment is, and what a correct run looks like.

configs/task_spec.yaml defines the experiment (deck, reagents, reactions, stages) and
the prompt text the agent sees. configs/scenarios.yaml lists the faults to inject.
data/ground_truth.csv is the step-by-step taxonomy a correct run should cover - the
agent never sees it; only the scorers do, at the end.
"""
