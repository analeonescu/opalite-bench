"""The simulated lab that sits between the agent and the robot.

The agent's protocol is dry-run against the robot API (and, with backend=robots,
played as motion on an Inspect Robots embodiment); the transfers it performs are
applied to well contents; QC results are computed from those contents.
Scenario faults are applied on top, the way a real robot or instrument would fail.
"""
