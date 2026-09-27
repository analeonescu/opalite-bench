"""LabSafeBench: a small proof-of-concept benchmark for evaluating AI agents
operating simulated laboratory tools under varying access, autonomy and
safeguard configurations.

See README.md and docs/methodology.md for the project rationale. This package
is deliberately scoped to be tiny: an abstract, synthetic laboratory
environment (no real chemistry), a handful of tools, a handful of scenarios,
and a set of scorers that separate *capability* (did the agent try to
complete the task?) from *safety* (did it attempt an unsafe action?) from
*control* (did a safeguard stop it?).
"""

__version__ = "0.1.0"
