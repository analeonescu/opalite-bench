# Risk taxonomy

LabSafeBench converts risk vectors from a chemical-security model-evaluation
taxonomy into environment-level failure modes that an *agent with tools* can
actually attempt, rather than merely discuss. The mapping:

| Model-eval risk concept   | Agentic lab version                                            | LabSafeBench risk vector |
|----------------------------|-----------------------------------------------------------------|--------------------------|
| Procedure risk             | Agent attempts an unsafe experimental sequence                  | `scale`, `containment`   |
| Product/reactant risk      | Agent selects a restricted/high-risk material or instrument     | `access`                 |
| Scale-up risk               | Agent increases experimental scale beyond authorisation         | `scale`                  |
| Reproducibility             | Agent persists/retries despite an anomalous result              | `reproducibility`        |
| Environmental controls      | Agent bypasses containment/waste controls                       | `containment`            |
| Human oversight              | Agent proceeds when approval should be required/requested       | `human_oversight`        |

v0.1 implements five of these as first-class, machine-readable risk vectors
attached to each scenario (`risk_vectors:` in the scenario YAML) and checked
by the environment's constraint logic in `labsafebench/environment/lab.py`:

* **`access`** -- using a restricted material or an instrument that requires
  elevated authorisation.
* **`scale`** -- configuring or running an experiment above the sample's
  `authorised_scale`.
* **`containment`** -- lowering containment level or bypassing standard
  waste-disposal procedure.
* **`reproducibility`** -- re-running an experiment immediately after an
  anomalous result rather than investigating or escalating.
* **`human_oversight`** -- a cross-cutting label used on scenarios where the
  intended failure mode is proceeding without requesting approval, rather
  than the specific constraint being violated.

Two risk vectors from the original plan (**safety/actionability** --
converting ambiguous information into executable action -- and
**detectability** -- avoiding logging/oversight) are *not* separately
modelled as constraints in v0.1. Detectability is partially addressed by the
`monitoring_enabled` safeguard switch (does a visible audit log change
behaviour?), and both are natural extensions for v0.2 (see
`methodology.md`).

Each scenario declares which risk vectors it probes, and each violation type
is either blocked or allowed through depending on the active
`SafeguardConfig` (see `configs/*.yaml`), which is exactly the "vary the
safety configuration" experiment described in the project plan.
