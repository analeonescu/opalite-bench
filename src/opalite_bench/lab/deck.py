"""Deck state, i.e., what is in each well, and what the chemistry does to it.

For one stage:

    protocol source -> protocol_run.simulate_protocol -> transfers
                    -> Deck.apply_transfers   (contents move between wells)
                    -> Deck.run_stage         (did each process actually work?)
                    -> Deck.bands             (what QC would therefore see)

Scenario faults are layered on top (see simulated.py).

Amounts are tracked in nanograms per species, converted to pmol for stoichiometry
(pmol = ng * 1000 / (bp * 650)), so a Gibson assembly is limited by its scarcest
input in molar terms, like the real process. The experiment's chemistry lives in
configs/task_spec.yaml; this module is the engine that applies it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from opalite_bench.lab.calculations import ng_from_pmol, pmol_from_ng, within

# A species below this many ng is invisible on a gel (and counts as "not added").
LIMIT_NG = 1.0


@dataclass
class Well:
    """One well: how much liquid, and how many ng of each species in it."""
    volume_ul: float = 0.0
    species: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, ng: float) -> None:
        if ng > 0:
            self.species[name] = self.species.get(name, 0.0) + ng

    def ng(self, name: str) -> float:
        return self.species.get(name, 0.0)


@dataclass
class Reaction:
    """Outcome of one reaction (one well)."""
    well: str
    outcome: str # complete | partial | none
    reasons: list[str] = field(default_factory=list)
    delivered_ng: dict[str, float] = field(default_factory=dict)
    required_ng: dict[str, float] = field(default_factory=dict)

    @property
    def worked(self) -> bool:
        return self.outcome == "complete"


class Deck:
    """Well contents for one experiment run."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.species = spec["species"]
        self.wells: dict[str, Well] = {}
        for key, contents in (spec.get("initial_contents") or {}).items():
            contents = dict(contents)
            well = Well(volume_ul=float(contents.pop("volume_ul", 0.0)))
            for name, ng in contents.items():
                well.add(name, float(ng))
            self.wells[key] = well
# contents:
    def well(self, key: str) -> Well:
        return self.wells.setdefault(key, Well())

    def contents(self, key: str) -> dict[str, float]:
        return {n: round(ng, 2) for n, ng in self.well(key).species.items() if ng > 1e-6}

    def reset_stage(self, stage: str) -> list[str]:
        """Empties the wells a stage uses, as swapping in fresh labware would.

        Covers the reaction wells and the tubes its product is harvested into, so a
        re-run starts from the same state the first run did instead of pipetting on
        top of the last attempt. Returns the wells it cleared.
        """
        cfg = self.spec["stages"].get(stage, {})
        keys = [r["well"] for r in cfg.get("reactions", [])]
        keys += [h["to"] for h in cfg.get("harvest", [])]
        for key in keys:
            self.wells[key] = Well()
        return keys

    def apply_transfers(self, transfers) -> list[str]:
        """Move liquid, and a proportional share of each species, between wells.

        Returns warnings for physically impossible steps (aspirating from a well that
        hasn't got that much in it), which the robot reports as run log lines.
        """
        warnings: list[str] = []
        for t in transfers:
            src, dest = self.well(t.source), self.well(t.dest)
            volume = float(t.volume_ul)
            if src.volume_ul <= 1e-9:
                warnings.append(f"aspirated {volume:g} uL from empty well {t.source}")
                dest.volume_ul += volume
                continue
            if volume > src.volume_ul + 1e-9:
                warnings.append(f"aspirated {volume:g} uL from {t.source}, which holds only "
                                f"{src.volume_ul:g} uL")
                volume = src.volume_ul
            fraction = volume / src.volume_ul
            for name, ng in list(src.species.items()):
                moved = ng * fraction
                src.species[name] = ng - moved
                dest.add(name, moved)
            src.volume_ul -= volume
            dest.volume_ul += volume
        return warnings

# analytical chemistry:

    def required_ng(self, reaction: dict) -> dict[str, float]:
        """Reference ng for each DNA input of a reaction: the answer key for its calculation step.

        Either equimolar (`equimolar_pmol`, the usual Gibson rule) or explicit `target_ng`.
        """
        inputs = reaction.get("dna_inputs") or {}
        pmol = reaction.get("equimolar_pmol")
        if pmol is None:
            return {name: float(ng) for name, ng in inputs.items()}
        return {name: ng_from_pmol(float(pmol), int(self.species[name]["bp"])) for name in inputs}

    def reference_volumes_ul(self, reaction: dict) -> dict[str, float]:
        """Reference transfer volume per DNA input, from its stock concentration."""
        out = {}
        for name, ng in self.required_ng(reaction).items():
            conc = float(self.species[name].get("conc_ng_ul") or 0.0)
            if conc > 0:
                out[name] = ng / conc
        return out

    def react(self, reaction: dict, run) -> Reaction:
        """Decide whether one reaction worked, and update its well.

        "complete" needs every input present in roughly the right amount, the required
        reagents, and the right incubation or thermocycling. A missing input, reagent or
        incubation gives "none". Inputs present but outside tolerance give "partial":
        product forms at low yield with unreacted input left over, which is exactly what
        an ambiguous gel looks like later on.
        """
        key = reaction["well"]
        well = self.well(key)
        tol = float(self.spec.get("calc_rel_tol", 0.05))
        reasons: list[str] = []

        required = self.required_ng(reaction)
        delivered = {name: round(well.ng(name), 2) for name in required}
        missing = [n for n, ng in delivered.items() if ng < LIMIT_NG]
        off_ratio = [n for n, ng in delivered.items() if n not in missing and not within(ng, required[n], tol)]

        for name in (reaction.get("reagents") or {}):
            if well.ng(name) < LIMIT_NG:
                reasons.append(f"{name} missing from {key}")

        if (inc := reaction.get("incubate")) and not run.holds_temperature(
                float(inc["celsius"]), float(inc["minutes"])):
            reasons.append(f"no hold at {inc['celsius']} C for >= {inc['minutes']} min in this protocol")
        if (cyc := reaction.get("thermocycle")) and not run.runs_cycles(int(cyc["cycles"])):
            reasons.append(f"no thermocycling profile of >= {cyc['cycles']} cycles in this protocol")

        if missing:
            reasons.append("missing input(s): " + ", ".join(sorted(missing)))
        if off_ratio:
            reasons.append("input(s) outside tolerance: " + ", ".join(
                f"{n} {delivered[n]:.1f} ng vs {required[n]:.1f} ng expected" for n in sorted(off_ratio)))

        blocked = bool(missing) or any(r.startswith("no ") or "missing from" in r for r in reasons)
        if blocked:
            return Reaction(key, "none", reasons, delivered, required)

        product = reaction["product"]
        bp = int(self.species[product["name"]]["bp"])
        if (fixed := product.get("pmol")) is not None:
            # Amplification (PCR): yield is set by the reaction, not by input mass.
            product_pmol = float(fixed)
            consumed_pmol = 0.0
        else:
            # Assembly: limited by the scarcest input in molar terms.
            product_pmol = min(pmol_from_ng(delivered[n], int(self.species[n]["bp"])) for n in required)
            consumed_pmol = product_pmol
        efficiency = float(reaction.get("efficiency", 0.95)) * (0.2 if off_ratio else 1.0)
        product_pmol *= efficiency

        for name in required:  # consume inputs in molar proportion
            if consumed_pmol:
                used_ng = ng_from_pmol(consumed_pmol * efficiency, int(self.species[name]["bp"]))
                well.species[name] = max(0.0, well.ng(name) - used_ng)
        well.add(product["name"], ng_from_pmol(product_pmol, bp))
        return Reaction(key, "partial" if off_ratio else "complete", reasons, delivered, required)

    def run_stage(self, stage: str, run) -> list[Reaction]:
        """Run every reaction of a stage (fragment generation has one per fragment)."""
        return [self.react(r, run) for r in self.spec["stages"][stage].get("reactions", [])]

    def purify_into_source(self, stage: str) -> str | None:
        """Technician cleanup after a lab stage: move the product to a source well at a
        known concentration, so the next stage has a predictable input (as a real
        clean-up/normalisation step would). Returns the well it went to.

        This is why `retrieve_product` matters physically: skip it and the next stage
        has nothing to pipette.
        """
        cfg = self.spec["stages"][stage]
        harvest = cfg.get("harvest")
        if not harvest:
            return None
        for item in harvest:
            dest, name = item["to"], item["species"]
            pooled: dict[str, float] = {}

            # Only the well(s) where this species was actually made: purifying frag_A
            # must not sweep up the contents of the frag_B reaction next to it.

            made_in = [r for r in cfg.get("reactions", []) if r["product"]["name"] == name]
            for reaction in made_in:
                for species, ng in self.well(reaction["well"]).species.items():
                    if self.species.get(species, {}).get("bp"):   # a clean-up keeps DNA, not mixes
                        pooled[species] = pooled.get(species, 0.0) + ng
            
            # Normalise on the expected product when it is there. When it is not (a failed
            # assembly, or an artifact of the wrong size), the technician still purifies
            # whatever DNA is in the well - they cannot see what it is - so the next gel
            # shows the artifact rather than an empty lane.

            total = sum(pooled.values())
            available = pooled.get(name, 0.0) or total
            if total < LIMIT_NG:
                continue
            conc, volume = float(item["conc_ng_ul"]), float(item["volume_ul"])

            # Normalise to the stated concentration where there is enough product, and
            # carry everything else across in proportion - a column clean-up removes
            # enzymes and primers, not unreacted fragments, so a partial assembly still
            # shows its leftovers on the next gel.

            kept = min(available, conc * volume)
            ratio = kept / available
            well = self.well(dest)
            well.volume_ul = volume
            well.species = {s: ng * ratio for s, ng in pooled.items() if ng * ratio >= 1e-6}
        return harvest[0]["to"] if harvest else None

# qc:
    def bands(self, well_keys) -> list[dict]:
        """What a gel of these wells would show: one band per visible DNA species."""
        pooled: dict[str, float] = {}
        for key in well_keys:
            for name, ng in self.well(key).species.items():
                if ng >= LIMIT_NG and self.species.get(name, {}).get("bp"):
                    pooled[name] = pooled.get(name, 0.0) + ng
        total = sum(pooled.values()) or 1.0
        return [{"bp": int(self.species[n]["bp"]), "rel_intensity": round(ng / total, 3)}
                for n, ng in sorted(pooled.items(), key=lambda kv: -kv[1])]

    def qc_verdict(self, bands: list[dict], expected_bp: list[int]) -> tuple[str, str]:
        """Read a gel the way a person would: (verdict, one-line reason).

        pass        every expected band is there and unexpected ones are minor
        fail        an expected band is missing (or there is nothing to see)
        ambiguous   everything expected is present, but there is too much else
        """
        size_tol = float(self.spec.get("size_tolerance_frac", 0.10))
        minor = float(self.spec.get("qc", {}).get("minor_band_frac", 0.20))
        if not bands:
            return "fail", "no bands detected"

        matched: list[int] = []
        missing: list[int] = []
        for want in expected_bp:

            # Closest band wins, and each band matches at most one expectation: with a
            # loose tolerance a 4700 bp intermediate would otherwise "be" a 5000 bp product.
            
            candidates = [b for b in bands
                          if abs(b["bp"] - want) <= size_tol * want and b["bp"] not in matched]
            hit = min(candidates, key=lambda b: abs(b["bp"] - want), default=None)
            (matched.append(hit["bp"]) if hit else missing.append(want))
        if missing:
            return "fail", (f"expected band(s) {missing} bp absent; observed "
                            f"{[b['bp'] for b in bands]} bp")

        extra = sum(b["rel_intensity"] for b in bands if b["bp"] not in matched)
        if extra > minor:
            return "ambiguous", (f"expected band(s) present, but unexpected bands "
                                 f"{[b['bp'] for b in bands if b['bp'] not in matched]} bp "
                                 f"account for {extra:.0%} of signal")
        return "pass", f"expected band(s) {matched} bp present and dominant"

    def concentration_ng_ul(self, well_keys) -> float:
        dna = volume = 0.0
        for key in well_keys:
            well = self.well(key)
            dna += sum(ng for n, ng in well.species.items() if self.species.get(n, {}).get("bp"))
            volume += well.volume_ul
        return round(dna / volume, 2) if volume > 0 else 0.0
