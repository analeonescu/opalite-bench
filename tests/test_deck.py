"""Unit tests for the deck model: transfers, chemistry and gel reading.

These use a tiny hand-written spec and a stub protocol run, so they are fast and
independent of the robot dry run. test_end2end_mock.py covers the same engine with real
simulated protocols.
"""
import pytest

from opalite_bench.lab.deck import LIMIT_NG, Deck
from opalite_bench.lab.protocol_run import Transfer

SPEC = {
    "calc_rel_tol": 0.05,
    "size_tolerance_frac": 0.05,
    "qc": {"minor_band_frac": 0.20},
    "species": {
        "frag_X": {"bp": 1000, "conc_ng_ul": 10.0},
        "frag_Y": {"bp": 2000, "conc_ng_ul": 10.0},
        "product": {"bp": 3000, "conc_ng_ul": 10.0},
        "junk": {"bp": 500},
        "mix": {},
    },
    "initial_contents": {
        "B2:A1": {"frag_X": 100.0, "volume_ul": 10.0},
        "B2:A2": {"frag_Y": 100.0, "volume_ul": 10.0},
        "D2:A1": {"mix": 1000.0, "volume_ul": 1000.0},
    },
    "stages": {
        "assemble": {
            "kind": "lab",
            "reactions": [{
                "well": "B1:A1",
                "equimolar_pmol": 0.05,
                "dna_inputs": {"frag_X": None, "frag_Y": None},
                "reagents": {"mix": 10.0},
                "incubate": {"celsius": 50, "minutes": 15},
                "product": {"name": "product"},
            }],
            "harvest": [{"species": "product", "to": "B2:C1",
                         "conc_ng_ul": 20.0, "volume_ul": 6.0}],
        },
        "check": {"kind": "qc", "sample_from": ["B2:C1"], "expect_bands_bp": [3000]},
    },
}
REACTION = SPEC["stages"]["assemble"]["reactions"][0]


class StubRun:
    """Stands in for a simulated protocol: says whether conditions were met."""
    def __init__(self, incubated=True, cycled=True):
        self._incubated, self._cycled = incubated, cycled

    def holds_temperature(self, celsius, minutes, tol_c=2.0):
        return self._incubated

    def runs_cycles(self, repetitions, denature_c=90.0):
        return self._cycled


def deck_with(**ng_by_species) -> Deck:
    """A deck whose reaction well already holds the given amounts."""
    deck = Deck(SPEC)
    well = deck.well("B1:A1")
    well.volume_ul = 25.0
    for name, ng in ng_by_species.items():
        well.add(name, ng)
    return deck


# ------------------------------------------------------------------- transfers

def test_transfer_moves_a_proportional_share_of_contents():
    deck = Deck(SPEC)
    deck.apply_transfers([Transfer("B2:A1", "B1:A1", 2.5)])   # a quarter of 10 uL
    assert deck.well("B1:A1").ng("frag_X") == pytest.approx(25.0)
    assert deck.well("B2:A1").ng("frag_X") == pytest.approx(75.0)
    assert deck.well("B2:A1").volume_ul == pytest.approx(7.5)


def test_aspirating_more_than_a_well_holds_is_reported():
    deck = Deck(SPEC)
    warnings = deck.apply_transfers([Transfer("B2:A1", "B1:A1", 25.0)])
    assert warnings and "holds only" in warnings[0]
    assert deck.well("B1:A1").ng("frag_X") == pytest.approx(100.0), "takes what is there"


def test_aspirating_from_an_empty_well_is_reported():
    deck = Deck(SPEC)
    warnings = deck.apply_transfers([Transfer("B2:D9", "B1:A1", 5.0)])
    assert warnings and "empty well" in warnings[0]
    assert deck.well("B1:A1").species == {}


def test_a_self_transfer_changes_nothing():
    """pip.mix() shows up as aspirate+dispense in the same well."""
    deck = Deck(SPEC)
    deck.apply_transfers([Transfer("B2:A1", "B2:A1", 5.0)])
    assert deck.well("B2:A1").ng("frag_X") == pytest.approx(100.0)
    assert deck.well("B2:A1").volume_ul == pytest.approx(10.0)


# -------------------------------------------------------------- reference amounts

def test_equimolar_reference_follows_the_pmol_rule():
    required = Deck(SPEC).required_ng(REACTION)
    assert required["frag_X"] == pytest.approx(0.05 * 1000 * 650 / 1000)   # 32.5 ng
    assert required["frag_Y"] == pytest.approx(0.05 * 2000 * 650 / 1000)   # 65.0 ng


def test_reference_volumes_come_from_stock_concentration():
    volumes = Deck(SPEC).reference_volumes_ul(REACTION)
    assert volumes["frag_X"] == pytest.approx(3.25)


# ---------------------------------------------------------------------- reactions

def test_correct_inputs_give_a_complete_reaction():
    deck = deck_with(frag_X=32.5, frag_Y=65.0, mix=10.0)
    result = deck.react(REACTION, StubRun())
    assert result.outcome == "complete" and not result.reasons
    assert deck.well("B1:A1").ng("product") > 0


def test_a_missing_input_blocks_the_reaction():
    deck = deck_with(frag_Y=65.0, mix=10.0)
    result = deck.react(REACTION, StubRun())
    assert result.outcome == "none"
    assert any("missing input" in r for r in result.reasons)
    assert deck.well("B1:A1").ng("product") == 0


def test_a_missing_reagent_blocks_the_reaction():
    deck = deck_with(frag_X=32.5, frag_Y=65.0)
    result = deck.react(REACTION, StubRun())
    assert result.outcome == "none"
    assert any("mix missing" in r for r in result.reasons)


def test_skipping_the_incubation_blocks_the_reaction():
    deck = deck_with(frag_X=32.5, frag_Y=65.0, mix=10.0)
    result = deck.react(REACTION, StubRun(incubated=False))
    assert result.outcome == "none"
    assert any("no hold at 50" in r for r in result.reasons)


def test_wrong_stoichiometry_gives_a_partial_reaction():
    """Everything present but the ratio is off: product forms at low yield, inputs left over."""
    deck = deck_with(frag_X=16.0, frag_Y=65.0, mix=10.0)      # half the frag_X
    result = deck.react(REACTION, StubRun())
    assert result.outcome == "partial"
    assert any("outside tolerance" in r for r in result.reasons)
    well = deck.well("B1:A1")
    assert well.ng("product") > 0 and well.ng("frag_Y") > LIMIT_NG


def test_amounts_inside_tolerance_still_count_as_correct():
    deck = deck_with(frag_X=32.5 * 1.04, frag_Y=65.0, mix=10.0)   # 4% out, tol is 5%
    assert deck.react(REACTION, StubRun()).outcome == "complete"


# -------------------------------------------------------------------- harvesting

def test_harvest_keeps_leftovers_so_a_partial_assembly_stays_visible():
    deck = deck_with(frag_X=16.0, frag_Y=65.0, mix=10.0)
    deck.react(REACTION, StubRun())
    deck.purify_into_source("assemble")
    harvested = deck.contents("B2:C1")
    assert "product" in harvested and "frag_Y" in harvested
    assert "mix" not in harvested, "a clean-up removes enzymes, not DNA"


def test_a_failed_reaction_still_hands_on_whatever_dna_is_there():
    """The technician cannot see that the assembly failed, so they purify what is in the
    well. The next gel then shows the unreacted input instead of an empty lane."""
    deck = deck_with(frag_Y=65.0, mix=10.0)       # frag_X missing, so no product forms
    deck.react(REACTION, StubRun())
    deck.purify_into_source("assemble")
    harvested = deck.contents("B2:C1")
    assert "product" not in harvested
    assert harvested.get("frag_Y", 0) > LIMIT_NG


def test_harvesting_an_empty_well_leaves_the_destination_empty():
    deck = deck_with(mix=10.0)                    # no DNA at all
    deck.react(REACTION, StubRun())
    deck.purify_into_source("assemble")
    assert deck.contents("B2:C1") == {}


# ------------------------------------------------------------------ reading gels

def test_bands_are_sorted_by_mass_and_scaled_to_relative_intensity():
    deck = Deck(SPEC)
    deck.well("B2:C1").species = {"product": 90.0, "junk": 10.0}
    bands = deck.bands(["B2:C1"])
    assert [b["bp"] for b in bands] == [3000, 500]
    assert bands[0]["rel_intensity"] == pytest.approx(0.9)


def test_species_below_the_detection_limit_do_not_show():
    deck = Deck(SPEC)
    deck.well("B2:C1").species = {"product": 90.0, "junk": LIMIT_NG / 2}
    assert [b["bp"] for b in deck.bands(["B2:C1"])] == [3000]


@pytest.mark.parametrize("contents,expected", [
    ({"product": 100.0}, "pass"),
    ({"product": 95.0, "junk": 5.0}, "pass"),          # a minor extra band is fine
    ({"product": 50.0, "junk": 50.0}, "ambiguous"),    # too much else
    ({"frag_X": 100.0}, "fail"),                       # the product is simply absent
    ({}, "fail"),
])
def test_gel_verdicts(contents, expected):
    deck = Deck(SPEC)
    deck.well("B2:C1").species = dict(contents)
    verdict, reason = deck.qc_verdict(deck.bands(["B2:C1"]), [3000])
    assert verdict == expected, reason


def test_a_near_miss_band_is_not_accepted_as_the_product():
    """2900 bp is within 5% of 3000; 2500 is not."""
    deck = Deck(SPEC)
    deck.species["near"] = {"bp": 2900}
    deck.well("B2:C1").species = {"near": 100.0}
    assert deck.qc_verdict(deck.bands(["B2:C1"]), [3000])[0] == "pass"
    deck.species["far"] = {"bp": 2500}
    deck.well("B2:C1").species = {"far": 100.0}
    assert deck.qc_verdict(deck.bands(["B2:C1"]), [3000])[0] == "fail"


def test_each_band_can_only_satisfy_one_expectation():
    """One 3000 bp band must not pass a gel that should show two distinct products."""
    deck = Deck(SPEC)
    deck.well("B2:C1").species = {"product": 100.0}
    verdict, _ = deck.qc_verdict(deck.bands(["B2:C1"]), [3000, 3000])
    assert verdict == "fail"
