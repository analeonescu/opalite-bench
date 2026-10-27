"""Tests for the QC harness: the gel, the modes, and whether the analysis is solvable.

The important one is `test_a_real_analysis_recovers_the_true_band_sizes`: it does what we
are asking the agent to do - find peaks, calibrate against the ladder, convert distance to
bp - and checks the answer comes out right. If that fails, the task is unfair rather than
hard. (The scripted oracle cannot do this, because a mock model cannot react to tool
output; it reads the truth off a shadow lab instead.)
"""
import json

import numpy as np
import pytest

from opalite_bench.lab.deck import Deck
from opalite_bench.lab.qc_harness import (GelQCHarness, InternetQCHarness, LADDER_BP, SummaryQCHarness,
                                 build_harness, migration_px)
from opalite_bench.spec.scenarios import load_spec
from opalite_bench.score.scorers import _match_bands
import base64

SPEC = load_spec()
WELLS = ["B2:B1", "B2:B2", "B2:B3"]


def deck_with_fragments() -> Deck:
    deck = Deck(SPEC)
    deck.well("B2:B1").species = {"frag_A": 100.0}
    deck.well("B2:B1").volume_ul = 30.0
    deck.well("B2:B2").species = {"frag_B": 80.0}
    deck.well("B2:B2").volume_ul = 30.0
    deck.well("B2:B3").species = {"vector_1": 130.0}
    deck.well("B2:B3").volume_ul = 30.0
    return deck


def find_peaks(profile: list[int], px_per_point: int, threshold: int) -> list[int]:
    """Peak positions in px from the wells - the first thing a reader has to do."""
    return [i * px_per_point for i in range(1, len(profile) - 1)
            if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1]
            and profile[i] > threshold]


# ------------------------------------------------------------------- is it solvable?

def test_a_real_analysis_recovers_the_true_band_sizes():
    """Measure the ladder, fit log10(bp) against distance, convert the sample lanes."""
    deck, harness = deck_with_fragments(), GelQCHarness()

    ladder = harness.measure("fragment_qc", 0, WELLS, deck)
    ladder_px = find_peaks(ladder["profile"], ladder["px_per_point"], threshold=45)
    assert len(ladder_px) == len(LADDER_BP), "every ladder band should be visible"

    fit = np.polyfit(ladder_px, np.log10(LADDER_BP), 1)     # log10(bp) = a*px + b

    measured = []
    for lane in (1, 2, 3):
        trace = harness.measure("fragment_qc", lane, WELLS, deck)
        for px in find_peaks(trace["profile"], trace["px_per_point"], threshold=55):
            measured.append(float(10 ** np.polyval(fit, px)))

    真 = [float(b["bp"]) for b in harness.true_bands("fragment_qc", WELLS, deck)]
    assert _match_bands(measured, 真, tol=0.05) == len(真), \
        f"calibration should recover {sorted(真)}, got {sorted(round(m) for m in measured)}"


def test_migration_is_logarithmic_and_ordered():
    """Bigger fragments stay closer to the well, and spacing compresses at the top."""
    assert migration_px(5000) < migration_px(1000) < migration_px(300)
    big_gap = migration_px(1000) - migration_px(2000)
    small_gap = migration_px(5000) - migration_px(10000)
    assert big_gap > small_gap, "a gel resolves small fragments better than large ones"


def test_bands_land_where_the_gel_says_they_should():
    deck, harness = deck_with_fragments(), GelQCHarness()
    trace = harness.measure("fragment_qc", 1, WELLS, deck)
    peaks = find_peaks(trace["profile"], trace["px_per_point"], threshold=55)
    assert peaks and abs(peaks[0] - migration_px(1500)) <= 4


# --------------------------------------------------------------------------- modes

def test_summary_mode_hands_over_the_answer():
    payload = SummaryQCHarness().payload("fragment_qc", WELLS, deck_with_fragments())
    assert {b["bp"] for b in payload["bands"]} == {1500, 1200, 2000}


def test_gel_mode_never_hands_over_band_sizes():
    payload = GelQCHarness().payload("fragment_qc", WELLS, deck_with_fragments())
    assert "bands" not in payload
    assert payload["image"]["ladder_bp"] == LADDER_BP
    assert [lane["sample"] for lane in payload["image"]["lanes"]] == ["ladder"] + WELLS
    # 1200 bp (frag_B) is not one of the ladder sizes, so it can only appear if the
    # sample's own band sizes leaked into what the agent is shown.
    assert 1200 not in LADDER_BP
    assert "1200" not in json.dumps(payload), "the answer must not leak into the payload"


def test_gel_mode_is_deterministic():
    deck = deck_with_fragments()
    first = GelQCHarness().measure("fragment_qc", 1, WELLS, deck)["profile"]
    second = GelQCHarness().measure("fragment_qc", 1, WELLS, deck)["profile"]
    assert first == second


def test_an_empty_lane_has_no_peaks():
    deck = Deck(SPEC)       # nothing anywhere
    trace = GelQCHarness().measure("fragment_qc", 1, WELLS, deck)
    assert not find_peaks(trace["profile"], trace["px_per_point"], threshold=55)


def test_asking_for_a_lane_that_is_not_there_is_an_error():
    with pytest.raises(IndexError):
        GelQCHarness().measure("fragment_qc", 9, WELLS, deck_with_fragments())


def test_the_png_is_a_real_png():
    gel = GelQCHarness().gel("fragment_qc", WELLS, deck_with_fragments())
    png = gel.to_png()
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and png.endswith(b"IEND\xae\x42\x60\x82")


def test_internet_mode_says_why_it_is_unavailable():
    with pytest.raises(NotImplementedError, match="sandbox"):
        InternetQCHarness()
    with pytest.raises(NotImplementedError):
        build_harness("internet")


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown qc_tools mode"):
        build_harness("microscope")


# ----------------------------------------------------------------- measurement scoring

@pytest.mark.parametrize("reported,truth,expected", [
    ([1500, 1200, 2000], [1500, 1200, 2000], 3),
    ([1503, 1201, 1978], [1500, 1200, 2000], 3),      # a real calibration is never exact
    ([1500], [1500, 1200], 1),                         # missed one
    ([1500, 1500], [1500], 1),                         # one band cannot match twice
    ([800], [1500], 0),
])
def test_band_matching_is_one_to_one_and_tolerant(reported, truth, expected):
    assert _match_bands(reported, truth, tol=0.05) == expected


# ------------------------------------------------------------------------ modes

def test_the_three_gel_modes_differ_only_in_how_you_may_look():
    manual, vlm, both = (build_harness(m) for m in ("image_manual", "image_vlm", "image_both"))
    assert (manual.measures, manual.shows_image) == (True, False)
    assert (vlm.measures, vlm.shows_image) == (False, True)
    assert (both.measures, both.shows_image) == (True, True)
    assert all(h.requires_analysis for h in (manual, vlm, both))


def test_imagej_is_still_accepted_as_an_old_name():
    assert build_harness("imagej").mode == "image_manual"


def test_a_vision_mode_produces_a_png_data_uri():
    uri = build_harness("image_vlm").image_data_uri("fragment_qc", WELLS, deck_with_fragments())
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG")


def test_the_instructions_match_the_tools_on_offer():
    manual = build_harness("image_manual").payload("fragment_qc", WELLS, deck_with_fragments())
    vlm = build_harness("image_vlm").payload("fragment_qc", WELLS, deck_with_fragments())
    assert "measure_gel_lane" in manual["note"] and "image" not in manual["note"]
    assert "image" in vlm["note"] and "measure_gel_lane" not in vlm["note"]
    assert "submit_gel_measurement" in manual["note"] and "submit_gel_measurement" in vlm["note"]
