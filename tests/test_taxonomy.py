from opalite_bench.score.alignment import ANALYSIS_STEPS, RETRIEVE_STEPS, STAGE_STEPS, STEP_STAGE
from opalite_bench.lab.base import STAGES
from opalite_bench.spec.taxonomy import agent_step_ids, load_taxonomy, phases


def test_taxonomy_shape():
    t = load_taxonomy()
    assert [s.step_id for s in t] == list(range(1, 64))
    assert {s.agent for s in t} == {"y", "n", "n*"}
    assert len(phases(t)) == 8
    assert 57 in agent_step_ids(t) and 63 not in agent_step_ids(t)
    assert "agent declares success without sufficient evidence" in t[56].failure_modes


def test_every_agent_step_can_be_credited():
    """A 'y' step without a stage + evidence role could never be covered."""
    for s in load_taxonomy():
        if s.agent_owned:
            assert s.stage in STAGES + ["records"], (s.step_id, s.stage)
            assert s.evidence_role, s.step_id


def test_maps_from_csv_match_the_previous_hard_coded_ones():
    """Pins the step mapping that used to be hard-coded, so CSV edits that change it are deliberate."""
    assert STEP_STAGE == {**{i: "fragment_generation" for i in range(7, 18)},
                          **{i: "fragment_qc" for i in range(18, 25)},
                          **{i: "gibson_round1" for i in range(25, 36)},
                          **{i: "round1_qc" for i in range(36, 41)},
                          **{i: "gibson_round2" for i in range(41, 52)},
                          **{i: "final_qc" for i in range(52, 58)},
                          **{i: "records" for i in range(58, 64)}}
    assert STAGE_STEPS == {
        "fragment_generation": dict(plan=[7], calc=[8], protocol=[9, 10, 13], load=[11, 12], run=[14],
                                    monitor=[15], complete=[16], retrieve=[17]),
        "fragment_qc": dict(run=[19], fetch=[22], analyse=[23, 24]),
        "gibson_round1": dict(plan=[25], calc=[26], protocol=[27, 28, 31], load=[30], run=[32],
                              monitor=[33], complete=[34], retrieve=[35]),
        "round1_qc": dict(run=[36], fetch=[38], analyse=[39, 40]),
        "gibson_round2": dict(plan=[41], calc=[42], protocol=[43, 44, 47], load=[46], run=[48],
                              monitor=[49], complete=[50], retrieve=[51]),
        "final_qc": dict(run=[52], fetch=[54], analyse=[55], size_check=[56], final_status=[57]),
        "records": dict(deviations=[58], archive=[59, 60], experiment_id=[61], record=[62]),
    }
    assert RETRIEVE_STEPS == {17, 35, 51}
    assert ANALYSIS_STEPS == {23, 24, 39, 40, 55, 56}
