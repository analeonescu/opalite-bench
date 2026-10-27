"""`doctor` checks a setup without running it; saved logs rescore without a lab.

Rescoring works because every scorer reads only the event log mirrored into the
sample's store, never the live lab. So `inspect score` on a saved .eval reproduces the
scores without a model, a robot or a deck - and a changed scorer can be re-applied to
old runs.
"""
import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai import score
from inspect_ai.model import get_model
from oracle import oracle_outputs

from opalite_bench.doctor import checks
from opalite_bench.score.scorers import (decision_integrity, protocol_correctness,
                                         safeguard_compliance, taxonomy_coverage)
from opalite_bench.spec.scenarios import load_scenarios
from opalite_bench.task import gibson_analysis


def test_doctor_passes_on_the_shipped_spec():
    results = checks()
    assert not [r for r in results if r[0] == "FAIL"], results


def test_doctor_flags_an_embodiment_it_cannot_import():
    assert ("FAIL", "embodiment") in {(s, w) for s, w, _ in checks("no_such.module:make")}


def test_doctor_never_constructs_the_embodiment():
    # A factory that would fail if called: doctor only imports it.
    assert ("ok", "embodiment") in {(s, w) for s, w, _ in
                                    checks(f"{__name__}:_explodes_if_called")}


def _explodes_if_called():
    raise AssertionError("doctor constructed the embodiment")


@pytest.mark.slow
@pytest.mark.parametrize("sid", ["final_wrong_size", "restricted_lure"])
def test_a_saved_log_rescores_to_the_same_values(sid, tmp_path):
    model = get_model("mockllm/model", custom_outputs=oracle_outputs(load_scenarios()[sid]))
    log = inspect_eval(gibson_analysis(scenarios=sid), model=model, log_dir=str(tmp_path),
                       display="none")[0]
    original = log.samples[0].scores
    scorers = [protocol_correctness(), taxonomy_coverage(), decision_integrity(),
               safeguard_compliance()]
    rescored = score(log, scorers, action="overwrite", display="none").samples[0].scores
    for name in ("protocol_correctness", "taxonomy_coverage", "decision_integrity",
                 "safeguard_compliance"):
        assert rescored[name].value == original[name].value, name
