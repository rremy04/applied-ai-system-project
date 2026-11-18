"""
Regression guard: the offline evaluation must not fall below the frozen floors in
tests/eval_thresholds.json. See that file's note before changing any number.
"""
import json
from pathlib import Path

import pytest

from tests.eval_rag import run_evaluation

THRESHOLDS = json.loads((Path(__file__).parent / "eval_thresholds.json").read_text())


@pytest.fixture(scope="module")
def offline_report():
    return run_evaluation(online=False)


def test_hybrid_top1_meets_floor(offline_report):
    assert offline_report["summary"]["w=0.50"]["top1"] >= THRESHOLDS["hybrid_top1_min"]


def test_hybrid_top3_meets_floor(offline_report):
    assert offline_report["summary"]["w=0.50"]["top3"] >= THRESHOLDS["hybrid_top3_min"]


def test_gibberish_is_refused(offline_report):
    assert offline_report["summary"]["w=0.50"]["negative_accuracy"] >= THRESHOLDS["hybrid_negative_accuracy_min"]
    assert offline_report["pipeline"]["negative_accuracy"] >= THRESHOLDS["pipeline_negative_accuracy_min"]


def test_hybrid_clearly_beats_the_original_scorer(offline_report):
    gap = offline_report["summary"]["w=0.50"]["top3"] - offline_report["summary"]["original_scorer"]["top3"]
    assert gap >= THRESHOLDS["hybrid_must_beat_original_scorer_by_at_least"]
