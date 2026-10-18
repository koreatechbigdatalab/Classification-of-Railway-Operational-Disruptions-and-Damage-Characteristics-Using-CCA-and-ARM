from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RULE_RESULTS = ROOT / "results" / "association-rules"


def require_module(name: str):
    """Import a required analysis module and report a test failure if it is absent."""

    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"재현 스크립트가 아직 없습니다: {name}")


def test_table5_selection_reproduces_supplied_rules() -> None:
    """A changed ranking rule must not silently alter the 15 reported rules."""

    module = require_module("analysis.association_rules")
    candidates = pd.read_csv(
        RULE_RESULTS / "table5_association_rules_candidate.csv",
        encoding="utf-8-sig",
    )
    expected = pd.read_csv(
        RULE_RESULTS / "table5_association_rules.csv",
        encoding="utf-8-sig",
    )

    actual = module.select_table5_rules(candidates)

    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_cluster_profiles_reproduce_reported_effect_sizes() -> None:
    """Wrong joins or contingency calculations must change the known Cramer's V values."""

    module = require_module("analysis.cluster_profiles")
    frame = module.load_analysis_frame(
        DATA / "model_b_input.csv",
        DATA / "cluster_result.csv",
    )
    summary, _ = module.calculate_cluster_profiles(frame)
    actual = {
        row.variable: round(float(row.cramers_v), 3)
        for row in summary.itertuples()
    }

    assert actual == {
        "철도구분": 0.511,
        "열차종류": 0.473,
        "지연운행 현황": 0.763,
        "지연운행 원인": 0.683,
        "근본원인별 원인": 0.798,
        "장소유형A": 0.214,
    }


def test_impact_summary_reproduces_reported_cluster_medians() -> None:
    """Wrong filtering of zeros or missing values must change the published summaries."""

    module = require_module("analysis.impact_summary")
    summary = module.summarize_impacts(
        DATA / "case_lookup.csv",
        DATA / "cluster_result.csv",
    )
    values = summary.pivot(index="metric", columns="cluster", values="estimate")

    assert values.loc["damage_rate_pct"].round(1).tolist() == [
        86.6,
        24.2,
        51.5,
        48.9,
        18.8,
        51.4,
    ]
    assert values.loc["positive_damage_median_million_krw"].round(2).tolist() == [
        1.49,
        1.69,
        1.85,
        4.60,
        1.10,
        4.03,
    ]
    assert values.loc["delayed_train_count_median"].tolist() == [1, 2, 2, 6, 1, 5]
    assert values.loc["max_delay_minutes_median"].tolist() == [38, 30.5, 33, 30, 0, 41]
