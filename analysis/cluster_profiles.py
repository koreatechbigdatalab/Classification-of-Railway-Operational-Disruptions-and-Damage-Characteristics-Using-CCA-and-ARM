from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "model_b_input.csv"
DEFAULT_CLUSTERS = ROOT / "data" / "processed" / "cluster_result.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "cluster-profiles"
VARIABLES = (
    "철도구분",
    "열차종류",
    "지연운행 현황",
    "지연운행 원인",
    "근본원인별 원인",
    "장소유형A",
)
EXPECTED_COUNTS = {1: 337, 2: 132, 3: 97, 4: 90, 5: 48, 6: 35}


def load_analysis_frame(input_path: Path, cluster_path: Path) -> pd.DataFrame:
    model_input = pd.read_csv(input_path, encoding="utf-8-sig")
    clusters = pd.read_csv(cluster_path, encoding="utf-8-sig")
    required = {"case_id", *VARIABLES}
    missing = required.difference(model_input.columns)
    if missing:
        raise ValueError(f"입력 열이 누락되었습니다: {sorted(missing)}")
    if not {"case_id", "cluster"}.issubset(clusters.columns):
        raise ValueError("군집 결과에는 case_id와 cluster가 필요합니다.")
    if not model_input["case_id"].is_unique or not clusters["case_id"].is_unique:
        raise ValueError("case_id는 각 파일에서 유일해야 합니다.")
    if model_input[list(VARIABLES)].isna().any().any():
        raise ValueError("군집 특성 변수에 결측값이 있습니다.")
    clusters["cluster"] = pd.to_numeric(clusters["cluster"], errors="raise").astype(int)
    counts = clusters["cluster"].value_counts().sort_index().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"군집별 사례 수가 다릅니다: {counts}")
    frame = model_input.merge(clusters, on="case_id", how="inner", validate="one_to_one")
    if len(frame) != 739:
        raise ValueError(f"병합 결과는 739건이어야 합니다: {len(frame)}")
    return frame


def calculate_cluster_profiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    cell_frames: list[pd.DataFrame] = []

    for variable in VARIABLES:
        table = pd.crosstab(frame["cluster"], frame[variable]).sort_index()
        chi2, p_value, degrees_freedom, expected = chi2_contingency(
            table.to_numpy(), correction=False
        )
        n = int(table.to_numpy().sum())
        row_totals = table.sum(axis=1).to_numpy(dtype=float)
        column_totals = table.sum(axis=0).to_numpy(dtype=float)
        row_proportions = row_totals / n
        column_proportions = column_totals / n
        denominator = np.sqrt(
            expected
            * (1 - row_proportions[:, None])
            * (1 - column_proportions[None, :])
        )
        residuals = np.divide(
            table.to_numpy(dtype=float) - expected,
            denominator,
            out=np.full_like(expected, np.nan, dtype=float),
            where=denominator > 0,
        )
        cell_frames.append(
            pd.DataFrame(
                {
                    "variable": variable,
                    "cluster": np.repeat(table.index.to_numpy(), len(table.columns)),
                    "category": np.tile(table.columns.to_numpy(), len(table.index)),
                    "observed": table.to_numpy().ravel(),
                    "expected": expected.ravel(),
                    "adjusted_residual": residuals.ravel(),
                    "cluster_share_pct": (
                        table.to_numpy(dtype=float) / row_totals[:, None] * 100
                    ).ravel(),
                }
            )
        )
        summary_rows.append(
            {
                "variable": variable,
                "n": n,
                "chi_square": chi2,
                "degrees_freedom": int(degrees_freedom),
                "p_value": p_value,
                "cramers_v": math.sqrt(
                    chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1))
                ),
                "minimum_expected": float(expected.min()),
                "expected_below_5_cells": int((expected < 5).sum()),
            }
        )

    return pd.DataFrame(summary_rows), pd.concat(cell_frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="군집별 범주 특성을 재현합니다.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    frame = load_analysis_frame(args.input, args.clusters)
    summary, cells = calculate_cluster_profiles(frame)
    selected = (
        cells.loc[cells["adjusted_residual"] >= 1.96]
        .sort_values(
            ["cluster", "variable", "adjusted_residual", "observed"],
            ascending=[True, True, False, False],
        )
        .groupby(["cluster", "variable"], sort=False, group_keys=False)
        .head(2)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "chi_square_summary.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(args.output_dir / "adjusted_residuals.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(args.output_dir / "selected_characteristics.csv", index=False, encoding="utf-8-sig")
    print(summary[["variable", "cramers_v", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()

