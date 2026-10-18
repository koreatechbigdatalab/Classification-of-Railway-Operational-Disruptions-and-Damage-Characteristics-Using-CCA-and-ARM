from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


WORK_DIR = Path(__file__).resolve().parent / "plot_data_rebuild"
LOOKUP_PATH = WORK_DIR / "KAIA_DA_k=6_model_b_case_lookup.csv"
CLUSTER_PATH = WORK_DIR / "KAIA_DA_k=6_model_b_cluster_result.csv"
OUTPUT_PATH = WORK_DIR / "visualization_dataset.json"


def boxplot_whiskers(values: pd.Series) -> tuple[float, float, float, float, float]:
    clean = values.dropna().astype(float)
    q1 = clean.quantile(0.25)
    median = clean.quantile(0.50)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    whisker_low = clean.loc[clean >= lower_bound].min()
    whisker_high = clean.loc[clean <= upper_bound].max()
    return float(q1), float(median), float(q3), float(whisker_low), float(whisker_high)


def assert_close(actual: float, expected: float, tolerance: float = 1e-8) -> None:
    if not np.isclose(actual, expected, atol=tolerance, rtol=0):
        raise AssertionError(f"Expected {expected}, got {actual}")


def main() -> None:
    lookup = pd.read_csv(LOOKUP_PATH, encoding="utf-8-sig")
    clusters = pd.read_csv(CLUSTER_PATH, encoding="utf-8-sig")
    data = lookup.merge(clusters, on="case_id", how="inner", validate="one_to_one")
    data = data.sort_values("case_id").reset_index(drop=True)

    expected_cluster_sizes = {1: 337, 2: 132, 3: 97, 4: 90, 5: 48, 6: 35}
    if data["cluster"].value_counts().sort_index().to_dict() != expected_cluster_sizes:
        raise AssertionError("군집별 사례 수가 노트북 결과와 일치하지 않습니다.")

    damage_column = "총 피해액(백만원)"
    train_column = "합계지연열차수"
    time_column = "합계최대지연시간(분)"

    case_wide = data[
        ["case_id", "source_index", "cluster", damage_column, train_column, time_column]
    ].copy()
    case_wide.insert(4, "피해액발생여부", (case_wide[damage_column] > 0).astype(int))

    damage_long = data[["case_id", "cluster", damage_column]].rename(
        columns={damage_column: "value"}
    )
    damage_long.insert(2, "variable", damage_column)

    delay_long = data[["case_id", "cluster", train_column, time_column]].melt(
        id_vars=["case_id", "cluster"],
        value_vars=[train_column, time_column],
        var_name="variable",
        value_name="value",
    )

    summary_rows: list[dict[str, float | int]] = []
    for cluster in range(1, 7):
        group = data.loc[data["cluster"] == cluster]
        damage = group[damage_column].astype(float)
        positive = damage.loc[damage > 0]
        train = group[train_column].dropna().astype(float)
        delay_time = group[time_column].dropna().astype(float)

        damage_q1, damage_median, damage_q3, damage_whisker_low, damage_whisker_high = (
            boxplot_whiskers(positive)
        )
        row: dict[str, float | int] = {
            "군집": cluster,
            "전체사례수": int(len(group)),
            "피해액발생건수": int((damage > 0).sum()),
            "피해액0원건수": int((damage == 0).sum()),
            "피해액발생비율": float((damage > 0).mean()),
            "양의피해액_n": int(positive.count()),
            "양의피해액_Q1_백만원": damage_q1,
            "양의피해액_중앙값_백만원": damage_median,
            "양의피해액_Q3_백만원": damage_q3,
            "양의피해액_하단수염_백만원": damage_whisker_low,
            "양의피해액_상단수염_백만원": damage_whisker_high,
            "양의피해액_최댓값_백만원": float(positive.max()),
            "지연열차수_n": int(train.count()),
            "지연열차수_P10": float(train.quantile(0.10)),
            "지연열차수_Q1": float(train.quantile(0.25)),
            "지연열차수_중앙값": float(train.median()),
            "지연열차수_Q3": float(train.quantile(0.75)),
            "지연열차수_P90": float(train.quantile(0.90)),
            "지연열차수_최솟값": float(train.min()),
            "지연열차수_최댓값": float(train.max()),
            "최대지연시간_n": int(delay_time.count()),
            "최대지연시간_P10_분": float(delay_time.quantile(0.10)),
            "최대지연시간_Q1_분": float(delay_time.quantile(0.25)),
            "최대지연시간_중앙값_분": float(delay_time.median()),
            "최대지연시간_Q3_분": float(delay_time.quantile(0.75)),
            "최대지연시간_P90_분": float(delay_time.quantile(0.90)),
            "최대지연시간_최솟값_분": float(delay_time.min()),
            "최대지연시간_최댓값_분": float(delay_time.max()),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    expected_damage = {
        1: (292, 0.8664688427299704, 1.4895, 6440.0),
        2: (32, 0.24242424242424243, 1.6917, 61.2),
        3: (50, 0.5154639175257731, 1.85, 846.0),
        4: (44, 0.4888888888888889, 4.60395, 527.31),
        5: (9, 0.1875, 1.1, 20.0),
        6: (18, 0.5142857142857142, 4.02875, 249.5),
    }
    expected_train_medians = {1: 1.0, 2: 2.0, 3: 2.0, 4: 6.0, 5: 1.0, 6: 5.0}
    expected_time_medians = {1: 38.0, 2: 30.5, 3: 33.0, 4: 30.0, 5: 0.0, 6: 41.0}
    for row in summary.itertuples(index=False):
        cluster = int(row.군집)
        expected_n, expected_rate, expected_median, expected_max = expected_damage[cluster]
        if int(row.피해액발생건수) != expected_n:
            raise AssertionError(f"군집 {cluster} 피해액 발생건수 불일치")
        assert_close(float(row.피해액발생비율), expected_rate)
        assert_close(float(row.양의피해액_중앙값_백만원), expected_median)
        assert_close(float(row.양의피해액_최댓값_백만원), expected_max)
        assert_close(float(row.지연열차수_중앙값), expected_train_medians[cluster])
        assert_close(float(row.최대지연시간_중앙값_분), expected_time_medians[cluster])

    float_columns = summary.select_dtypes(include=["float", "float64"]).columns
    summary[float_columns] = summary[float_columns].round(6)

    metadata = [
        ["항목", "내용"],
        ["분석대상", "운행장애 739건(Model B 적극변수 결측 제거 후)"],
        ["군집해", "K=6, d=5, clusCA, ASW 기준, nstart=100, seed=1234"],
        ["피해액 발생", "총 피해액(백만원) > 0"],
        ["피해 규모", "피해액 발생 사례만 사용; 로그축은 시각화 단계에서 적용"],
        ["피해액 수염", "1.5×IQR 범위 안의 실제 최솟값·최댓값"],
        ["지연 구간", "가는 선=P10–P90, 굵은 선=Q1–Q3, 점=중앙값"],
        ["결측 처리", "지표별 n은 해당 값의 결측을 제외한 유효 사례 수"],
        ["원자료", "한국교통안전공단_국내 철도사고 및 운행장애 발생 정보_20241231.csv"],
        ["분석코드", "KAIA_DA_k=6.ipynb 및 KAIA_DA_k=6.R"],
    ]

    payload = {
        "summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "case_wide": case_wide.replace({np.nan: None}).to_dict(orient="records"),
        "damage_long": damage_long.replace({np.nan: None}).to_dict(orient="records"),
        "delay_long": delay_long.replace({np.nan: None}).to_dict(orient="records"),
        "metadata": metadata,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"summary_shape={summary.shape}")
    print(f"case_wide_shape={case_wide.shape}")
    print(f"damage_long_shape={damage_long.shape}")
    print(f"delay_long_shape={delay_long.shape}")
    print(summary.to_string(index=False))
    print(f"json_path={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
