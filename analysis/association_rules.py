from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "model_b_input.csv"
DEFAULT_CLUSTERS = ROOT / "data" / "processed" / "cluster_result.csv"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "association-rules"

VARIABLES = (
    "철도구분",
    "열차종류",
    "지연운행 현황",
    "지연운행 원인",
    "근본원인별 원인",
    "장소유형A",
)
CLUSTER_NAMES = {
    1: "여객열차·차량설비 고장형",
    2: "도시전동차·복합설비 장애형",
    3: "외부환경 요인형",
    4: "신호제어설비 장애형",
    5: "무정차통과·운전취급 부주의형",
    6: "전철전력설비·급전장애형",
}
EXPECTED_CLUSTER_COUNTS = {1: 337, 2: 132, 3: 97, 4: 90, 5: 48, 6: 35}


def load_analysis_frame(input_path: Path, cluster_path: Path) -> pd.DataFrame:
    model_input = pd.read_csv(input_path, encoding="utf-8-sig")
    clusters = pd.read_csv(cluster_path, encoding="utf-8-sig")

    for name, frame in {"model_input": model_input, "clusters": clusters}.items():
        if "case_id" not in frame:
            raise ValueError(f"{name}에 case_id가 없습니다.")
        if frame["case_id"].isna().any() or not frame["case_id"].is_unique:
            raise ValueError(f"{name}의 case_id는 결측이나 중복 없이 유일해야 합니다.")

    missing = set(VARIABLES).difference(model_input.columns)
    if missing:
        raise ValueError(f"분석 변수가 누락되었습니다: {sorted(missing)}")
    if model_input[list(VARIABLES)].isna().any().any():
        raise ValueError("연관규칙 분석 변수에 결측값이 있습니다.")

    clusters["cluster"] = pd.to_numeric(clusters["cluster"], errors="raise").astype(int)
    counts = clusters["cluster"].value_counts().sort_index().to_dict()
    if counts != EXPECTED_CLUSTER_COUNTS:
        raise ValueError(f"군집별 사례 수가 다릅니다: {counts}")

    merged = model_input.merge(
        clusters[["case_id", "cluster"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != 739:
        raise ValueError(f"병합 결과는 739건이어야 합니다: {len(merged)}")
    return merged


def make_transaction_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    categorical = frame[list(VARIABLES)].astype("string")
    for column in VARIABLES:
        categorical[column] = categorical[column].str.strip()
    matrix = pd.get_dummies(
        categorical,
        columns=list(VARIABLES),
        prefix=list(VARIABLES),
        prefix_sep="=",
        dtype=bool,
    )
    if not matrix.sum(axis=1).eq(len(VARIABLES)).all():
        raise ValueError("각 사례는 변수마다 정확히 하나의 범주를 가져야 합니다.")
    return matrix


def _itemset_text(itemset: frozenset[str]) -> str:
    return " & ".join(sorted(str(item) for item in itemset))


def _variables_in(itemset: frozenset[str]) -> set[str]:
    return {str(item).split("=", 1)[0] for item in itemset}


def mine_rules(
    frame: pd.DataFrame,
    min_support: float = 0.20,
    min_confidence: float = 0.60,
    min_lift: float = 1.20,
    max_len: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ModuleNotFoundError as error:
        raise RuntimeError("연관규칙 계산에는 mlxtend가 필요합니다.") from error

    candidates: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for cluster_id, expected_n in EXPECTED_CLUSTER_COUNTS.items():
        subset = frame.loc[frame["cluster"] == cluster_id]
        if len(subset) != expected_n:
            raise ValueError(f"군집 {cluster_id}의 사례 수가 다릅니다: {len(subset)}")

        itemsets = apriori(
            make_transaction_matrix(subset),
            min_support=min_support,
            use_colnames=True,
            max_len=max_len,
        )
        if itemsets.empty:
            rules = pd.DataFrame()
        else:
            rules = association_rules(
                itemsets,
                metric="confidence",
                min_threshold=min_confidence,
            )
            rules = rules.loc[rules["lift"] >= min_lift].copy()

        if not rules.empty:
            valid = rules.apply(
                lambda row: _variables_in(row["antecedents"]).isdisjoint(
                    _variables_in(row["consequents"])
                ),
                axis=1,
            )
            rules = rules.loc[valid].copy()
            rules["cluster"] = cluster_id
            rules["cluster_name"] = CLUSTER_NAMES[cluster_id]
            rules["cluster_n"] = len(subset)
            rules["antecedent"] = rules["antecedents"].map(_itemset_text)
            rules["consequent"] = rules["consequents"].map(_itemset_text)
            rules["antecedent_length"] = rules["antecedents"].map(len)
            rules["consequent_length"] = rules["consequents"].map(len)
            rules["support_count"] = (rules["support"] * len(subset)).round().astype(int)
            rules = rules.sort_values(
                ["lift", "confidence", "support"],
                ascending=[False, False, False],
            )
            candidates.append(rules)

        summary_rows.append(
            {
                "cluster": cluster_id,
                "cluster_name": CLUSTER_NAMES[cluster_id],
                "n": len(subset),
                "minimum_support": min_support,
                "minimum_support_count": math.ceil(min_support * len(subset)),
                "minimum_confidence": min_confidence,
                "minimum_lift": min_lift,
                "maximum_itemset_length": max_len,
                "frequent_itemset_count": len(itemsets),
                "rule_count": len(rules),
            }
        )

    columns = [
        "cluster",
        "cluster_name",
        "cluster_n",
        "antecedent",
        "consequent",
        "antecedent_length",
        "consequent_length",
        "support_count",
        "support",
        "confidence",
        "lift",
        "antecedent support",
        "consequent support",
        "leverage",
        "conviction",
        "zhangs_metric",
    ]
    if candidates:
        candidate_frame = pd.concat(candidates, ignore_index=True)
        candidate_frame = candidate_frame[[c for c in columns if c in candidate_frame]]
    else:
        candidate_frame = pd.DataFrame(columns=columns)
    return candidate_frame, pd.DataFrame(summary_rows)


def select_table5_rules(candidates: pd.DataFrame, per_cluster: int = 3) -> pd.DataFrame:
    required = {
        "cluster",
        "cluster_name",
        "antecedent",
        "consequent",
        "support_count",
        "support",
        "confidence",
        "lift",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"후보 규칙 열이 누락되었습니다: {sorted(missing)}")

    selected = (
        candidates.sort_values(
            ["cluster", "lift", "confidence", "support"],
            ascending=[True, False, False, False],
        )
        .groupby("cluster", sort=True, group_keys=False)
        .head(per_cluster)
        .copy()
    )
    selected["순위"] = selected.groupby("cluster").cumcount() + 1
    output = selected.rename(
        columns={
            "cluster": "군집",
            "cluster_name": "군집명",
            "support_count": "건수",
        }
    )[
        [
            "군집",
            "군집명",
            "순위",
            "antecedent",
            "consequent",
            "support",
            "confidence",
            "lift",
            "건수",
        ]
    ]
    output[["support", "confidence", "lift"]] = output[
        ["support", "confidence", "lift"]
    ].round(3)
    return output.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="군집별 연관규칙을 재현합니다.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    frame = load_analysis_frame(args.input, args.clusters)
    candidates, summary = mine_rules(frame)
    table5 = select_table5_rules(candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(
        args.output_dir / "table5_association_rules_candidate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        args.output_dir / "association_rule_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table5.to_csv(
        args.output_dir / "table5_association_rules.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary[["cluster", "n", "rule_count"]].to_string(index=False))


if __name__ == "__main__":
    main()

