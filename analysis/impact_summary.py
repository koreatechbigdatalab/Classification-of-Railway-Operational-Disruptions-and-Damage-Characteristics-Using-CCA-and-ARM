from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOOKUP = ROOT / "data" / "processed" / "case_lookup.csv"
DEFAULT_CLUSTERS = ROOT / "data" / "processed" / "cluster_result.csv"
DEFAULT_OUTPUT = ROOT / "results" / "impact_summary.csv"
EXPECTED_COUNTS = {1: 337, 2: 132, 3: 97, 4: 90, 5: 48, 6: 35}


def summarize_impacts(lookup_path: Path, cluster_path: Path) -> pd.DataFrame:
    lookup = pd.read_csv(lookup_path, encoding="utf-8-sig")
    clusters = pd.read_csv(cluster_path, encoding="utf-8-sig")
    needed = {
        "case_id",
        "총 피해액(백만원)",
        "합계지연열차수",
        "합계최대지연시간(분)",
    }
    missing = needed.difference(lookup.columns)
    if missing:
        raise ValueError(f"피해·지연 열이 누락되었습니다: {sorted(missing)}")
    if not lookup["case_id"].is_unique or not clusters["case_id"].is_unique:
        raise ValueError("case_id는 각 파일에서 유일해야 합니다.")
    frame = lookup.merge(clusters, on="case_id", how="inner", validate="one_to_one")
    if len(frame) != 739:
        raise ValueError(f"병합 결과는 739건이어야 합니다: {len(frame)}")
    frame["cluster"] = pd.to_numeric(frame["cluster"], errors="raise").astype(int)
    counts = frame["cluster"].value_counts().sort_index().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"군집별 사례 수가 다릅니다: {counts}")

    rows: list[dict[str, object]] = []
    for cluster in range(1, 7):
        group = frame.loc[frame["cluster"] == cluster]
        damage = pd.to_numeric(group["총 피해액(백만원)"], errors="coerce")
        positive_damage = damage.loc[damage > 0]
        delayed_trains = pd.to_numeric(group["합계지연열차수"], errors="coerce").dropna()
        delay_minutes = pd.to_numeric(
            group["합계최대지연시간(분)"], errors="coerce"
        ).dropna()
        positive_count = int((damage > 0).sum())
        metrics = (
            ("damage_rate_pct", len(damage), positive_count, 100 * positive_count / len(group)),
            (
                "positive_damage_median_million_krw",
                len(positive_damage),
                None,
                float(positive_damage.median()),
            ),
            (
                "delayed_train_count_median",
                len(delayed_trains),
                None,
                float(delayed_trains.median()),
            ),
            (
                "max_delay_minutes_median",
                len(delay_minutes),
                None,
                float(delay_minutes.median()),
            ),
        )
        for metric, n, numerator, estimate in metrics:
            rows.append(
                {
                    "cluster": cluster,
                    "metric": metric,
                    "n": n,
                    "numerator": numerator,
                    "estimate": estimate,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="군집별 피해·지연 통계를 재현합니다.")
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = summarize_impacts(args.lookup, args.clusters)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

