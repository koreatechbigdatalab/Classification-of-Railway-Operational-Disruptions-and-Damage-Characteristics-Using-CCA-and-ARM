from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_CSV = Path(
    r"C:\Users\l0535\AppData\Local\Temp\한국교통안전공단_국내 철도사고 및 운행장애 발생 정보_20241231.csv"
)
WORK_DIR = Path(__file__).resolve().parent / "plot_data_rebuild"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def clean_object_values(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.select_dtypes(include="object").columns:
        out[column] = (
            out[column]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
        )
    return out


def add_date_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["발생일자_dt"] = pd.to_datetime(out["발생일자"], errors="coerce")
    out["발생연도"] = out["발생일자_dt"].dt.year.astype("Int64")
    out["발생월"] = out["발생일자_dt"].dt.month.astype("Int64")
    out["발생요일번호"] = out["발생일자_dt"].dt.dayofweek.astype("Int64")

    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    out["발생요일"] = out["발생요일번호"].map(weekday_map).astype("string")
    out["주말여부"] = np.where(out["발생요일번호"].isin([5, 6]), "주말", "평일")
    out.loc[out["발생요일번호"].isna(), "주말여부"] = pd.NA

    time_text = out["발생시간"].astype("string").str.extract(r"(\d{1,2}):(\d{2})")
    out["발생시"] = pd.to_numeric(time_text[0], errors="coerce").astype("Int64")
    out["시간대"] = pd.cut(
        out["발생시"],
        bins=[-1, 5, 9, 15, 19, 23],
        labels=[
            "심야(00~05)",
            "오전혼잡(06~09)",
            "주간(10~15)",
            "오후혼잡(16~19)",
            "야간(20~23)",
        ],
    ).astype("string")

    season_map = {
        12: "겨울",
        1: "겨울",
        2: "겨울",
        3: "봄",
        4: "봄",
        5: "봄",
        6: "여름",
        7: "여름",
        8: "여름",
        9: "가을",
        10: "가을",
        11: "가을",
    }
    out["계절"] = out["발생월"].map(season_map).astype("string")
    return out


def to_numeric_safely(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in [
        "총피해명",
        "사망",
        "부상(중상)",
        "총 피해액(백만원)",
        "합계지연열차수",
        "합계최대지연시간(분)",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def main() -> None:
    frame = None
    used_encoding = None
    for encoding in ["utf-8-sig", "cp949", "euc-kr"]:
        try:
            frame = pd.read_csv(SOURCE_CSV, encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue
    if frame is None:
        raise RuntimeError("원자료 CSV의 인코딩을 판별하지 못했습니다.")

    frame.columns = frame.columns.str.strip()
    disorder_raw = frame.loc[frame["사고 종류"].isin(["장애(지연)", "장애(무정차)"])].copy()
    disorder = (
        disorder_raw.pipe(clean_object_values).pipe(add_date_time_features).pipe(to_numeric_safely)
    )
    mapping = {
        "열차종류": {
            "입환차량": "기타열차",
            "혼합열차": "기타열차",
            "시운전열차": "기타열차",
            "작업차량": "기타열차",
            "회송열차": "기타열차",
            "기타": "기타열차",
        },
        "장소유형A": {
            "역(대합실)": "역(기타)",
            "역(작업장)": "역(기타)",
            "열차 내": "기타장소",
            "기타": "기타장소",
            "차량기지": "기타장소",
            "운행선": "기타장소",
        },
        "지연운행 현황": {
            "지연운행 발생현황-열차분리": "지연운행 발생현황-기타(소수)",
            "지연운행 발생현황-선로장애": "지연운행 발생현황-기타(소수)",
            "지연운행 발생현황-규정위반": "지연운행 발생현황-기타(소수)",
        },
    }
    for column, column_mapping in mapping.items():
        disorder[column] = disorder[column].replace(column_mapping)

    if len(disorder) != 754:
        raise AssertionError(f"운행장애 관측치가 754건이 아닙니다: {len(disorder)}")

    cca_source = disorder.copy()
    cca_source["source_index"] = cca_source.index
    cca_source["case_id"] = [f"DIS_{index:04d}" for index in range(1, len(cca_source) + 1)]

    active_columns = [
        "철도구분",
        "열차종류",
        "지연운행 현황",
        "지연운행 원인",
        "근본원인별 원인",
        "장소유형A",
    ]
    model_b = cca_source[["case_id", *active_columns]].dropna(subset=active_columns).copy()
    for column in active_columns:
        model_b[column] = model_b[column].astype("string")

    lookup_columns = [
        "case_id",
        "source_index",
        "사고 종류",
        "발생연도",
        "등록기관",
        "노선",
        "근본원인별 그룹",
        "총 피해액(백만원)",
        "합계지연열차수",
        "합계최대지연시간(분)",
    ]
    lookup = cca_source.loc[cca_source["case_id"].isin(model_b["case_id"]), lookup_columns].copy()

    if len(model_b) != 739 or len(lookup) != 739:
        raise AssertionError(f"Model B 관측치가 739건이 아닙니다: {len(model_b)}, {len(lookup)}")
    if lookup["총 피해액(백만원)"].isna().any():
        raise AssertionError("총 피해액에 결측값이 있습니다.")

    model_path = WORK_DIR / "KAIA_DA_k=6_model_b_input.csv"
    lookup_path = WORK_DIR / "KAIA_DA_k=6_model_b_case_lookup.csv"
    model_b.to_csv(model_path, index=False, encoding="utf-8-sig")
    lookup.to_csv(lookup_path, index=False, encoding="utf-8-sig")

    print(f"encoding={used_encoding}")
    print(f"raw_shape={frame.shape}")
    print(f"disorder_shape={disorder.shape}")
    print(f"model_b_shape={model_b.shape}")
    print(f"lookup_shape={lookup.shape}")
    print(f"model_path={model_path}")
    print(f"lookup_path={lookup_path}")


if __name__ == "__main__":
    main()
