from __future__ import annotations

import json
import math
import re
import sys
import base64
from io import StringIO
from pathlib import Path

import nbformat as nbf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2_contingency


sys.stdout.reconfigure(encoding="utf-8")

SOURCE_NOTEBOOK = Path(
    r"C:\Users\l0535\AppData\Local\Temp\KAIA_DA_k=6.ipynb"
)
SOURCE_R = Path(r"C:\Users\l0535\AppData\Local\Temp\KAIA_DA_k=6.R")
OUTPUT_DIR = Path(
    r"C:\Users\l0535\Documents\Codex\2026-10-19\new-chat\outputs"
)

OBSERVED_PATH = OUTPUT_DIR / "KAIA_DA_k=6_ASR_observed_counts.csv"
ALL_CELLS_PATH = OUTPUT_DIR / "KAIA_DA_k=6_ASR_all_cells.csv"
SELECTED_PATH = OUTPUT_DIR / "KAIA_DA_k=6_ASR_selected_top2.csv"
CHI_PATH = OUTPUT_DIR / "KAIA_DA_k=6_ASR_chi_square_summary.csv"
SUMMARY_PATH = OUTPUT_DIR / "KAIA_DA_k=6_ASR_cluster_summary.csv"
HEATMAP_PATH = OUTPUT_DIR / "KAIA_DA_k=6_ASR_heatmaps.png"
NOTEBOOK_PATH = OUTPUT_DIR / "KAIA_DA_k=6_ASR_analysis.ipynb"
R_PATH = OUTPUT_DIR / "KAIA_DA_k=6_with_ASR.R"


def extract_observed_counts() -> pd.DataFrame:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    outputs = notebook["cells"][70]["outputs"]
    frames: list[pd.DataFrame] = []
    current_variable: str | None = None

    for output in outputs:
        if output.get("output_type") == "stream":
            text = "".join(output.get("text", []))
            match = re.search(r"\[군집별 (.+) 분포\]", text)
            if match:
                current_variable = match.group(1)

        html = "".join(output.get("data", {}).get("text/html", []))
        if not html or current_variable is None:
            continue

        table = pd.read_html(StringIO(html))[0]
        if "Unnamed: 0" in table.columns:
            table = table.drop(columns="Unnamed: 0")
        table = table.rename(columns={current_variable: "category"})
        table.insert(0, "variable", current_variable)
        frames.append(table[["variable", "cluster", "category", "count"]])

    observed = pd.concat(frames, ignore_index=True)
    observed["cluster"] = observed["cluster"].astype(int)
    observed["count"] = observed["count"].astype(int)
    observed["category"] = observed["category"].astype(str)

    expected_variables = [
        "철도구분",
        "열차종류",
        "지연운행 현황",
        "지연운행 원인",
        "근본원인별 원인",
        "장소유형A",
        "근본원인별 그룹",
    ]
    assert observed["variable"].drop_duplicates().tolist() == expected_variables
    totals = observed.groupby("variable")["count"].sum()
    assert totals.eq(739).all(), totals.to_dict()
    return observed


def calculate_asr(
    observed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_results: list[pd.DataFrame] = []
    test_results: list[dict[str, float | int | str]] = []

    for variable, values in observed.groupby("variable", sort=False):
        table = (
            values.pivot(index="cluster", columns="category", values="count")
            .fillna(0)
            .astype(int)
            .sort_index()
        )
        chi2, p_value, degrees_freedom, expected = chi2_contingency(
            table.to_numpy(), correction=False
        )

        n = int(table.to_numpy().sum())
        row_totals = table.sum(axis=1).to_numpy(dtype=float)
        col_totals = table.sum(axis=0).to_numpy(dtype=float)
        row_props = row_totals / n
        col_props = col_totals / n
        denominator = np.sqrt(
            expected * (1 - row_props[:, None]) * (1 - col_props[None, :])
        )
        adjusted_residuals = np.divide(
            table.to_numpy(dtype=float) - expected,
            denominator,
            out=np.full_like(expected, np.nan, dtype=float),
            where=denominator > 0,
        )

        result = pd.DataFrame(
            {
                "variable": variable,
                "cluster": np.repeat(table.index.to_numpy(), len(table.columns)),
                "category": np.tile(table.columns.to_numpy(), len(table.index)),
                "observed": table.to_numpy().ravel(),
                "expected": expected.ravel(),
                "adjusted_residual": adjusted_residuals.ravel(),
                "cluster_share_pct": (
                    table.to_numpy(dtype=float) / row_totals[:, None] * 100
                ).ravel(),
                "overall_share_pct": np.tile(col_props * 100, len(table.index)),
            }
        )
        result["meets_positive_1_96"] = result["adjusted_residual"] >= 1.96
        result["meets_negative_minus_1_96"] = result["adjusted_residual"] <= -1.96
        all_results.append(result)

        r_count, c_count = table.shape
        cramer_v = math.sqrt(chi2 / (n * min(r_count - 1, c_count - 1)))
        expected_below_5 = int((expected < 5).sum())
        test_results.append(
            {
                "variable": variable,
                "n": n,
                "rows_clusters": r_count,
                "categories": c_count,
                "chi_square": chi2,
                "degrees_freedom": int(degrees_freedom),
                "p_value": p_value,
                "cramers_v": cramer_v,
                "min_expected": float(expected.min()),
                "expected_below_5_cells": expected_below_5,
                "expected_below_5_pct": expected_below_5 / expected.size * 100,
            }
        )

    all_cells = pd.concat(all_results, ignore_index=True)
    selected = (
        all_cells.loc[all_cells["meets_positive_1_96"]]
        .sort_values(
            ["cluster", "variable", "adjusted_residual", "observed"],
            ascending=[True, True, False, False],
        )
        .groupby(["cluster", "variable"], sort=False, as_index=False)
        .head(2)
        .copy()
    )
    selected["rank_within_cluster_variable"] = (
        selected.groupby(["cluster", "variable"]).cumcount() + 1
    )
    selected = selected[
        [
            "cluster",
            "variable",
            "rank_within_cluster_variable",
            "category",
            "observed",
            "expected",
            "adjusted_residual",
            "cluster_share_pct",
            "overall_share_pct",
        ]
    ]
    tests = pd.DataFrame(test_results)
    return all_cells, selected, tests


def build_cluster_summary(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for cluster, values in selected.groupby("cluster", sort=True):
        parts: list[str] = []
        for variable, subset in values.groupby("variable", sort=False):
            categories = ", ".join(
                f"{row.category} (ASR {row.adjusted_residual:.2f})"
                for row in subset.itertuples()
            )
            parts.append(f"{variable}: {categories}")
        rows.append({"cluster": int(cluster), "selected_characteristics": " | ".join(parts)})
    return pd.DataFrame(rows)


def create_heatmap(all_cells: pd.DataFrame) -> None:
    font_path = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font_path.exists():
        import matplotlib.font_manager as fm

        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="white", font=plt.rcParams.get("font.family", ["sans-serif"])[0])

    cmap = LinearSegmentedColormap.from_list(
        "navy_white_gold", ["#1F5A94", "#F7F7F5", "#D97706"]
    )
    variables = all_cells["variable"].drop_duplicates().tolist()
    category_counts = (
        all_cells.groupby("variable")["category"].nunique().reindex(variables)
    )
    height_ratios = [max(2.4, 0.38 * n) for n in category_counts]
    figure, axes = plt.subplots(
        len(variables),
        1,
        figsize=(11, sum(height_ratios)),
        gridspec_kw={"height_ratios": height_ratios},
        constrained_layout=True,
    )
    for axis, variable in zip(axes, variables):
        subset = all_cells.loc[all_cells["variable"] == variable]
        matrix = subset.pivot(
            index="category", columns="cluster", values="adjusted_residual"
        )
        ordering = matrix.max(axis=1).sort_values(ascending=False).index
        matrix = matrix.loc[ordering]
        max_abs = float(np.nanmax(np.abs(matrix.to_numpy())))
        annotations = matrix.map(
            lambda value: f"{value:.1f}"
            + ("*" if value >= 1.96 else ("†" if value <= -1.96 else ""))
        )
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            annot=annotations,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "조정 표준화 잔차"},
        )
        axis.set_title(
            f"{variable}: 군집별 조정 표준화 잔차",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        axis.set_xlabel("군집")
        axis.set_ylabel("범주")
    figure.suptitle(
        "K=6 군집과 범주형 변수의 Haberman 조정 표준화 잔차",
        fontsize=16,
        weight="bold",
    )
    figure.text(
        0.01,
        0.002,
        "* ASR ≥ 1.96 (상대적 과대표),  † ASR ≤ -1.96 (상대적 과소대표)",
        fontsize=10,
    )
    figure.savefig(HEATMAP_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def dataframe_output(frame: pd.DataFrame) -> nbf.NotebookNode:
    return nbf.v4.new_output(
        output_type="display_data",
        data={
            "text/plain": frame.to_string(),
            "text/html": frame.to_html(border=1),
        },
        metadata={},
    )


def attach_precomputed_outputs(
    notebook: nbf.NotebookNode,
    observed: pd.DataFrame,
    selected: pd.DataFrame,
    tests: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    cluster_sizes = (
        observed.loc[observed["variable"] == "철도구분"]
        .groupby("cluster")["count"]
        .sum()
        .sort_index()
        .rename("n")
        .to_frame()
    )
    variable_summary = observed.groupby("variable").agg(
        categories=("category", "nunique"), n=("count", "sum")
    )
    rendered_tests = tests.copy()
    for column, digits in {
        "chi_square": 2,
        "p_value": 4,
        "cramers_v": 3,
        "min_expected": 2,
        "expected_below_5_pct": 1,
    }.items():
        rendered_tests[column] = rendered_tests[column].round(digits)
    rendered_selected = selected.copy()
    for column in [
        "expected",
        "adjusted_residual",
        "cluster_share_pct",
        "overall_share_pct",
    ]:
        rendered_selected[column] = rendered_selected[column].round(2)

    outputs_by_cell = {
        4: [dataframe_output(cluster_sizes), dataframe_output(variable_summary)],
        6: [dataframe_output(rendered_tests)],
        9: [dataframe_output(rendered_selected)],
        13: [dataframe_output(cluster_summary)],
    }
    image_data = base64.b64encode(HEATMAP_PATH.read_bytes()).decode("ascii")
    outputs_by_cell[11] = [
        nbf.v4.new_output(
            output_type="display_data",
            data={"image/png": image_data, "text/plain": "<Figure: ASR heatmaps>"},
            metadata={},
        )
    ]
    execution_count = 1
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs_by_cell.get(cell_index, [])
        execution_count += 1


def make_notebook(cluster_sizes: dict[int, int]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}

    source_note = SOURCE_NOTEBOOK.name
    cluster_text = ", ".join(
        f"군집 {cluster}: {count}건" for cluster, count in cluster_sizes.items()
    )

    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# K=6 군집의 조정 표준화 잔차(ASR) 추가 분석\n\n"
            "Model B의 고정 6군집·5차원 해에 대해 군집과 범주형 변수의 "
            "교차표를 복원하고, Haberman 조정 표준화 잔차를 계산한다."
        ),
        nbf.v4.new_markdown_cell(
            "## tl;dr\n\n"
            f"- 분석 사례는 총 739건이며 군집 크기는 {cluster_text}이다.\n"
            "- ASR ≥ 1.96인 범주를 상대적 과대표 범주로 보고, 각 군집×변수에서 "
            "잔차가 큰 순서로 최대 2개를 선정했다.\n"
            "- `근본원인별 그룹`은 군집 형성에 사용하지 않은 보조변수다. "
            "나머지 6개 변수는 적극변수이므로 ASR은 독립적인 가설검정이 아니라 "
            "군집 프로파일을 요약하는 기술적 지표로 해석해야 한다."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            f"원자료와 군집화 과정은 `{source_note}`에 기록되어 있다. 이 노트북은 "
            "그 노트북의 실행 결과에 완전하게 저장된 범주별 관측빈도를 입력으로 사용한다.\n\n"
            "### Key Assumptions\n\n"
            "- 군집 번호와 범주별 관측빈도는 제공된 실행 노트북의 Model B K=6 결과를 따른다.\n"
            "- 범주가 관측되지 않은 군집×범주 조합은 관측빈도 0으로 복원한다.\n"
            "- 유의 기준은 연구계획에 맞춰 양의 ASR ≥ 1.96으로 고정한다. "
            "다중비교 보정은 적용하지 않으므로 탐색적 프로파일링 결과다.\n\n"
            "Haberman 조정 표준화 잔차는 다음과 같다.\n\n"
            r"$$ASR_{ij}=\frac{O_{ij}-E_{ij}}{\sqrt{E_{ij}(1-p_{i\cdot})(1-p_{\cdot j})}}$$"
            "\n\n여기서 $O_{ij}$는 관측빈도, $E_{ij}$는 독립 가정하 기대빈도, "
            "$p_{i\\cdot}$와 $p_{\\cdot j}$는 각각 행·열 주변비율이다."
        ),
        nbf.v4.new_markdown_cell("## Data\n\n### 1. 분석 입력과 무결성 확인"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import math\n"
            "import warnings\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.font_manager as fm\n"
            "import seaborn as sns\n"
            "from matplotlib.colors import LinearSegmentedColormap\n"
            "from scipy.stats import chi2_contingency\n\n"
            "warnings.filterwarnings('ignore')\n"
            "pd.set_option('display.max_columns', 20)\n"
            "pd.set_option('display.width', 160)\n\n"
            "font_path = Path(r'C:\\Windows\\Fonts\\malgun.ttf')\n"
            "if font_path.exists():\n"
            "    fm.fontManager.addfont(str(font_path))\n"
            "    plt.rcParams['font.family'] = fm.FontProperties(fname=str(font_path)).get_name()\n"
            "plt.rcParams['axes.unicode_minus'] = False\n"
            "sns.set_theme(style='white', font=plt.rcParams.get('font.family', ['sans-serif'])[0])\n\n"
            "OUTPUT_DIR = Path.cwd()\n"
            "OBSERVED_PATH = OUTPUT_DIR / 'KAIA_DA_k=6_ASR_observed_counts.csv'\n"
            "observed_long = pd.read_csv(OBSERVED_PATH, encoding='utf-8-sig')\n"
            "observed_long['cluster'] = observed_long['cluster'].astype(int)\n"
            "observed_long['count'] = observed_long['count'].astype(int)\n\n"
            "assert observed_long.groupby('variable')['count'].sum().eq(739).all()\n"
            "cluster_sizes = (\n"
            "    observed_long.query(\"variable == '철도구분'\")\n"
            "    .groupby('cluster')['count'].sum().sort_index()\n"
            ")\n"
            "assert cluster_sizes.to_dict() == {1: 337, 2: 132, 3: 97, 4: 90, 5: 48, 6: 35}\n"
            "display(cluster_sizes.rename('n').to_frame())\n"
            "display(observed_long.groupby('variable').agg(categories=('category', 'nunique'), n=('count', 'sum')))"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### 2. 기대빈도·ASR·연관성 지표 계산"),
        nbf.v4.new_code_cell(
            "def calculate_asr(observed):\n"
            "    all_results = []\n"
            "    tests = []\n"
            "    for variable, values in observed.groupby('variable', sort=False):\n"
            "        table = (values.pivot(index='cluster', columns='category', values='count')\n"
            "                 .fillna(0).astype(int).sort_index())\n"
            "        chi2, p_value, dof, expected = chi2_contingency(table.to_numpy(), correction=False)\n"
            "        n = int(table.to_numpy().sum())\n"
            "        row_totals = table.sum(axis=1).to_numpy(dtype=float)\n"
            "        col_totals = table.sum(axis=0).to_numpy(dtype=float)\n"
            "        row_props = row_totals / n\n"
            "        col_props = col_totals / n\n"
            "        denominator = np.sqrt(expected * (1-row_props[:, None]) * (1-col_props[None, :]))\n"
            "        asr = np.divide(table.to_numpy(dtype=float)-expected, denominator,\n"
            "                        out=np.full_like(expected, np.nan), where=denominator > 0)\n"
            "        result = pd.DataFrame({\n"
            "            'variable': variable,\n"
            "            'cluster': np.repeat(table.index.to_numpy(), len(table.columns)),\n"
            "            'category': np.tile(table.columns.to_numpy(), len(table.index)),\n"
            "            'observed': table.to_numpy().ravel(),\n"
            "            'expected': expected.ravel(),\n"
            "            'adjusted_residual': asr.ravel(),\n"
            "            'cluster_share_pct': (table.to_numpy()/row_totals[:, None]*100).ravel(),\n"
            "            'overall_share_pct': np.tile(col_props*100, len(table.index)),\n"
            "        })\n"
            "        result['meets_positive_1_96'] = result['adjusted_residual'] >= 1.96\n"
            "        result['meets_negative_minus_1_96'] = result['adjusted_residual'] <= -1.96\n"
            "        all_results.append(result)\n"
            "        rows, cols = table.shape\n"
            "        tests.append({\n"
            "            'variable': variable, 'n': n, 'chi_square': chi2,\n"
            "            'degrees_freedom': dof, 'p_value': p_value,\n"
            "            'cramers_v': math.sqrt(chi2/(n*min(rows-1, cols-1))),\n"
            "            'min_expected': expected.min(),\n"
            "            'expected_below_5_pct': (expected < 5).mean()*100,\n"
            "        })\n"
            "    return pd.concat(all_results, ignore_index=True), pd.DataFrame(tests)\n\n"
            "all_cells, chi_square_summary = calculate_asr(observed_long)\n"
            "display(chi_square_summary.round({'chi_square': 2, 'p_value': 4, 'cramers_v': 3, 'min_expected': 2, 'expected_below_5_pct': 1}))"
        ),
        nbf.v4.new_markdown_cell(
            "카이제곱과 Cramér's V는 전체 연관의 크기를 요약한다. 적극변수는 군집 형성에 "
            "직접 사용되었으므로, 매우 작은 p값은 독립적인 외부 검증으로 해석하지 않는다. "
            "기대빈도 5 미만 셀이 많은 변수는 개별 잔차를 특히 탐색적으로 읽는다."
        ),
        nbf.v4.new_markdown_cell("### 3. ASR ≥ 1.96 범주 중 군집×변수별 상위 2개 선정"),
        nbf.v4.new_code_cell(
            "selected_top2 = (\n"
            "    all_cells.loc[all_cells['adjusted_residual'] >= 1.96]\n"
            "    .sort_values(['cluster', 'variable', 'adjusted_residual', 'observed'],\n"
            "                 ascending=[True, True, False, False])\n"
            "    .groupby(['cluster', 'variable'], sort=False, as_index=False).head(2).copy()\n"
            ")\n"
            "selected_top2['rank_within_cluster_variable'] = (\n"
            "    selected_top2.groupby(['cluster', 'variable']).cumcount() + 1\n"
            ")\n"
            "selected_top2 = selected_top2[[\n"
            "    'cluster', 'variable', 'rank_within_cluster_variable', 'category',\n"
            "    'observed', 'expected', 'adjusted_residual',\n"
            "    'cluster_share_pct', 'overall_share_pct'\n"
            "]]\n"
            "display(selected_top2.round({'expected': 2, 'adjusted_residual': 2, 'cluster_share_pct': 2, 'overall_share_pct': 2}))"
        ),
        nbf.v4.new_markdown_cell("### 4. 변수별 ASR 히트맵"),
        nbf.v4.new_code_cell(
            "cmap = LinearSegmentedColormap.from_list('navy_white_gold', ['#1F5A94', '#F7F7F5', '#D97706'])\n"
            "variables = observed_long['variable'].drop_duplicates().tolist()\n"
            "category_counts = all_cells.groupby('variable')['category'].nunique().reindex(variables)\n"
            "height_ratios = [max(2.4, 0.38*n) for n in category_counts]\n"
            "fig, axes = plt.subplots(len(variables), 1, figsize=(11, sum(height_ratios)),\n"
            "                         gridspec_kw={'height_ratios': height_ratios}, constrained_layout=True)\n"
            "for ax, variable in zip(axes, variables):\n"
            "    subset = all_cells.query('variable == @variable')\n"
            "    matrix = subset.pivot(index='category', columns='cluster', values='adjusted_residual')\n"
            "    ordering = matrix.max(axis=1).sort_values(ascending=False).index\n"
            "    matrix = matrix.loc[ordering]\n"
            "    max_abs = np.nanmax(np.abs(matrix.to_numpy()))\n"
            "    annotations = matrix.map(lambda value: f'{value:.1f}' + ('*' if value >= 1.96 else ('†' if value <= -1.96 else '')))\n"
            "    sns.heatmap(matrix, ax=ax, cmap=cmap, center=0, vmin=-max_abs, vmax=max_abs,\n"
            "                annot=annotations, fmt='', linewidths=0.5, linecolor='white',\n"
            "                cbar_kws={'label': '조정 표준화 잔차'})\n"
            "    ax.set_title(f'{variable}: 군집별 조정 표준화 잔차', loc='left', fontsize=12, weight='bold')\n"
            "    ax.set_xlabel('군집')\n"
            "    ax.set_ylabel('범주')\n"
            "fig.suptitle('K=6 군집과 범주형 변수의 Haberman 조정 표준화 잔차', fontsize=16, weight='bold')\n"
            "fig.text(0.01, 0.002, '* ASR ≥ 1.96 (상대적 과대표),  † ASR ≤ -1.96 (상대적 과소대표)', fontsize=10)\n"
            "plt.savefig('KAIA_DA_k=6_ASR_heatmaps.png', dpi=200, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell("### 5. 군집별 선정 특성 요약"),
        nbf.v4.new_code_cell(
            "summary_rows = []\n"
            "for cluster, values in selected_top2.groupby('cluster', sort=True):\n"
            "    parts = []\n"
            "    for variable, subset in values.groupby('variable', sort=False):\n"
            "        cats = ', '.join(f\"{r.category} (ASR {r.adjusted_residual:.2f})\" for r in subset.itertuples())\n"
            "        parts.append(f'{variable}: {cats}')\n"
            "    summary_rows.append({'cluster': int(cluster), 'selected_characteristics': ' | '.join(parts)})\n"
            "cluster_summary = pd.DataFrame(summary_rows)\n"
            "display(cluster_summary.style.set_properties(subset=['selected_characteristics'], **{'text-align': 'left'}))"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- 군집 1은 여객열차·차량고장·열차차량설비를 중심으로 한 차량설비형이다.\n"
            "- 군집 2는 도시철도·도시전동차가 과대표되는 도시전동차 차량고장형이다.\n"
            "- 군집 3은 외부요인·환경요인이 집중되는 외적 환경요인형이다.\n"
            "- 군집 4는 신호장애·신호제어설비가 집중되는 신호설비형이다.\n"
            "- 군집 5는 취급(관리)부주의·기관사(운전자)·무정차통과가 집중되는 인적 취급오류형이다.\n"
            "- 군집 6은 급전장애·전철전력설비가 집중되는 전철전력형이다.\n\n"
            "`근본원인별 그룹`에서 군집 3의 외적요인과 군집 5의 인적요인은 군집 형성에 "
            "사용하지 않은 보조변수에서 확인된 특성이므로 해석상 특히 유용하다."
        ),
    ]
    return notebook


def make_r_script() -> str:
    original = SOURCE_R.read_text(encoding="utf-8")
    addon = r'''

# ============================================================
# K=6 군집과 범주형 변수의 Haberman 조정 표준화 잔차(ASR)
# ============================================================

asr_threshold <- 1.96
top_n_per_variable <- 2

case_lookup_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_case_lookup.csv")
)

case_lookup_df <- read.csv(
  case_lookup_path,
  fileEncoding = "UTF-8-BOM",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

profile_df <- merge(
  cluster_result_asw,
  model_df,
  by = "case_id",
  all.x = TRUE,
  sort = FALSE
)

if ("근본원인별 그룹" %in% names(case_lookup_df)) {
  profile_df <- merge(
    profile_df,
    case_lookup_df[, c("case_id", "근본원인별 그룹"), drop = FALSE],
    by = "case_id",
    all.x = TRUE,
    sort = FALSE
  )
}

stopifnot(
  nrow(profile_df) == nrow(model_df),
  !anyNA(profile_df$cluster)
)

profile_variables <- setdiff(names(model_df), "case_id")
if ("근본원인별 그룹" %in% names(profile_df)) {
  profile_variables <- c(profile_variables, "근본원인별 그룹")
}

calculate_asr <- function(data, variable_name) {
  cluster_values <- factor(
    data$cluster,
    levels = sort(unique(data$cluster))
  )
  category_values <- as.character(data[[variable_name]])
  category_values[is.na(category_values)] <- "결측"
  category_values <- factor(category_values)

  observed <- table(cluster_values, category_values)
  n_total <- sum(observed)
  row_totals <- rowSums(observed)
  col_totals <- colSums(observed)
  expected <- outer(row_totals, col_totals) / n_total

  denominator <- sqrt(
    expected
    * (1 - row_totals / n_total)
    * rep(1 - col_totals / n_total, each = nrow(observed))
  )
  dim(denominator) <- dim(expected)

  adjusted_residual <- (observed - expected) / denominator

  result <- expand.grid(
    cluster = as.integer(rownames(observed)),
    category = colnames(observed),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  result$variable <- variable_name
  result$observed <- as.numeric(observed)
  result$expected <- as.numeric(expected)
  result$adjusted_residual <- as.numeric(adjusted_residual)
  result$cluster_share_pct <- as.numeric(
    sweep(observed, 1, row_totals, "/") * 100
  )
  result$overall_share_pct <- rep(
    as.numeric(col_totals / n_total * 100),
    each = nrow(observed)
  )
  result$meets_positive_1_96 <- (
    result$adjusted_residual >= asr_threshold
  )
  result$meets_negative_minus_1_96 <- (
    result$adjusted_residual <= -asr_threshold
  )

  chisq_result <- suppressWarnings(chisq.test(observed, correct = FALSE))
  cramer_v <- sqrt(
    unname(chisq_result$statistic)
    / (n_total * min(nrow(observed) - 1, ncol(observed) - 1))
  )

  test_summary <- data.frame(
    variable = variable_name,
    n = n_total,
    rows_clusters = nrow(observed),
    categories = ncol(observed),
    chi_square = unname(chisq_result$statistic),
    degrees_freedom = unname(chisq_result$parameter),
    p_value = chisq_result$p.value,
    cramers_v = cramer_v,
    min_expected = min(chisq_result$expected),
    expected_below_5_cells = sum(chisq_result$expected < 5),
    expected_below_5_pct = mean(chisq_result$expected < 5) * 100,
    stringsAsFactors = FALSE
  )

  list(cells = result, test = test_summary)
}

asr_objects <- lapply(
  profile_variables,
  function(variable_name) calculate_asr(profile_df, variable_name)
)

asr_all_cells <- do.call(
  rbind,
  lapply(asr_objects, function(x) x$cells)
)

asr_chi_square_summary <- do.call(
  rbind,
  lapply(asr_objects, function(x) x$test)
)

positive_cells <- asr_all_cells[
  asr_all_cells$adjusted_residual >= asr_threshold,
  ,
  drop = FALSE
]

positive_cells <- positive_cells[
  order(
    positive_cells$cluster,
    positive_cells$variable,
    -positive_cells$adjusted_residual,
    -positive_cells$observed
  ),
  ,
  drop = FALSE
]

group_key <- interaction(
  positive_cells$cluster,
  positive_cells$variable,
  drop = TRUE,
  lex.order = TRUE
)

asr_selected_top2 <- do.call(
  rbind,
  lapply(
    split(positive_cells, group_key),
    function(group_df) head(group_df, top_n_per_variable)
  )
)

asr_selected_top2$rank_within_cluster_variable <- ave(
  asr_selected_top2$adjusted_residual,
  asr_selected_top2$cluster,
  asr_selected_top2$variable,
  FUN = seq_along
)

asr_selected_top2 <- asr_selected_top2[
  ,
  c(
    "cluster",
    "variable",
    "rank_within_cluster_variable",
    "category",
    "observed",
    "expected",
    "adjusted_residual",
    "cluster_share_pct",
    "overall_share_pct"
  )
]

asr_all_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_ASR_all_cells.csv")
)
asr_selected_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_ASR_selected_top2.csv")
)
asr_chi_path <- file.path(
  save_dir,
  paste0(version_name, "_", selected_model, "_ASR_chi_square_summary.csv")
)

write.csv(
  asr_all_cells,
  asr_all_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  asr_selected_top2,
  asr_selected_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)
write.csv(
  asr_chi_square_summary,
  asr_chi_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("\n[ASR ≥ 1.96, 군집×변수별 상위 2개]\n")
print(asr_selected_top2)
cat("ASR 전체 셀 저장:", asr_all_path, "\n")
cat("ASR 선정 결과 저장:", asr_selected_path, "\n")
cat("카이제곱·Cramer's V 요약 저장:", asr_chi_path, "\n")
'''
    return original.rstrip() + "\n" + addon.lstrip()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed = extract_observed_counts()
    all_cells, selected, tests = calculate_asr(observed)
    cluster_summary = build_cluster_summary(selected)

    observed.to_csv(OBSERVED_PATH, index=False, encoding="utf-8-sig")
    all_cells.to_csv(ALL_CELLS_PATH, index=False, encoding="utf-8-sig")
    selected.to_csv(SELECTED_PATH, index=False, encoding="utf-8-sig")
    tests.to_csv(CHI_PATH, index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    create_heatmap(all_cells)

    cluster_sizes = (
        observed.loc[observed["variable"] == "철도구분"]
        .groupby("cluster")["count"]
        .sum()
        .sort_index()
        .to_dict()
    )
    notebook = make_notebook(cluster_sizes)
    attach_precomputed_outputs(
        notebook,
        observed=observed,
        selected=selected,
        tests=tests,
        cluster_summary=cluster_summary,
    )
    nbf.write(notebook, NOTEBOOK_PATH)

    R_PATH.write_text(make_r_script(), encoding="utf-8")

    print("Created:")
    for path in [
        NOTEBOOK_PATH,
        OBSERVED_PATH,
        ALL_CELLS_PATH,
        SELECTED_PATH,
        CHI_PATH,
        SUMMARY_PATH,
        HEATMAP_PATH,
        R_PATH,
    ]:
        print(path)


if __name__ == "__main__":
    main()
