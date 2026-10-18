from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "work" / "plot_data_rebuild" / "visualization_dataset.json"
OUTPUT_DIR = ROOT / "outputs" / "kaia-k6-paper-figure"
PNG_PATH = OUTPUT_DIR / "KAIA_DA_k6_paper_interval_figure.png"
SVG_PATH = OUTPUT_DIR / "KAIA_DA_k6_paper_interval_figure.svg"
PDF_PATH = OUTPUT_DIR / "KAIA_DA_k6_paper_interval_figure.pdf"
CSV_PATH = OUTPUT_DIR / "KAIA_DA_k6_paper_interval_statistics.csv"

CLUSTERS = tuple(range(1, 7))
ACCENT = "#2F5597"
INK = "#202A36"
MID_GREY = "#7B8794"
LIGHT_GREY = "#D9DEE5"


def load_data() -> pd.DataFrame:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["case_wide"])
    required = {
        "cluster",
        "총 피해액(백만원)",
        "피해액발생여부",
        "합계지연열차수",
        "합계최대지연시간(분)",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {sorted(missing)}")
    if len(frame) != 739:
        raise ValueError(f"예상 사례 수 739건과 다름: {len(frame)}")
    return frame


def build_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for cluster in CLUSTERS:
        group = frame.loc[frame["cluster"] == cluster]
        total = len(group)
        damage = pd.to_numeric(group["총 피해액(백만원)"], errors="coerce")
        successes = int((damage > 0).sum())
        rows.append(
            {
                "panel": "A",
                "metric": "피해액 발생률",
                "cluster": cluster,
                "n": total,
                "numerator": successes,
                "estimate": 100 * successes / total,
                "low": np.nan,
                "q1": np.nan,
                "median": np.nan,
                "q3": np.nan,
                "high": np.nan,
                "interval": "none",
            }
        )

    metric_specs = (
        ("B", "양의 피해액", "총 피해액(백만원)", True),
        ("C", "지연열차 수", "합계지연열차수", False),
        ("D", "최대 지연시간", "합계최대지연시간(분)", False),
    )
    for panel, label, column, positive_only in metric_specs:
        for cluster in CLUSTERS:
            values = pd.to_numeric(
                frame.loc[frame["cluster"] == cluster, column], errors="coerce"
            ).dropna()
            if positive_only:
                values = values[values > 0]
            p10, q1, median, q3, p90 = values.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            rows.append(
                {
                    "panel": panel,
                    "metric": label,
                    "cluster": cluster,
                    "n": len(values),
                    "numerator": np.nan,
                    "estimate": median,
                    "low": p10,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "high": p90,
                    "interval": "P10–P90; IQR",
                }
            )

    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes, xmax: float, ticks: list[float]) -> None:
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.set_ylim(-0.6, 5.6)
    ax.set_yticks(np.arange(6))
    ax.invert_yaxis()
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.65)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", labelsize=7.5, colors=INK, width=0.6, length=3)
    ax.tick_params(axis="y", labelsize=7.5, colors=INK, length=0, pad=5)
    ax.set_axisbelow(True)


def set_panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.2, fontweight="bold", color=INK, pad=8)


def plot_rate_panel(ax: plt.Axes, stats: pd.DataFrame) -> None:
    panel = stats.loc[stats["panel"] == "A"].sort_values("cluster")
    style_axis(ax, 100, [0, 20, 40, 60, 80, 100])
    labels = [
        f"군집 {int(row.cluster)}  ({int(row.numerator)}/{int(row.n)})"
        for row in panel.itertuples()
    ]
    ax.set_yticklabels(labels)

    for y, row in enumerate(panel.itertuples()):
        ax.barh(
            y,
            row.estimate,
            height=0.34,
            color=ACCENT,
            edgecolor="none",
            zorder=2,
        )
        ax.text(
            min(row.estimate + 1.2, 98.5),
            y,
            f"{row.estimate:.1f}%",
            ha="left",
            va="center",
            fontsize=7.4,
            color=INK,
        )

    set_panel_title(ax, "A  피해액 발생률")
    ax.set_xlabel("피해액 발생률 (%)", fontsize=8.0, color=INK, labelpad=5)


def plot_interval_panel(
    ax: plt.Axes,
    stats: pd.DataFrame,
    panel_id: str,
    title: str,
    xlabel: str,
    xmax: float,
    ticks: list[float],
    decimals: int,
) -> None:
    panel = stats.loc[stats["panel"] == panel_id].sort_values("cluster")
    style_axis(ax, xmax, ticks)
    ax.set_yticklabels([f"군집 {int(row.cluster)}  (n={int(row.n)})" for row in panel.itertuples()])

    for y, row in enumerate(panel.itertuples()):
        ax.hlines(y, row.low, row.high, color=MID_GREY, linewidth=1.1, zorder=2)
        ax.hlines(y, row.q1, row.q3, color=ACCENT, linewidth=4.2, zorder=3)
        ax.scatter(
            row.median,
            y,
            s=34,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.0,
            zorder=4,
            clip_on=False,
        )
        if panel_id == "B":
            ax.annotate(
                f"{row.median:,.{decimals}f}",
                xy=(row.median, y),
                xytext=(6, -4),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=6.6,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.92},
                clip_on=False,
                zorder=5,
            )
        else:
            ax.text(
                row.median,
                y + 0.20,
                f"{row.median:,.{decimals}f}",
                ha="center",
                va="top",
                fontsize=6.6,
                color=INK,
                clip_on=False,
                zorder=5,
            )

    set_panel_title(ax, f"{panel_id}  {title}")
    ax.set_xlabel(xlabel, fontsize=8.0, color=INK, labelpad=5)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()
    stats = build_statistics(frame)
    stats.to_csv(CSV_PATH, index=False, encoding="utf-8-sig", float_format="%.6f")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Malgun Gothic", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # 180 mm wide: suitable for a typical two-column journal figure.
    fig, axes = plt.subplots(2, 2, figsize=(7.087, 6.94))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.955, bottom=0.235, wspace=0.42, hspace=0.54)

    plot_rate_panel(axes[0, 0], stats)
    plot_interval_panel(
        axes[0, 1],
        stats,
        "B",
        "피해액 규모",
        "총 피해액 (백만원)",
        90,
        [0, 20, 40, 60, 80],
        2,
    )
    plot_interval_panel(
        axes[1, 0],
        stats,
        "C",
        "지연열차 수",
        "합계 지연열차 수 (대)",
        35,
        [0, 5, 10, 15, 20, 25, 30, 35],
        0,
    )
    plot_interval_panel(
        axes[1, 1],
        stats,
        "D",
        "최대 지연시간",
        "합계 최대 지연시간 (분)",
        110,
        [0, 20, 40, 60, 80, 100],
        1,
    )

    legend_handles = [
        Line2D([0], [0], color=MID_GREY, linewidth=1.1, label="P10–P90"),
        Line2D([0], [0], color=ACCENT, linewidth=4.2, label="사분위 범위 (IQR)"),
        Line2D(
            [0], [0], marker="o", linestyle="None", markerfacecolor="white",
            markeredgecolor=INK, markersize=5.5, label="중앙값"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
        fontsize=7.6,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.text(
        0.5,
        0.012,
        "주: A의 막대는 발생률이다. B–D는 P10–P90, IQR, 중앙값을 나타내며 숫자는 중앙값이다. n은 지표별 유효 사례 수이다.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MID_GREY,
    )

    fig.savefig(PNG_PATH, dpi=600, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"PNG={PNG_PATH}")
    print(f"SVG={SVG_PATH}")
    print(f"PDF={PDF_PATH}")
    print(f"CSV={CSV_PATH}")


if __name__ == "__main__":
    main()
