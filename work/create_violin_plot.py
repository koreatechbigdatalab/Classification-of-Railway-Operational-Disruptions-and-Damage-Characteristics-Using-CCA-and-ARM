from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "work" / "plot_data_rebuild" / "visualization_dataset.json"
OUTPUT_DIR = ROOT / "outputs" / "kaia-k6-violin"
PNG_PATH = OUTPUT_DIR / "KAIA_DA_k6_violin_plot.png"
SVG_PATH = OUTPUT_DIR / "KAIA_DA_k6_violin_plot.svg"

CLUSTERS = [1, 2, 3, 4, 5, 6]
COLORS = {
    1: "#334E8A",
    2: "#4C78A8",
    3: "#2A9D8F",
    4: "#E9A23B",
    5: "#E76F51",
    6: "#7A5195",
}


@dataclass(frozen=True)
class Segment:
    low: float
    high: float
    ticks: tuple[float, ...]
    height: float


@dataclass(frozen=True)
class Metric:
    column: str
    title: str
    subtitle: str
    ylabel: str
    segments: tuple[Segment, ...]  # bottom to top
    positive_only: bool = False
    min_bandwidth: float = 0.5
    formatter: Callable[[float], str] = lambda value: f"{value:,.0f}"


METRICS = (
    Metric(
        column="총 피해액(백만원)",
        title="A. 피해액 발생 사례의 규모",
        subtitle="피해액 > 0인 사례 · 선형축 · 큰 값은 절단축으로 분리",
        ylabel="총 피해액 (백만원)",
        segments=(
            Segment(0, 100, (0, 20, 40, 60, 80, 100), 0.67),
            Segment(110, 900, (200, 400, 600, 800), 0.20),
            Segment(6200, 6500, (6200, 6300, 6400, 6500), 0.13),
        ),
        positive_only=True,
        min_bandwidth=0.15,
    ),
    Metric(
        column="합계지연열차수",
        title="B. 지연열차 수",
        subtitle="결측 제외 · 선형축 · 70–120 구간 절단",
        ylabel="합계 지연열차 수 (대)",
        segments=(
            Segment(0, 70, (0, 10, 20, 30, 40, 50, 60, 70), 0.72),
            Segment(120, 280, (120, 160, 200, 240, 280), 0.28),
        ),
        min_bandwidth=0.65,
    ),
    Metric(
        column="합계최대지연시간(분)",
        title="C. 최대 지연시간",
        subtitle="전체 사례 · 선형축 · 320–1,400 구간 절단",
        ylabel="합계 최대 지연시간 (분)",
        segments=(
            Segment(0, 320, (0, 50, 100, 150, 200, 250, 300), 0.84),
            Segment(1400, 1460, (1400, 1420, 1440, 1460), 0.16),
        ),
        min_bandwidth=2.0,
    ),
)


def load_data() -> pd.DataFrame:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(payload["case_wide"])
    expected = {"cluster", *(metric.column for metric in METRICS)}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"필수 열 누락: {sorted(missing)}")
    if len(frame) != 739:
        raise ValueError(f"예상 사례 수 739건과 다름: {len(frame)}")
    return frame


def robust_kde(values: np.ndarray, min_bandwidth: float) -> gaussian_kde:
    std = float(np.std(values, ddof=1))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    robust_scale = float((q3 - q1) / 1.349)
    scale = min(std, robust_scale) if robust_scale > 0 else std
    bandwidth = max(min_bandwidth, 0.9 * scale * len(values) ** (-0.2))
    return gaussian_kde(values, bw_method=bandwidth / std)


def panel_arrays(frame: pd.DataFrame, metric: Metric) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for cluster in CLUSTERS:
        values = pd.to_numeric(
            frame.loc[frame["cluster"] == cluster, metric.column], errors="coerce"
        ).dropna()
        if metric.positive_only:
            values = values[values > 0]
        array = values.to_numpy(dtype=float)
        if len(array) < 2 or not np.isfinite(array).all():
            raise ValueError(f"군집 {cluster}, {metric.column}: KDE에 사용할 값이 부족함")
        arrays.append(array)
    return arrays


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.8, alpha=0.82)
    ax.grid(axis="x", visible=False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8792A2")
    ax.spines["bottom"].set_color("#8792A2")
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="y", colors="#4B5563", labelsize=9.0)


def draw_wave(ax: plt.Axes, y: float) -> None:
    phase = np.linspace(0, 3 * np.pi, 80)
    dx = np.linspace(-0.018, 0.018, 80)
    dy = 0.010 * np.sin(phase)
    for x in (0.0, 1.0):
        ax.plot(
            x + dx,
            y + dy,
            transform=ax.transAxes,
            color="#596579",
            linewidth=1.25,
            clip_on=False,
            zorder=20,
        )


def draw_panel(
    fig: plt.Figure,
    outer_slot,
    frame: pd.DataFrame,
    metric: Metric,
) -> tuple[plt.Axes, plt.Axes]:
    arrays = panel_arrays(frame, metric)
    top_first = tuple(reversed(metric.segments))
    inner = outer_slot.subgridspec(
        len(top_first),
        1,
        height_ratios=[segment.height for segment in top_first],
        hspace=0.09,
    )
    axes = [fig.add_subplot(inner[index, 0]) for index in range(len(top_first))]
    positions = np.arange(1, len(CLUSTERS) + 1)

    # One robust-bandwidth KDE per cluster; widths are normalized within cluster
    # across every visible linear segment.
    density_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    maxima: dict[int, float] = {}
    for cluster_index, values in enumerate(arrays):
        kde = robust_kde(values, metric.min_bandwidth)
        local_max = 0.0
        for segment_index, segment in enumerate(top_first):
            grid = np.linspace(segment.low, segment.high, 500)
            density = kde(grid)
            density_cache[(cluster_index, segment_index)] = (grid, density)
            local_max = max(local_max, float(density.max()))
        maxima[cluster_index] = local_max

    rng = np.random.default_rng(20261019)
    for segment_index, (ax, segment) in enumerate(zip(axes, top_first)):
        style_axis(ax)
        ax.set_ylim(segment.low, segment.high)
        ax.set_yticks(segment.ticks)
        ax.set_yticklabels([metric.formatter(value) for value in segment.ticks])
        ax.set_xlim(0.45, 6.55)

        for cluster_index, (cluster, values) in enumerate(zip(CLUSTERS, arrays)):
            x = positions[cluster_index]
            grid, density = density_cache[(cluster_index, segment_index)]
            half_width = 0.37 * density / maxima[cluster_index]
            visible = half_width > 0.001
            if visible.any():
                ax.fill_betweenx(
                    grid[visible],
                    x - half_width[visible],
                    x + half_width[visible],
                    facecolor=COLORS[cluster],
                    edgecolor="#24303F",
                    linewidth=0.75,
                    alpha=0.82,
                    zorder=2,
                )

            # Sparse observations above the main segment are shown explicitly.
            if segment_index < len(top_first) - 1:
                in_segment = values[(values >= segment.low) & (values <= segment.high)]
                if len(in_segment):
                    jitter = rng.uniform(-0.075, 0.075, len(in_segment))
                    ax.scatter(
                        x + jitter,
                        in_segment,
                        s=20,
                        facecolor=COLORS[cluster],
                        edgecolor="#24303F",
                        linewidth=0.6,
                        alpha=0.92,
                        zorder=5,
                    )

        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    bottom_ax = axes[-1]
    bottom_ax.tick_params(axis="x", bottom=True, length=0, pad=8, labelbottom=True)
    bottom_ax.set_xticks(
        positions,
        [f"군집 {cluster}\nn={len(values):,}" for cluster, values in zip(CLUSTERS, arrays)],
    )
    bottom_ax.tick_params(axis="x", labelsize=9.2)

    # Quartiles and median lie in the main (bottom) segment for these metrics.
    main_segment = metric.segments[0]
    for x, values in zip(positions, arrays):
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        bottom_ax.plot(
            [x, x],
            [max(q1, main_segment.low), min(q3, main_segment.high)],
            color="#1F2937",
            linewidth=4.2,
            solid_capstyle="round",
            zorder=6,
        )
        bottom_ax.scatter(
            [x],
            [median],
            s=34,
            facecolor="white",
            edgecolor="#111827",
            linewidth=1.1,
            zorder=7,
            clip_on=False,
        )

    top_ax = axes[0]
    top_ax.set_title(metric.title, loc="left", fontsize=15, fontweight="bold", pad=26)
    top_ax.text(
        0,
        1.035,
        metric.subtitle,
        transform=top_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.6,
        color="#596579",
    )

    for upper_ax, lower_ax in zip(axes[:-1], axes[1:]):
        upper_ax.spines["bottom"].set_visible(False)
        lower_ax.spines["top"].set_visible(False)
        draw_wave(upper_ax, 0.0)
        draw_wave(lower_ax, 1.0)

    top_box = top_ax.get_position()
    bottom_box = bottom_ax.get_position()
    fig.text(
        bottom_box.x0 - 0.034,
        (top_box.y1 + bottom_box.y0) / 2,
        metric.ylabel,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10.3,
        color="#374151",
    )
    return top_ax, bottom_ax


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_data()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Malgun Gothic", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "#F7F8FA",
            "text.color": "#1F2937",
            "svg.fonttype": "none",
        }
    )

    fig = plt.figure(figsize=(17.5, 8.35), dpi=150)
    outer = fig.add_gridspec(
        1,
        3,
        left=0.058,
        right=0.985,
        top=0.77,
        bottom=0.19,
        wspace=0.25,
    )
    for index, metric in enumerate(METRICS):
        draw_panel(fig, outer[0, index], frame, metric)

    fig.text(
        0.058,
        0.935,
        "운행장애 군집별 핵심 결과 분포",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
        color="#18212F",
    )
    fig.text(
        0.058,
        0.885,
        "KAIA_DA · K=6 · 모든 눈금은 선형이며 0에서 시작 · 물결표시는 값이 없는 큰 구간의 축 절단",
        fontsize=11.2,
        ha="left",
        va="top",
        color="#596579",
    )

    legend_handles = [
        Line2D([0], [0], color="#1F2937", linewidth=4.2, label="사분위 범위 (Q1–Q3)"),
        Line2D(
            [0], [0], marker="o", linestyle="None", markerfacecolor="white",
            markeredgecolor="#111827", markersize=7, label="중앙값"
        ),
        Line2D(
            [0], [0], marker="o", linestyle="None", markerfacecolor="#7A5195",
            markeredgecolor="#24303F", markersize=6, label="절단축 구간의 개별 관측값"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.058, 0.095),
        ncol=3,
        frameon=False,
        fontsize=9.4,
        handlelength=2.0,
        columnspacing=1.5,
    )

    fig.text(
        0.058,
        0.045,
        "주: n은 지표별 유효 사례 수. 바이올린 폭은 군집 내부의 상대적 밀도이며 군집 간 사례 수를 뜻하지 않는다. "
        "절단된 구간에는 관측값이 없다.",
        fontsize=8.8,
        ha="left",
        va="bottom",
        color="#667085",
    )
    fig.text(
        0.985,
        0.045,
        "자료: 한국교통안전공단 국내 철도사고 및 운행장애 발생 정보 (2024-12-31 기준)",
        fontsize=8.8,
        ha="right",
        va="bottom",
        color="#667085",
    )

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(SVG_PATH, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"PNG={PNG_PATH}")
    print(f"SVG={SVG_PATH}")


if __name__ == "__main__":
    main()
