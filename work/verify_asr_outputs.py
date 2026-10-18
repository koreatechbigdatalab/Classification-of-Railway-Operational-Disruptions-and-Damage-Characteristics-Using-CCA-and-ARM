from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
from statsmodels.stats.contingency_tables import Table


OUTPUT_DIR = Path(
    r"C:\Users\l0535\Documents\Codex\2026-10-19\new-chat\outputs"
)

required_files = [
    "KAIA_DA_k=6_ASR_analysis.ipynb",
    "KAIA_DA_k=6_ASR_observed_counts.csv",
    "KAIA_DA_k=6_ASR_all_cells.csv",
    "KAIA_DA_k=6_ASR_selected_top2.csv",
    "KAIA_DA_k=6_ASR_chi_square_summary.csv",
    "KAIA_DA_k=6_ASR_cluster_summary.csv",
    "KAIA_DA_k=6_ASR_heatmaps.png",
    "KAIA_DA_k=6_with_ASR.R",
]

for filename in required_files:
    path = OUTPUT_DIR / filename
    assert path.exists() and path.stat().st_size > 0, filename

observed = pd.read_csv(
    OUTPUT_DIR / "KAIA_DA_k=6_ASR_observed_counts.csv", encoding="utf-8-sig"
)
all_cells = pd.read_csv(
    OUTPUT_DIR / "KAIA_DA_k=6_ASR_all_cells.csv", encoding="utf-8-sig"
)
selected = pd.read_csv(
    OUTPUT_DIR / "KAIA_DA_k=6_ASR_selected_top2.csv", encoding="utf-8-sig"
)
tests = pd.read_csv(
    OUTPUT_DIR / "KAIA_DA_k=6_ASR_chi_square_summary.csv", encoding="utf-8-sig"
)

assert observed.groupby("variable")["count"].sum().eq(739).all()
cluster_sizes = (
    observed.loc[observed["variable"] == "철도구분"]
    .groupby("cluster")["count"]
    .sum()
    .sort_index()
    .to_dict()
)
assert cluster_sizes == {1: 337, 2: 132, 3: 97, 4: 90, 5: 48, 6: 35}
assert len(tests) == 7
assert (all_cells["expected"] > 0).all()

maximum_residual_difference = 0.0
for variable, values in observed.groupby("variable", sort=False):
    table = (
        values.pivot(index="cluster", columns="category", values="count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    reference = Table(table.to_numpy(), shift_zeros=False).standardized_resids
    actual = (
        all_cells.loc[all_cells["variable"] == variable]
        .pivot(index="cluster", columns="category", values="adjusted_residual")
        .loc[table.index, table.columns]
        .to_numpy()
    )
    maximum_residual_difference = max(
        maximum_residual_difference, float(np.max(np.abs(reference - actual)))
    )
assert maximum_residual_difference < 1e-10

recomputed_selection = (
    all_cells.loc[all_cells["adjusted_residual"] >= 1.96]
    .sort_values(
        ["cluster", "variable", "adjusted_residual", "observed"],
        ascending=[True, True, False, False],
    )
    .groupby(["cluster", "variable"], sort=False, as_index=False)
    .head(2)
    .copy()
)
recomputed_selection["rank_within_cluster_variable"] = (
    recomputed_selection.groupby(["cluster", "variable"]).cumcount() + 1
)
recomputed_selection = recomputed_selection[
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
].reset_index(drop=True)
pd.testing.assert_frame_equal(
    selected.reset_index(drop=True), recomputed_selection, check_exact=False, rtol=1e-12
)
assert selected.groupby(["cluster", "variable"]).size().le(2).all()
assert selected["adjusted_residual"].ge(1.96).all()

notebook_path = OUTPUT_DIR / "KAIA_DA_k=6_ASR_analysis.ipynb"
notebook = nbformat.read(notebook_path, as_version=4)
nbformat.validate(notebook)
assert len(notebook.cells) == 15
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert len(code_cells) == 5
assert all(cell.execution_count is not None for cell in code_cells)
assert all(cell.outputs for cell in code_cells)

embedded_image = base64.b64decode(notebook.cells[11].outputs[0].data["image/png"])
saved_image = (OUTPUT_DIR / "KAIA_DA_k=6_ASR_heatmaps.png").read_bytes()
assert hashlib.sha256(embedded_image).digest() == hashlib.sha256(saved_image).digest()
with Image.open(OUTPUT_DIR / "KAIA_DA_k=6_ASR_heatmaps.png") as image:
    assert image.width >= 2000 and image.height >= 4000
    image_dimensions = image.size

r_text = (OUTPUT_DIR / "KAIA_DA_k=6_with_ASR.R").read_text(encoding="utf-8")
for required_text in [
    "asr_threshold <- 1.96",
    "adjusted_residual <- (observed - expected) / denominator",
    "asr_selected_top2",
    "근본원인별 그룹",
]:
    assert required_text in r_text

print("PASS: required artifacts exist and are non-empty")
print("PASS: 7 variable tables each reconcile to n=739")
print(f"PASS: cluster sizes reconcile: {cluster_sizes}")
print(
    "PASS: Haberman ASR matches statsmodels for all variables; "
    f"max absolute difference={maximum_residual_difference:.3e}"
)
print(f"PASS: top-2 selection rule reproduced exactly ({len(selected)} selected rows)")
print("PASS: notebook schema and populated outputs validated")
print(f"PASS: saved and embedded heatmap match; dimensions={image_dimensions}")
print("PASS: R deliverable contains ASR calculation and selection pipeline")
