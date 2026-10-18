import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


path = Path(r"C:\Users\l0535\AppData\Local\Temp\KAIA_DA_k=6.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))

print(
    "nbformat",
    notebook.get("nbformat"),
    notebook.get("nbformat_minor"),
    "cells",
    len(notebook["cells"]),
)

for index, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", [])).strip().replace("\n", " ↩ ")
    print(
        f"{index:03d} {cell['cell_type']} "
        f"exec={cell.get('execution_count')} src={source[:300]}"
    )
    outputs = []
    for output in cell.get("outputs", []):
        outputs.append(
            (
                output.get("output_type"),
                list(output.get("data", {}).keys()),
                len("".join(output.get("text", []))) if "text" in output else "",
            )
        )
    print("    outputs", outputs[:8])
