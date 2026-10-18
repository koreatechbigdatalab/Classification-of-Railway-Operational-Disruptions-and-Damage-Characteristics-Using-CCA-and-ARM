import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
path = Path(r"C:\Users\l0535\AppData\Local\Temp\KAIA_DA_k=6.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))

default_indices = [2, 4, 19, 22, 25, 49, 51, 53, 57, 58, 60, 62, 66, 68, 70, 72, 74, 77, 83, 88, 89, 90]
source_only = "--source-only" in sys.argv
indices = [int(value) for value in sys.argv[1:] if value != "--source-only"] or default_indices

for index in indices:
    cell = notebook["cells"][index]
    print("\n" + "=" * 100)
    print(f"CELL {index} SOURCE")
    print("".join(cell.get("source", [])))
    if source_only:
        continue
    for output_index, output in enumerate(cell.get("outputs", [])):
        print("-" * 100)
        print(f"OUTPUT {output_index} type={output.get('output_type')}")
        if "text" in output:
            print("".join(output["text"]))
        data = output.get("data", {})
        if "text/plain" in data:
            print("[text/plain]")
            print("".join(data["text/plain"]))
        if "text/html" in data:
            html = "".join(data["text/html"])
            print(f"[text/html chars={len(html)}]")
            print(html)
