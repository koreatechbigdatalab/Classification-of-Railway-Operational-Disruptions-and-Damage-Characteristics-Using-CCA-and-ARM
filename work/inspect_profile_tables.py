import json
import re
import sys
from io import StringIO
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
path = Path(r"C:\Users\l0535\AppData\Local\Temp\KAIA_DA_k=6.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
outputs = notebook["cells"][70]["outputs"]

current_variable = None
for output in outputs:
    if output.get("output_type") == "stream":
        text = "".join(output.get("text", []))
        match = re.search(r"\[군집별 (.+) 분포\]", text)
        if match:
            current_variable = match.group(1)
    html = "".join(output.get("data", {}).get("text/html", []))
    if html and current_variable:
        frame = pd.read_html(StringIO(html))[0]
        value_column = current_variable
        print(
            current_variable,
            "rows=",
            len(frame),
            "clusters=",
            sorted(frame["cluster"].dropna().unique().tolist()),
            "ellipsis=",
            frame.astype(str).apply(lambda col: col.str.contains("\.\.\.", regex=True).any()).any(),
            "categories=",
            frame[value_column].nunique(dropna=True),
        )
