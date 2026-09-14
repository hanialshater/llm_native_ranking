"""Update only PoC source files and re-evaluate saved Colab checkpoints."""
import ast
import base64
import io
import json
from pathlib import Path
import runpy
import sys
import urllib.request
import zipfile

NOTEBOOK_URL="https://raw.githubusercontent.com/hanialshater/llm_native_ranking/autocomplete-grpo-colab-v2/Autocomplete_GRPO.ipynb"


def repair_and_evaluate(root,download=True):
    root=Path(root).resolve()
    if not (root/"results/llm/grpo/adapter_config.json").exists():
        raise FileNotFoundError("Saved GRPO adapter not found. Run this in the original Colab runtime that completed training.")
    notebook=json.loads(urllib.request.urlopen(NOTEBOOK_URL,timeout=60).read())
    setup="".join(notebook["cells"][1]["source"])
    assignment=next(n for n in ast.parse(setup).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="PAYLOAD" for t in n.targets))
    payload=ast.literal_eval(assignment.value)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as z:
        for member in z.infolist():
            rel=Path(member.filename)
            if len(rel.parts)!=2 or rel.parts[0]!="autocomplete_grpo" or rel.suffix!=".py":continue
            destination=(root/rel).resolve()
            if not destination.is_relative_to(root):raise ValueError("Invalid source path")
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(z.read(member.filename))
    # Existing data, adapters, logs, and original run.json remain available.
    runner=runpy.run_path(str(root/"autocomplete_grpo/runner.py"))["run_visible"]
    runner([sys.executable,"-u","-m","autocomplete_grpo.reevaluate"],root,"reevaluate.log")
    report=runpy.run_path(str(root/"autocomplete_grpo/report.py"))["show_results"]
    return report(root,run_dir="results/llm-constrained",download=download)
