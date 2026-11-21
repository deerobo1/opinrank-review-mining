"""
nbtools.py - turn a percent-format script (cells separated by '# %%' lines, markdown cells by
'# %% [markdown]') into a Jupyter notebook, execute it, and save it with all outputs.

    python nbtools.py 02_eda_plots.py [03_arm.py ...]

The .py scripts are the readable source files; the .ipynb files carry the outputs.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def script_to_notebook(path):
    """Parse '# %%' cells. Markdown cells have their leading '# ' stripped."""
    cells, kind, buf = [], None, []

    def flush():
        src = "\n".join(buf).strip("\n")
        if kind is None or not src.strip():
            return
        if kind == "md":
            src = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in src.splitlines())
            cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            cells.append(nbformat.v4.new_code_cell(src))

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("# %%"):
            flush()
            buf = []
            kind = "md" if "[markdown]" in line else "code"
        else:
            buf.append(line)
    flush()
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def run(script, timeout=None):
    """Build + execute; the notebook is written even if a cell fails (partial outputs kept)."""
    script = Path(script).resolve()
    nb = script_to_notebook(script)
    out = script.with_suffix(".ipynb")
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(script.parent)}})
    t = time.time()
    try:
        client.execute()
        print(f"{out.name}: {len(nb.cells)} cells executed in {time.time() - t:.0f}s")
    finally:
        nbformat.write(nb, out)


def merge(script):
    """Rebuild the notebook from the script WITHOUT re-executing: every code cell whose source is
    unchanged keeps its outputs from the last executed .ipynb. Used to add markdown commentary
    (interpretations) after a long run. Returns the number of code cells left without outputs."""
    script = Path(script).resolve()
    out = script.with_suffix(".ipynb")
    old = nbformat.read(out, as_version=4)
    done = {c.source: c for c in old.cells if c.cell_type == "code"}
    nb = script_to_notebook(script)
    missing = 0
    for c in nb.cells:
        if c.cell_type == "code":
            if c.source in done:
                c.outputs = done[c.source].outputs
                c.execution_count = done[c.source].execution_count
            else:
                missing += 1
    nbformat.write(nb, out)
    print(f"{out.name}: merged, {missing} code cells without outputs")
    return missing


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--merge":
        for s in args[1:]:
            merge(s)
    else:
        for s in args:
            run(s)
