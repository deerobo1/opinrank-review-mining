"""
build_full.py - merge the five executed part-notebooks into ONE submission notebook,
ASBD_OpinRank_full.ipynb, without re-running anything.

Layout of the combined notebook
  0. title + contents + how it was run
  1. helper code: one `%%writefile <module>.py` cell per helper module (the standard notebook idiom),
     executed here so their outputs are genuine, plus the algorithm self-tests
  2-6. Parts 1-5: every cell of 01..05 with its saved outputs, in order
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SRC = Path(__file__).resolve().parent
PARTS = ["01_load_clean", "02_eda_plots", "03_arm", "04_kmeans", "05_pyspark"]
HELPERS = ["algos.py", "textprep.py", "vader_worker.py", "plots.py"]
v4 = nbformat.v4

intro = v4.new_markdown_cell("""# ASBD Lab Exam: descriptive and predictive analytics on the OpinRank Review Dataset

**Dataset:** OpinRank Review Dataset (UCI #205; Ganesan & Zhai, 2011). 42k Edmunds car reviews (2007–2009) and
259k TripAdvisor hotel reviews from 10 cities. The report (PDF) summarises the results; this notebook contains all
the code and outputs.

| Part | Contents | Question item |
|---|---|---|
| 0 | Helper code: Apriori, A-Close, GSP (from scratch, self-tested), text parsing/cleaning, plot style | – |
| 1 | Loading and **data cleaning** (C1–C10), attribute construction (aspects, sentiment, dates) | preprocessing (i) |
| 2 | **Data selection**, **data transformation**, **14 visualisations** with inferences | preprocessing (ii), (iii), plots |
| 3 | **Apriori** association rules, **Apriori-based sequential patterns** (GSP), **A-Close** closed itemsets and rules | predictive (i), (iii) |
| 4 | **k-means** clustering (elbow, silhouette, if-then cluster descriptions) and a **dendrogram** | predictive (ii) |
| 5 | **PySpark** test drive: FP-Growth and k-means vs single-machine Python, scaling and speed-up | PySpark |

**How it was run.** Run from the `src/` folder with the raw data extracted to
`C:\\data\\ASBD_exam\\raw\\OpinRank`. Each part was executed in its own kernel session to keep memory use low on a
7.7 GB laptop, which is why the execution counters restart at the start of each part. Running this notebook top to
bottom reproduces every output (about 30 minutes in total).""")


def helper_cells():
    """%%writefile cells for the helper modules + a self-test cell, executed for real (in src/)."""
    cells = [v4.new_markdown_cell(
        "# Part 0 · Helper code\n"
        "The parts below import these modules. Each cell writes one module file (`%%writefile`), so the notebook is "
        "self-contained. `algos.py` holds the from-scratch algorithms; its self-test reproduces standard textbook examples "
        "(Han & Kamber's 9-transaction database for Apriori, a 5-transaction A-Close example) and checks A-Close against Apriori "
        "on 30 random databases.")]
    for h in HELPERS:
        cells.append(v4.new_code_cell(f"%%writefile {h}\n" + (SRC / h).read_text(encoding="utf-8")))
    cells.append(v4.new_code_cell("import algos\nalgos._selftest()"))
    nb = v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(SRC)}}).execute()
    return nb.cells


def main():
    cells = [intro] + helper_cells()
    for i, part in enumerate(PARTS, start=1):
        nb = nbformat.read(SRC / f"{part}.ipynb", as_version=4)
        first = nb.cells[0]
        if first.cell_type == "markdown" and first.source.startswith("# "):
            first.source = first.source.replace("# 0" + str(i) + " ·", f"# Part {i} ·", 1)
        cells.extend(nb.cells)
    full = v4.new_notebook(cells=cells)
    full.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    full.metadata["language_info"] = {"name": "python"}
    out = SRC / "ASBD_OpinRank_full.ipynb"
    nbformat.write(full, out)
    n_code = sum(c.cell_type == "code" for c in cells)
    print(f"{out.name}: {len(cells)} cells ({n_code} code), {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
