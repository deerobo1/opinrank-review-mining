"""
plots.py - one consistent look for every figure in the ASBD exam report, plus helpers for the
plot types matplotlib does not draw directly (Pareto, radar, Wilkinson dot plot, stem-and-leaf,
doughnut, 100% stacked bar, stacked area).

Colour rules (palette checked with a colour-vision-deficiency validator before use):
  * categorical : SERIES, always assigned in this fixed order; past 7 groups the tail folds
                  into a grey "Other" so no colour is ever reused or generated.
  * polarity    : sentiment and correlation use a diverging red <-> grey <-> blue scheme.
  * magnitude   : a single blue hue, light -> dark.
  * one y-axis per chart: the Pareto chart draws its bars as % shares on the SAME 0-100 % axis
    as the cumulative line, instead of the usual (misleading) twin axis.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, PercentFormatter

# ---- palette & chrome ------------------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#c3c2b7"                                   # neutral grey for the folded "Other" bucket
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"  # primary / secondary / muted text
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DOMAIN = {"hotel": SERIES[0], "car": SERIES[1]}     # hotels = blue, cars = orange, everywhere
SENT = {"neg": "#e34948", "neu": "#c3c2b7", "pos": "#2a78d6"}   # diverging: red / grey / blue
DIVERGING = LinearSegmentedColormap.from_list("div", [SENT["neg"], "#f0efec", SENT["pos"]])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figs"


def set_style():
    """Apply the shared matplotlib style (thin marks, hairline solid grid, muted axes)."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150,
        "font.family": ["Segoe UI", "DejaVu Sans"],     # per-glyph fallback (e.g. for ⊆, ≥, →)
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "text.color": INK, "legend.frameon": False, "legend.fontsize": 9,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "lines.linewidth": 1.6,
    })


def save(fig, name, folder=None):
    """Save a figure as PNG into outputs/figs (or `folder`) and return the path."""
    out = Path(folder) if folder else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def _text_on(fill):
    """Label colour inside a filled mark: white, except near-black on yellow / light grey."""
    return INK if fill in (SERIES[3], OTHER, SENT["neu"], "#f0efec") else "#ffffff"


def fold_other(counts, keep=7):
    """Largest `keep` categories + 'Other' (only when there are more than 8 categories)."""
    s = pd.Series(counts).sort_values(ascending=False)
    if len(s) <= keep + 1:
        return s
    return pd.concat([s.iloc[:keep], pd.Series({"Other": s.iloc[keep:].sum()})])


def colors_for(names, palette=None):
    """Categorical colours for a list of names; 'Other' is grey.
    palette: optional {name: colour} built once (see entity_palette) so that an entity keeps the
    same colour in every chart; without it, colours follow the fixed slot order."""
    out, i = [], 0
    for n in names:
        if n == "Other":
            out.append(OTHER)
        elif palette and n in palette:
            out.append(palette[n])
        else:
            out.append(SERIES[i % len(SERIES)])
            i += 1
    return out


def entity_palette(counts, keep=7):
    """{name: colour} for the `keep` largest entities (slot order = size order). Everything else
    is drawn as 'Other', so colour follows the entity in every chart, never its rank there."""
    top = pd.Series(counts).sort_values(ascending=False).index[:keep]
    return {name: SERIES[i] for i, name in enumerate(top)}


# ---- 1. pie & 2. doughnut ----------------------------------------------------------------------
def pie(ax, counts, title, palette=None):
    """Pie chart: share of a whole. Slices sorted by size; % inside, name + count outside."""
    s = fold_other(counts)
    cols = colors_for(s.index, palette)
    _, _, autotexts = ax.pie(
        s.values, labels=[f"{k}\n{v:,}" for k, v in s.items()], colors=cols,
        autopct=lambda p: f"{p:.1f}%" if p >= 4 else "", pctdistance=0.72, labeldistance=1.1,
        startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor=SURFACE, linewidth=2), textprops=dict(color=INK2, fontsize=9))
    for t, c in zip(autotexts, cols):
        t.set_color(_text_on(c))
        t.set_fontsize(9)
    ax.set_title(title)
    ax.axis("equal")
    return s


def doughnut(ax, counts, title, colors, center):
    """Doughnut chart: a pie with a hole; the hole carries the headline (here the total).
    counts: Series in display order; colors: one colour per segment; center: text in the hole."""
    total = counts.sum()
    ax.pie(counts.values, colors=colors, startangle=90, counterclock=False,
           labels=[f"{k}  {v / total:.1%}" for k, v in counts.items()], labeldistance=1.08,
           textprops=dict(color=INK2, fontsize=9),
           wedgeprops=dict(width=0.36, edgecolor=SURFACE, linewidth=2))
    ax.text(0, 0, center, ha="center", va="center", fontsize=11, color=INK, fontweight="semibold")
    ax.set_title(title)
    ax.axis("equal")


# ---- 3. histogram -----------------------------------------------------------------------------
def histogram(ax, values, title, color, xlabel, bins=40, logx=False):
    """Histogram: frequencies of a continuous attribute in adjacent bins (bars touch).
    logx=True uses log-spaced bins, which suits right-skewed counts such as review length."""
    v = np.asarray(values, dtype=float)
    if logx:
        v = v[v > 0]
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), bins)
        ax.set_xscale("log")
    ax.hist(v, bins=bins, color=color, edgecolor="none")
    med = np.median(v)
    ax.axvline(med, color=INK, lw=1)
    ax.annotate(f"median {med:,.0f}", xy=(med, 1), xycoords=("data", "axes fraction"),
                xytext=(5, -12), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of reviews")
    ax.grid(axis="x", visible=False)


# ---- 4. dot plot & 5. stem-and-leaf -------------------------------------------------------------
def dot_plot(ax, values, title, color, xlabel):
    """Wilkinson dot plot: one dot per observation, stacked over its (integer) value.
    Suits small-to-moderate data and keeps every individual value visible."""
    counts = pd.Series(values).value_counts().sort_index()
    for x, c in counts.items():
        ax.scatter(np.full(c, x), np.arange(1, c + 1), s=60, color=color,
                   edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_ylim(0, counts.max() + 1)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


def stem_and_leaf(values, max_stems=15):
    """Stem-and-leaf display as text: stem = leading digits, leaf = last digit.
    The leaf unit grows (1, 10, 100, ...) until there are at most `max_stems` stems."""
    v = np.sort(np.asarray(values, dtype=float))
    unit = 1
    while int(v.max() // unit) // 10 - int(v.min() // unit) // 10 + 1 > max_stems:
        unit *= 10
    scaled = (v // unit).astype(int)                     # truncate to the leaf unit
    stems = defaultdict(list)
    for x in scaled:
        stems[x // 10].append(x % 10)
    lo, hi = min(stems), max(stems)
    w = len(str(hi))
    lines = [f"{s:>{w}} | {' '.join(str(leaf) for leaf in stems.get(s, []))}" for s in range(lo, hi + 1)]
    key = f"Key: 1 | 2 = {12 * unit:g}" + (f"   (leaf unit {unit:g}; values truncated)" if unit > 1 else "")
    return "\n".join(lines + ["", key])


def text_figure(text, title, fontsize=10):
    """Render monospaced text (e.g. a stem-and-leaf display) as a figure for the report."""
    n = text.count("\n") + 1
    fig = plt.figure(figsize=(7.5, 0.23 * n + 0.9))
    fig.text(0.02, 0.97, title, fontsize=12, fontweight="semibold", va="top")
    fig.text(0.02, 0.97 - 0.5 / (0.23 * n + 0.9), text, family="monospace", fontsize=fontsize,
             va="top", color=INK)
    return fig


# ---- 6. scatter & 7. box plot -------------------------------------------------------------------
def scatter(ax, x, y, title, xlabel, ylabel, color, logx=False, s=16, alpha=0.45):
    """Scatter plot with a least-squares trend line and Pearson r (on log10 x if logx)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ax.scatter(x, y, s=s, color=color, alpha=alpha, edgecolor="none", rasterized=True)
    xx = np.log10(x) if logx else x
    r = np.corrcoef(xx, y)[0, 1]
    b1, b0 = np.polyfit(xx, y, 1)
    grid = np.linspace(xx.min(), xx.max(), 100)
    ax.plot(10 ** grid if logx else grid, b0 + b1 * grid, color=INK, lw=1.2)
    if logx:
        ax.set_xscale("log")
    ax.annotate(f"Pearson r = {r:.2f}", xy=(0.02, 0.96), xycoords="axes fraction",
                va="top", color=INK2, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return r


def boxplot(ax, groups, title, color, ylabel, showfliers=True):
    """Box plot: five-number summary per group; points beyond 1.5*IQR are outliers.
    groups: dict {name: values} in display order."""
    face = mpl.colors.to_rgba(color, 0.18)
    ax.boxplot(list(groups.values()), tick_labels=list(groups.keys()), patch_artist=True,
               widths=0.55, showfliers=showfliers,
               medianprops=dict(color=INK, lw=1.5),
               boxprops=dict(facecolor=face, edgecolor=color, lw=1.2),
               whiskerprops=dict(color=color, lw=1.2), capprops=dict(color=color, lw=1.2),
               flierprops=dict(marker="o", markersize=2, markerfacecolor=MUTED,
                               markeredgecolor="none", alpha=0.25))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


# ---- 8. Pareto ------------------------------------------------------------------------------------
def pareto(ax, counts, title, color, threshold=80, xlabel=None):
    """Pareto chart: categories sorted by size (bars = % share) + cumulative % line, both on one
    0-100 % axis. Returns k = the number of categories needed to reach `threshold` %."""
    s = pd.Series(counts).sort_values(ascending=False)
    share = s / s.sum() * 100
    cum = share.cumsum().values
    x = np.arange(len(s))
    many = len(s) > 40
    ax.bar(x, share.values, color=color, width=1.0 if many else 0.75,
           edgecolor="none" if many else SURFACE, linewidth=0 if many else 1, label="share of total")
    ax.plot(x, cum, color=INK, lw=1.4, label="cumulative share")
    k = int(np.searchsorted(cum, threshold - 1e-9) + 1)
    ax.axhline(threshold, color=MUTED, lw=0.8)
    ax.axvline(k - 0.5, color=MUTED, lw=0.8)
    ax.annotate(f"{k} of {len(s):,} ({k / len(s):.0%}) cover {threshold}%",
                xy=(k - 0.5, threshold), xytext=(6, -16), textcoords="offset points",
                color=INK2, fontsize=9)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    if many:
        ax.set_xticks([])
        ax.set_xlabel(xlabel or f"{len(s):,} categories, ranked by size")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(s.index, rotation=45, ha="right")
        if xlabel:
            ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(loc="center right")
    ax.grid(axis="x", visible=False)
    return k


# ---- 9. radar ---------------------------------------------------------------------------------------
def radar(ax, profiles, title, ref=1.0):
    """Radar chart: one closed polygon per series over the same axes.
    profiles: DataFrame, rows = series (<= 3 for colour safety), columns = axes.
    ax must be polar (subplot_kw={'projection': 'polar'}). `ref` draws a reference ring
    (1.0 = the overall average when the values are indices)."""
    labels = list(profiles.columns)
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang = np.r_[ang, ang[:1]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for i, (name, row) in enumerate(profiles.iterrows()):
        v = np.r_[row.values, row.values[:1]]
        ax.plot(ang, v, color=SERIES[i], lw=1.8, label=name)
        ax.fill(ang, v, color=SERIES[i], alpha=0.08)
    if ref is not None:
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(t, np.full_like(t, ref), color=INK, lw=0.9, label="overall average")
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, color=INK2)
    ax.tick_params(axis="x", pad=7)
    ax.tick_params(axis="y", labelcolor=MUTED, labelsize=8)
    ax.spines["polar"].set_color(AXIS)
    ax.grid(color=GRID)
    ax.set_title(title, pad=22)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.08))


# ---- 10. stacked area, plus line, 100% stacked bar, horizontal bar, heatmap ------------------------
def stacked_area(ax, ts, title, ylabel="reviews per month", palette=None):
    """Stacked area chart: volume over time and each group's contribution.
    ts: DataFrame indexed by date, one column per group (counts). >8 groups -> top 7 + Other
    (or, with a palette, every group outside the palette folds into Other)."""
    order = ts.sum().sort_values(ascending=False).index
    ts = ts[order]
    if palette:
        keep = [c for c in order if c in palette]
        rest = [c for c in order if c not in palette]
        ts = ts[keep].assign(Other=ts[rest].sum(axis=1)) if rest else ts[keep]
    elif ts.shape[1] > 8:
        ts = ts[order[:7]].assign(Other=ts[order[7:]].sum(axis=1))
    ax.stackplot(ts.index, ts.T.values, labels=list(ts.columns),
                 colors=colors_for(ts.columns, palette), edgecolor=SURFACE, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", ncol=2)   # legend order = stack order
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(x=0)
    return ts


def lines(ax, df, title, ylabel, colors=None):
    """Line chart: one 2px line per column, labelled directly at its last point."""
    for i, col in enumerate(df.columns):
        c = (colors or {}).get(col, SERIES[i])
        d = df[col].dropna()
        ax.plot(d.index, d.values, color=c, lw=1.8, label=col)
        ax.annotate(str(col), xy=(d.index[-1], d.values[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK2, va="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower left")


def stacked_share(ax, ct, title, colors):
    """100% stacked horizontal bar: each group's split across classes (rows = groups)."""
    share = ct.div(ct.sum(axis=1), axis=0) * 100
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for col in share.columns:
        vals = share[col].values
        ax.barh(y, vals, left=left, color=colors[col], edgecolor=SURFACE, linewidth=1.5,
                height=0.72, label=str(col))
        for yi, (lft, v) in enumerate(zip(left, vals)):
            if v >= 7:
                ax.text(lft + v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color=_text_on(colors[col]))
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(share.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
    ax.set_title(title)
    ax.legend(ncol=len(share.columns), loc="upper center", bbox_to_anchor=(0.5, -0.06))
    ax.grid(axis="y", visible=False)
    return share


def hbar(ax, counts, title, color, xlabel, label_top=5):
    """Horizontal bar chart sorted by value (many categories, where a pie would be unreadable).
    Only the `label_top` largest bars carry a value label."""
    s = pd.Series(counts).sort_values()
    y = np.arange(len(s))
    ax.barh(y, s.values, color=color, height=0.72, edgecolor=SURFACE, linewidth=1)
    for yi in y[-label_top:]:
        ax.text(s.values[yi], yi, f" {s.values[yi]:,}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)


def corr_heatmap(ax, corr, title):
    """Correlation heatmap (lower triangle), diverging red-grey-blue scale fixed to [-1, 1]."""
    import seaborn as sns
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, annot=True, fmt=".2f",
                linewidths=2, linecolor=SURFACE, square=True, cbar_kws=dict(shrink=0.7),
                annot_kws=dict(fontsize=8), ax=ax)
    ax.set_title(title)
    ax.grid(False)


# ---- demo on synthetic data: python plots.py <output-folder> ------------------------------------------
def _demo(folder):
    mpl.use("Agg")
    set_style()
    rng = np.random.default_rng(0)
    cities = ["london", "new-york-city", "beijing", "chicago", "dubai", "las-vegas",
              "san-francisco", "shanghai", "montreal", "new-delhi"]
    counts = pd.Series(rng.integers(8000, 40000, len(cities)), index=cities)

    fig, ax = plt.subplots(figsize=(6.5, 6.5)); pie(ax, counts, "Hotel reviews by city"); save(fig, "demo_pie", folder)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (dom, n) in zip(axs, [("Hotels", 250000), ("Cars", 42000)]):
        sc = pd.Series({"pos": n * 0.8, "neu": n * 0.08, "neg": n * 0.12})
        doughnut(ax, sc, f"Sentiment mix: {dom.lower()}", [SENT[k] for k in sc.index], f"{dom}\n{n:,}")
    save(fig, "demo_doughnut", folder)
    fig, ax = plt.subplots(figsize=(7, 4)); histogram(ax, rng.lognormal(5, 0.8, 50000), "Review length", DOMAIN["hotel"], "words", logx=True); save(fig, "demo_hist", folder)
    fig, ax = plt.subplots(figsize=(7, 3.5)); dot_plot(ax, rng.integers(1, 15, 30), "Models per make", DOMAIN["car"], "models reviewed"); save(fig, "demo_dot", folder)
    print(stem_and_leaf(rng.integers(5, 640, 30)))
    save(text_figure(stem_and_leaf(rng.integers(5, 640, 30)), "Stem-and-leaf: reviews per model"), "demo_stem", folder)
    fig, ax = plt.subplots(figsize=(7, 4.5)); scatter(ax, rng.lognormal(4, 1, 800), rng.normal(0.6, 0.2, 800), "Hotel popularity vs sentiment", "reviews per hotel", "mean sentiment", DOMAIN["hotel"], logx=True); save(fig, "demo_scatter", folder)
    fig, ax = plt.subplots(figsize=(10, 4.5)); boxplot(ax, {c: rng.normal(0.5, 0.3, 2000).clip(-1, 1) for c in cities}, "Sentiment by city", DOMAIN["hotel"], "compound score"); ax.tick_params(axis="x", rotation=30); save(fig, "demo_box", folder)
    aspects = pd.Series(rng.pareto(1.2, 18) * 1000 + 50, index=[f"aspect{i}" for i in range(18)])
    fig, ax = plt.subplots(figsize=(8, 4.5)); pareto(ax, aspects, "Pareto: aspect mentions", DOMAIN["hotel"]); save(fig, "demo_pareto", folder)
    fig, ax = plt.subplots(figsize=(8, 4)); pareto(ax, pd.Series(rng.pareto(1.1, 1500) * 20 + 1), "Pareto: reviews per hotel", DOMAIN["hotel"], xlabel="hotels ranked by review count"); save(fig, "demo_pareto_many", folder)
    prof = pd.DataFrame(rng.uniform(0.5, 1.6, (3, 8)), index=cities[:3], columns=["room", "staff", "location", "breakfast", "price", "service", "bed", "view"])
    fig, ax = plt.subplots(figsize=(6.5, 6), subplot_kw={"projection": "polar"}); radar(ax, prof, "Aspect emphasis (1 = average)"); save(fig, "demo_radar", folder)
    idx = pd.date_range("2003-01-01", "2009-12-01", freq="MS")
    ts = pd.DataFrame({c: np.maximum(0, rng.normal(1, 0.3, len(idx)).cumsum() * 3) for c in cities}, index=idx)
    fig, ax = plt.subplots(figsize=(10, 4.5)); stacked_area(ax, ts, "Hotel reviews per month"); save(fig, "demo_area", folder)
    fig, ax = plt.subplots(figsize=(10, 4)); lines(ax, pd.DataFrame({"hotel": rng.normal(0.6, 0.05, len(idx)), "car": rng.normal(0.5, 0.08, len(idx))}, index=idx), "Mean sentiment per month", "compound", DOMAIN); save(fig, "demo_line", folder)
    ct = pd.DataFrame(rng.integers(100, 2000, (10, 3)), index=cities, columns=["neg", "neu", "pos"])
    fig, ax = plt.subplots(figsize=(9, 5)); stacked_share(ax, ct, "Sentiment class share by city", SENT); save(fig, "demo_stacked", folder)
    fig, ax = plt.subplots(figsize=(7, 7)); hbar(ax, pd.Series(rng.integers(50, 5000, 30), index=[f"make{i}" for i in range(30)]), "Reviews per car make", DOMAIN["car"], "reviews"); save(fig, "demo_hbar", folder)
    X = pd.DataFrame(rng.normal(size=(500, 6)), columns=list("abcdef")); X["g"] = X["a"] * 0.95 + rng.normal(0, 0.1, 500)
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); corr_heatmap(ax, X.corr(), "Correlation of derived features"); save(fig, "demo_heatmap", folder)
    plt.close("all")
    print("demo figures written to", folder)


if __name__ == "__main__":
    import sys
    _demo(sys.argv[1] if len(sys.argv) > 1 else "demo_figs")
