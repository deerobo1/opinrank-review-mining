# %% [markdown]
# # 01 · Loading and cleaning the OpinRank review dataset
#
# **ASBD lab exam: descriptive analytics, KDD preprocessing (cleaning) and attribute construction**
#
# **Dataset:** OpinRank Review Dataset (UCI #205; Ganesan & Zhai, *Opinion-Based Entity Ranking*, 2011).
#
# | Part | Source | Unit | Raw format | Fields |
# |---|---|---|---|---|
# | **Cars** | Edmunds, model years 2007–2009 | one file per car model | pseudo-XML `<DOC>` blocks | DATE, AUTHOR, TEXT, FAVORITE |
# | **Hotels** | TripAdvisor, 10 cities | one file per hotel | one review per line, tab-separated | date, title, text |
#
# Neither part carries a numeric rating, so every numeric attribute used later (length, sentiment,
# aspect mentions, dates) is **constructed from the text** in this notebook.
#
# | Step | Dirty-data type | Technique applied |
# |---|---|---|
# | C1 | format / encoding | robust parsing: regex per `<DOC>` block, tab-split lines, UTF-8 or else cp1252 decoding, re-joining of multi-line reviews, skipping binary files |
# | C2 | incomplete | ignore the tuple when the review text is missing |
# | C3 | incomplete | fill optional fields with a global constant |
# | C4 | noisy | smoothing of the text: HTML entities, tags, URLs, repeated characters, missing spaces |
# | C5 | redundant | drop duplicate reviews |
# | C6 | noisy / outliers | drop near-empty reviews; flag length outliers with box-plot IQR fences |
# | C7 | noisy | drop garbled or non-English text |
# | C8 | inconsistent | unify two date formats; check for impossible dates |
# | C9 | inconsistent naming | normalise entity names taken from inconsistent file names |
# | C10 | noisy (numeric) | equal-depth binning, then smoothing by bin means and by bin boundaries |

# %%
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import textprep as tp      # parsing / cleaning / feature helpers (src/textprep.py)

warnings.filterwarnings("ignore", category=FutureWarning)
RAW = Path(r"C:\data\ASBD_exam\raw\OpinRank")      # extracted archive, kept outside OneDrive
CLEAN = Path(r"C:\data\ASBD_exam\clean")
CLEAN.mkdir(parents=True, exist_ok=True)
TABLES = Path("../outputs/tables")
TABLES.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 180)
pd.set_option("display.max_colwidth", 90)

log, issues = [], []          # cleaning log (rows per step) and issue counts (for the report)


def record(step, rule, kind, before, after):
    """Add one step to the cleaning log and print it."""
    log.append(dict(step=step, dirty_data_type=kind, rule=rule, rows_before=before,
                    rows_after=after, removed=before - after))
    print(f"{step} [{kind}] {rule}\n     rows: {before:,} -> {after:,}  (removed {before - after:,})")


def issue(name, mask):
    """Count an issue per domain (used for the before/after cleaning bar chart)."""
    for d, n in df.loc[mask].groupby("domain").size().items():
        issues.append(dict(issue=name, domain=d, count=int(n)))

# %% [markdown]
# ## C1 · Parsing the raw files (format and encoding problems)
# * **Hotels:** a record is a line `date<TAB>title<TAB>text<TAB>`. Reviews that contain line breaks
#   continue on following lines that have no date and fewer than 3 fields. These continuation lines
#   are appended to the previous review instead of being read as reviews of their own.
# * **Cars:** each `<DOC>…</DOC>` block is one review. The files contain unescaped `&`
#   characters, which break an XML parser, so the fields are extracted with regular expressions.
# * **Encoding:** a file is decoded as UTF-8 when it is valid UTF-8, otherwise as Windows-1252.
# * **Binary files:** any file with an extension or NUL bytes is skipped.
#
# *Finding:* the hotel folder contains one stray RAR archive (`…riviera_hotel_casino.rar`), a
# compressed copy of a hotel file that also exists as plain text. Read as text, it produced about
# 8,900 malformed "lines" and a couple of junk records. Once it is skipped, no hotel review spans
# more than one line, so the continuation rule never fires (0 lines re-joined).

# %%
t0 = time.time()
hotels, n_cont, skipped = tp.parse_hotels(RAW)
cars = tp.parse_cars(RAW)
df = pd.concat([hotels, cars], ignore_index=True)
print(f"parsed {len(hotels):,} hotel and {len(cars):,} car records in {time.time() - t0:.0f}s")
print(f"multi-line hotel reviews re-joined: {n_cont:,} continuation lines appended")
print(f"binary files skipped: {skipped}")
print("records by decoding used:")
print(df.groupby(["domain", "encoding"]).size().unstack(fill_value=0))
issues.append(dict(issue="continuation lines re-joined", domain="hotel", count=n_cont))
record("C1", "parse <DOC> blocks / tab-separated lines; UTF-8 else cp1252; re-join multi-line reviews; skip binary files",
       "format / encoding", len(df), len(df))

# %% [markdown]
# ## C2 · Missing review text: ignore the tuple
# The text is the attribute every later task depends on (aspects, sentiment, ARM, clustering),
# so a review without text has nothing to analyse. The standard "ignore the tuple" rule applies
# (it is normally used when the class label is missing).

# %%
df["text_raw"] = df.pop("text").fillna("")       # pop: keep one copy of the text in memory
empty = df["text_raw"].str.strip().eq("")
issue("empty review text", empty)
print(empty.groupby(df["domain"]).sum().rename("empty texts"))
before = len(df)
df = df[~empty].copy()
record("C2", "ignore the tuple when the review TEXT is missing or empty", "incomplete", before, len(df))

# %% [markdown]
# ## C3 · Missing optional fields: fill with a global constant
# Title (hotels), FAVORITE and AUTHOR (cars) are optional. They are filled with a constant (`""`, or
# `"unknown"` for the author) so that no row is lost. A missing **date** is kept as `NaT`: an
# invented date would be wrong, and those rows are only left out of the time-based plots.

# %%
H, C = df["domain"].eq("hotel"), df["domain"].eq("car")
miss = {
    "hotel title": H & df["title"].fillna("").str.strip().eq(""),
    "car favorite": C & df["favorite"].fillna("").str.strip().eq(""),
    "car author": C & df["author"].fillna("").str.strip().eq(""),
    "date (any)": df["date_raw"].fillna("").str.strip().eq(""),
}
print(pd.Series({k: int(v.sum()) for k, v in miss.items()}, name="missing values"))
issue("missing date", miss["date (any)"])
issue("missing favorite / title", miss["hotel title"] | miss["car favorite"])
df["title"] = df["title"].fillna("").str.strip()
df["favorite"] = df["favorite"].fillna("").str.strip()
df["author"] = df["author"].fillna("").str.strip().replace("", "unknown")
record("C3", "fill missing title/favorite with '' and author with 'unknown'; keep missing dates as NaT",
       "incomplete", len(df), len(df))

# %% [markdown]
# ## C4 · Noise smoothing of the text
# Count the noise patterns first, then smooth them with `textprep.clean_text`:
# 1. unescape HTML entities (`&amp;`, `&quot;`, `&#39;`);
# 2. strip HTML tags, URLs and control characters;
# 3. collapse repeated punctuation (`!!!!` → `!`) and elongated letters (`sooooo` → `soo`);
# 4. put back the spaces lost where scraped lines were glued together
#    (`breakfast.Only` → `breakfast. Only`, `InexpensiveCon's` → `Inexpensive Con's`).
#
# Shouting and exclamation are measured **before** smoothing, because smoothing removes them.

# %%
patterns = {
    "HTML entity": r"&(?:[a-z]+|#\d+);",
    "HTML tag": r"<[^>]{1,200}>",
    "URL": r"https?://|www\.",
    "repeated punctuation": r"([!?.,*~\-])\1{2,}",
    "elongated word": r"([a-zA-Z])\1{2,}",
    "missing space": r"[a-z][.!?;:,][A-Z]|[a-z]{2}[A-Z][a-z]",
}
# Python's re is used here because pandas' pyarrow-backed strings use RE2, which has no
# back-references (\1) for the repeated-character patterns
noise = pd.DataFrame({name: df["text_raw"].map(re.compile(p).search).notna() for name, p in patterns.items()})
print("reviews affected by each noise pattern:")
print(noise.groupby(df["domain"]).sum().T)
for name in patterns:
    issue(name, noise[name])

df["exclaim_count"] = df["text_raw"].str.count("!")
letters = df["text_raw"].str.count(r"[A-Za-z]").clip(lower=1)
df["upper_ratio"] = df["text_raw"].str.count(r"[A-Z]") / letters
t0 = time.time()
df["text"] = df["text_raw"].map(tp.clean_text)
df["title"] = df["title"].map(tp.clean_text)
df["favorite"] = df["favorite"].map(tp.clean_text)
print(f"smoothed in {time.time() - t0:.0f}s")
changed_early = df["text_raw"].str[:160] != df["text"].str[:160]
for name in ["HTML entity", "missing space", "repeated punctuation"]:
    i = df.index[noise[name] & changed_early][0]
    print(f"\n[{name}]\n  BEFORE: {df.at[i, 'text_raw'][:160]}\n  AFTER:  {df.at[i, 'text'][:160]}")
df = df.drop(columns="text_raw")                 # raw text no longer needed (saves memory)
del noise, changed_early
record("C4", "unescape HTML entities, strip tags/URLs/control chars, collapse repeats, restore missing spaces",
       "noisy", len(df), len(df))

# %% [markdown]
# ## C5 · Duplicate reviews
# Reviews are compared on a **normalised** text (lower-case, punctuation removed). A copy can occur
# inside the same entity (a review scraped twice) or under a different entity (a hotel crawled under
# two file names, or a review cross-posted). Only the first occurrence is kept, so repeated text
# does not inflate pattern supports later.

# %%
df["norm"] = df["text"].str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
dup_same = df.duplicated(["domain", "entity", "norm"])
dup_any = df.duplicated(["domain", "norm"])
issue("duplicate review", dup_any)
print(pd.DataFrame({"same entity": dup_same.groupby(df["domain"]).sum(),
                    "different entity": (dup_any & ~dup_same).groupby(df["domain"]).sum()}))
before = len(df)
df = df[~dup_any].drop(columns="norm")
record("C5", "drop duplicate reviews (same normalised text within a domain, keep the first)",
       "redundant", before, len(df))

# %% [markdown]
# ## C6 · Near-empty reviews and length outliers
# * Fewer than **5 words** ("Great!", "ok") carries no analysable content → removed.
# * Very long reviews are flagged with the **box-plot rule**: above the upper fence
#   Q3 + 1.5·IQR, computed per domain because car reviews are much shorter. They are genuine,
#   detailed reviews, so they are **kept and flagged**, not removed. TF-IDF normalisation later
#   stops them from dominating.

# %%
df["word_count"] = df["text"].str.split().str.len()
short = df["word_count"] < 5
issue("fewer than 5 words", short)
before = len(df)
df = df[~short].copy()
record("C6a", "drop reviews with fewer than 5 words", "noisy", before, len(df))

fence = {}
for d, g in df.groupby("domain"):
    q1, q3 = g["word_count"].quantile([0.25, 0.75])
    fence[d] = dict(Q1=q1, Q3=q3, IQR=q3 - q1, lower=q1 - 1.5 * (q3 - q1), upper=q3 + 1.5 * (q3 - q1))
print(pd.DataFrame(fence).T.round(1))
df["long_outlier"] = df["word_count"] > df["domain"].map({d: f["upper"] for d, f in fence.items()})
issue("length outlier (flagged)", df["long_outlier"])
print(df.groupby("domain")["long_outlier"].agg(["sum", "mean"]).rename(columns={"sum": "flagged", "mean": "share"}))
print("\nlongest reviews:")
print(df.nlargest(3, "word_count")[["domain", "entity", "word_count"]].assign(start=df["text"].str[:100]))
record("C6b", "flag (keep) reviews above the upper IQR fence Q3 + 1.5*IQR", "outliers", len(df), len(df))

# %% [markdown]
# ## C7 · Garbled or non-English text
# CJK characters were lost in the original crawl and survive only as runs of `?`. Two signals
# flag such text without a language-detection library:
# * the share of `?` among non-space characters is above 30 %, or
# * the review has at least 40 words but fewer than 5 % of them are common English stop-words.
#   (At 20 words the rule also caught terse English bullet lists such as "great breakfast, flat
#   screen TV, …", so the minimum length was raised to 40.)

# %%
sc = df["text"].map(tp.english_score)
df["q_share"], df["stop_share"] = sc.str[0], sc.str[1]
non_en = (df["q_share"] > 0.3) | ((df["word_count"] >= 40) & (df["stop_share"] < 0.05))
issue("garbled / non-English", non_en)
print(df.loc[non_en, ["domain", "entity", "word_count", "stop_share"]].assign(start=df["text"].str[:90]).head(8))
before = len(df)
df = df[~non_en].copy()
record("C7", "drop garbled/non-English text ('?' share > 30%, or >= 40 words with < 5% English stop-words)",
       "noisy", before, len(df))

# %% [markdown]
# ## C8 · Inconsistent dates
# The two parts use different formats (`Sep 9 2009` for hotels, `06/15/2009` for cars). Both are
# parsed into a single datetime type and checked for values that cannot be right:
# * a car reviewed more than a year before its model year (model-year-*Y* cars go on sale during *Y−1*);
# * dates after the collection (the files are dated Dec 2009 – Feb 2010);
# * hotel reviews before 2001 (TripAdvisor started in 2000).

# %%
H, C = df["domain"].eq("hotel"), df["domain"].eq("car")
d_h = pd.to_datetime(df["date_raw"].where(H), format="%b %d %Y", errors="coerce")
d_c = pd.to_datetime(df["date_raw"].where(C), format="%m/%d/%Y", errors="coerce")
df["date"] = d_h.fillna(d_c)
unparsed = df["date"].isna() & df["date_raw"].fillna("").str.strip().ne("")
bad = ((C & (df["date"].dt.year < df["model_year"] - 1))
       | (df["date"] > "2010-03-01")
       | (H & (df["date"] < "2001-01-01")))
print(f"unparseable non-empty dates: {unparsed.sum()}   impossible dates: {bad.sum()}")
print(df.groupby("domain")["date"].agg(["min", "max"]))
print("\ncar reviews: model year (rows) x review year (columns)")
print(pd.crosstab(df.loc[C, "model_year"], df.loc[C, "date"].dt.year))
issue("impossible / unparseable date", bad | unparsed)
before = len(df)
df = df[~(bad | unparsed)].copy()
record("C8", "unify the two date formats; drop impossible or unparseable dates", "inconsistent", before, len(df))

# %% [markdown]
# ## C9 · Entity names from inconsistent file names
# The hotel files use different prefix conventions: `usa_illinois_chicago_…`, `can_montreal_…` vs
# `can_quebec_montreal_…`, `india_new delhi_…` (with a space). The parser cut each name at the
# city name, so the same hotel stored under two prefixes becomes one entity. Car files follow
# `year_make_model`. Display names are title-cased.

# %%
hot = df[df["domain"].eq("hotel")]
files = hot.groupby("city").agg(files=("file", "nunique"), hotels=("entity", "nunique"))
files["merged"] = files["files"] - files["hotels"]
print(files)
multi = hot.groupby(["city", "entity"])["file"].nunique()
print(f"\nhotels stored under more than one file name: {(multi > 1).sum()}")
print("file-name conventions seen (country_region_city vs country_city):")
print(hot.drop_duplicates("file").assign(prefix=lambda d: d["file"].str.split("_").str[:3].str.join("_"))
      .groupby("city")["prefix"].agg(lambda s: s.value_counts().index[:2].tolist()).to_string())
CITY = {c: c.replace("-", " ").title() for c in df["city"].dropna().unique()}
CITY["new-york-city"] = "New York"
df["city_name"] = df["city"].map(CITY)
df["make"] = df["make"].str.title().replace({"Bmw": "BMW", "Gmc": "GMC", "Mercedes-Benz": "Mercedes-Benz"})
df["entity"] = df["entity"].str.title()
record("C9", "normalise entity names (hotel = file name after the city; car = year_make_model)",
       "inconsistent naming", len(df), len(df))

# %% [markdown]
# ## C10 · Smoothing a numeric attribute by binning
# Equal-depth (equal-frequency) binning on review length, worked on 12 values: sort, split into 3 bins of 4, then replace each value by its **bin mean** or by the
# **closer bin boundary**. The same idea is applied to the whole column with 10 equal-depth bins,
# giving a smoothed length attribute.

# %%
sample = sorted(df.loc[df["domain"].eq("car"), "word_count"].sample(12, random_state=7).tolist())
bins = [sample[i * 4:(i + 1) * 4] for i in range(3)]
demo = pd.DataFrame({
    "bin": [f"Bin {i + 1}" for i in range(3)],
    "values": [", ".join(map(str, b)) for b in bins],
    "smoothed by means": [", ".join([str(round(np.mean(b)))] * 4) for b in bins],
    "smoothed by boundaries": [", ".join(str(b[0] if x - b[0] <= b[-1] - x else b[-1]) for x in b) for b in bins],
})
print(demo.to_string(index=False))

df["wc_bin"] = df.groupby("domain")["word_count"].transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
df["word_count_smoothed"] = df.groupby(["domain", "wc_bin"])["word_count"].transform("mean").round(1)
print("\nequal-depth bins on the full column (hotels):")
print(df[df["domain"].eq("hotel")].groupby("wc_bin")["word_count"].agg(["min", "max", "mean", "size"]).round(1))
record("C10", "equal-depth binning of word_count (10 bins per domain) + smoothing by bin means",
       "noisy (numeric)", len(df), len(df))

# %% [markdown]
# ## Attribute construction (transformation, part 1)
# New attributes built from the cleaned text and dates:
#
# | Attribute | Meaning |
# |---|---|
# | `char_count`, `sentence_count`, `avg_word_len`, `title_words` | size and style of the review |
# | `exclaim_count`, `upper_ratio` | emphasis and "shouting", measured on the raw text |
# | `year`, `month`, `quarter`, `weekday` | date concept hierarchy: day < month < quarter < year |
# | `aspects` | the product/hotel aspects the review mentions, **in the order of first mention**. Lexicon of 22 hotel and 19 car aspects in `textprep.py` |
# | `sentiment` | VADER compound polarity averaged over sentences, in [−1, 1] |
# | `sent_class` | discretised sentiment: neg < −0.05 ≤ neu ≤ 0.05 < pos (VADER's standard cut-offs) |

# %%
df["char_count"] = df["text"].str.len()
df["sentence_count"] = df["text"].str.count(r"[.!?]+(?:\s|$)").clip(lower=1)
df["avg_word_len"] = (df["text"].str.count(r"[A-Za-z]") / df["word_count"]).round(3)
df["title_words"] = np.where(df["domain"].eq("hotel"), df["title"].str.split().str.len(), np.nan)
df["year"] = df["date"].dt.year.astype("Int64")
df["month"] = df["date"].dt.month.astype("Int64")
df["quarter"] = df["date"].dt.quarter.astype("Int64")
df["weekday"] = df["date"].dt.day_name()

t0 = time.time()
df["aspects"] = [tp.aspect_sequence(t, d) for t, d in zip(df["text"], df["domain"])]
df["n_aspects"] = df["aspects"].str.len()
print(f"aspects extracted in {time.time() - t0:.0f}s; mean aspects per review:")
print(df.groupby("domain")["n_aspects"].describe().round(2))

t0 = time.time()
df["sentiment"] = tp.sentiment(df["text"].tolist(), workers=6)
df["sent_class"] = pd.cut(df["sentiment"], [-1.001, -0.05, 0.05, 1.001], labels=["neg", "neu", "pos"]).astype(str)
print(f"\nsentiment scored in {time.time() - t0:.0f}s")
print(pd.crosstab(df["domain"], df["sent_class"], normalize="index").round(3))

# %% [markdown]
# ## Save the clean dataset and the cleaning log

# %%
keep = ["domain", "city", "city_name", "entity", "make", "model", "model_year", "file", "encoding",
        "date", "year", "month", "quarter", "weekday", "title", "text", "favorite",
        "word_count", "word_count_smoothed", "char_count", "sentence_count", "avg_word_len", "title_words",
        "exclaim_count", "upper_ratio", "long_outlier", "n_aspects", "aspects", "sentiment", "sent_class"]
df = df[keep].reset_index(drop=True)
df.to_parquet(CLEAN / "reviews_clean.parquet", index=False)
log_df = pd.DataFrame(log)
log_df.to_csv(TABLES / "cleaning_log.csv", index=False)
pd.DataFrame(issues).to_csv(TABLES / "noise_counts.csv", index=False)
print(log_df[["step", "dirty_data_type", "rows_before", "rows_after", "removed"]].to_string(index=False))
print(f"\nraw records: {len(hotels) + len(cars):,}  ->  clean reviews: {len(df):,}")
summary = df.groupby("domain").agg(reviews=("text", "size"), entities=("entity", "nunique"),
                                   first=("date", "min"), last=("date", "max"),
                                   median_words=("word_count", "median"), mean_sentiment=("sentiment", "mean"))
summary["mean_sentiment"] = summary["mean_sentiment"].round(3)
print(summary)
