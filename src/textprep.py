"""
textprep.py - parsing, cleaning and feature construction for the OpinRank review files.

Used by 01_load_clean.py. Kept in a module (not the notebook) so that the sentiment scorer can
run in parallel worker processes on Windows, which must import the worker function by name.
"""
import html
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# =================================================================================================
# C1 - robust parsing of the two raw formats
# =================================================================================================
DATE_HOTEL = re.compile(r"^[A-Z][a-z]{2} \d{1,2} \d{4}\s*$")       # e.g. "Sep 9 2009 "


def read_text(path):
    """Decode a raw file: UTF-8 when valid, otherwise Windows-1252 (most hotel files).
    Returns (text, encoding_used)."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


def parse_hotels(root):
    """hotels/<city>/<country>_[<region>_]<city>_<hotel>: one review per line,
    'date<TAB>title<TAB>text<TAB>'. Lines that do not start a record (no date and fewer than 3
    fields) are continuation lines of a multi-line review and are appended to the previous one."""
    rows, n_cont, skipped = [], 0, []
    for f in sorted(Path(root, "hotels").glob("*/*")):
        city = f.parent.name                                         # e.g. 'new-york-city'
        # C9: the city appears in the file name with '-', '_' or ' ' between its words
        city_pat = re.compile(re.escape(city).replace(r"\-", "[-_ ]") + "_", re.I)
        parts = city_pat.split(f.name, maxsplit=1)
        hotel = (parts[1] if len(parts) == 2 else f.name).replace("_", " ").strip()
        text, enc = read_text(f)
        if f.suffix or "\x00" in text:          # e.g. a stray .rar archive: binary, not reviews
            skipped.append(f.name)
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            starts_record = len(fields) >= 3 and (DATE_HOTEL.match(fields[0]) or not fields[0].strip())
            if starts_record:
                rows.append(dict(domain="hotel", city=city, entity=hotel, date_raw=fields[0].strip(),
                                 title=fields[1].strip(), text=" ".join(fields[2:]).strip(),
                                 encoding=enc, file=f.name))
            elif rows and rows[-1]["file"] == f.name:
                rows[-1]["text"] += " " + " ".join(x.strip() for x in fields if x.strip())
                n_cont += 1
    return pd.DataFrame(rows), n_cont, skipped


DOC = re.compile(r"<DOC>(.*?)</DOC>", re.S)
TAG = {t: re.compile(rf"<{t}>(.*?)</{t}>", re.S) for t in ("DATE", "AUTHOR", "TEXT", "FAVORITE")}


def parse_cars(root):
    """cars/<year>/<year>_<make>_<model>: pseudo-XML <DOC> blocks. Parsed with regular expressions
    (not an XML parser) because the files contain ~5,300 unescaped '&' characters."""
    rows = []
    for f in sorted(Path(root, "cars").glob("*/*")):
        if f.suffix:
            continue
        year, make, *model = f.name.split("_")                     # C9: entity from the file name
        text, enc = read_text(f)
        for block in DOC.findall(text):
            get = {t: (m.group(1).strip() if (m := rx.search(block)) else None) for t, rx in TAG.items()}
            rows.append(dict(domain="car", model_year=int(year), make=make, model=" ".join(model),
                             entity=f"{make} {' '.join(model)}", date_raw=get["DATE"], author=get["AUTHOR"],
                             text=get["TEXT"], favorite=get["FAVORITE"], encoding=enc, file=f.name))
    return pd.DataFrame(rows)


# =================================================================================================
# C5 - noise smoothing of the review text
# =================================================================================================
TAGS = re.compile(r"<[^>]{1,200}>")
URL = re.compile(r"(https?://\S+|www\.\S+)")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]")
REPEAT_PUNCT = re.compile(r"([!?.,*~\-])\1{2,}")          # '!!!!' -> '!'
ELONGATED = re.compile(r"([a-zA-Z])\1{2,}")               # 'sooooo' -> 'soo'
NO_SPACE_PUNCT = re.compile(r"([a-z])([.!?;:,])([A-Z])")  # 'breakfast.Only' -> 'breakfast. Only'
NO_SPACE_CAMEL = re.compile(r"([a-z]{2})([A-Z][a-z])")    # 'InexpensiveCon' -> 'Inexpensive Con'
SPACES = re.compile(r"\s+")


def clean_text(s):
    """HTML entities -> characters, strip tags / URLs / control characters, collapse repeated
    punctuation and elongated letters, restore spaces lost where scraped lines were joined,
    normalise whitespace."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(html.unescape(s))                    # some entities are double-escaped
    s = TAGS.sub(" ", s)
    s = URL.sub(" ", s)
    s = CTRL.sub(" ", s)
    s = REPEAT_PUNCT.sub(r"\1", s)
    s = ELONGATED.sub(r"\1\1", s)
    s = NO_SPACE_PUNCT.sub(r"\1\2 \3", s)
    s = NO_SPACE_CAMEL.sub(r"\1 \2", s)
    return SPACES.sub(" ", s).strip()


# =================================================================================================
# C7 - language heuristic (no external language-detection library needed)
# =================================================================================================
EN_STOP = set("""the a an and or but of to in on at for with is was were are be been it this that
i we you he she they my our your his her their me us them as by from not no so very too had have
has do did there here if then than all would could should can will just""".split())
WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*")


def english_score(s):
    """Returns (share of '?' among non-space characters, share of English stop-words among
    tokens). Garbled CJK text survives in the raw files only as runs of '?'."""
    nonspace = len(s) - s.count(" ")
    q = s.count("?") / nonspace if nonspace else 1.0
    toks = WORD.findall(s.lower())
    stop = sum(t in EN_STOP for t in toks) / len(toks) if toks else 0.0
    return q, stop


# =================================================================================================
# Aspect lexicons (data selection of task-relevant vocabulary) and first-mention sequences
# =================================================================================================
HOTEL_ASPECTS = {
    "room": "room rooms suite suites",
    "staff": "staff employee employees receptionist receptionists reception concierge doorman doormen "
             "bellman bellmen housekeeping housekeeper manager front_desk",
    "location": "location located neighborhood neighbourhood subway metro tube station downtown walking_distance",
    "breakfast": "breakfast buffet brunch",
    "bed": "bed beds mattress pillow pillows sheets linen linens",
    "bathroom": "bathroom bathrooms shower showers bath bathtub tub toilet toilets towel towels",
    "price": "price prices priced value expensive cheap cost money worth overpriced affordable rate rates",
    "service": "service services",
    "view": "view views",
    "noise": "noise noisy quiet loud soundproof",
    "internet": "wifi wi-fi internet wireless",
    "pool": "pool pools swimming",
    "food": "restaurant restaurants food dinner meal meals lunch cuisine",
    "bar": "bar bars lounge drinks cocktail cocktails",
    "cleanliness": "clean cleanliness dirty spotless stain stains stained dusty",
    "parking": "parking valet garage",
    "gym_spa": "gym fitness spa sauna",
    "transport": "airport taxi cab shuttle transfer",
    "checkin": "checkin check-in checkout check-out check_in check_out",
    "lobby": "lobby",
    "elevator": "elevator elevators lift lifts",
    "size": "spacious small tiny cramped size huge",
}
CAR_ASPECTS = {
    "engine": "engine power powerful horsepower hp acceleration accelerate v6 v8 turbo torque performance pickup",
    "fuel_economy": "mpg mileage fuel gas economy hybrid",
    "comfort": "comfort comfortable uncomfortable",
    "ride": "ride rides smooth bumpy",
    "seats": "seat seats seating legroom headroom",
    "transmission": "transmission shifting shifts gears gear cvt manual automatic",
    "handling": "handling handles steering cornering corners maneuverability",
    "interior": "interior dashboard dash cabin console upholstery leather",
    "styling": "styling style looks exterior design stylish sporty attractive",
    "price": "price value cost money worth expensive cheap affordable priced",
    "noise": "noise noisy quiet loud",
    "reliability": "reliable reliability problem problems issue issues repair repairs recall recalls "
                   "defect defects broke broken warranty",
    "dealer": "dealer dealers dealership salesman salesperson",
    "brakes": "brake brakes braking",
    "cargo_space": "cargo trunk storage space roomy spacious",
    "safety": "safety safe airbag airbags crash",
    "electronics": "navigation nav stereo audio radio bluetooth ipod speakers electronics",
    "tires": "tire tires tyre tyres",
    "visibility": "visibility blind_spot",
}


def _compile(aspects):
    uni, bi = {}, {}
    for aspect, words in aspects.items():
        for w in words.split():
            if "_" in w:
                bi[tuple(w.split("_"))] = aspect
            else:
                uni[w] = aspect
    return uni, bi


LEX = {"hotel": _compile(HOTEL_ASPECTS), "car": _compile(CAR_ASPECTS)}


def aspect_sequence(text, domain):
    """Aspects in the order the reviewer FIRST mentions them (one pass over the tokens).
    The set of these aspects is the review's transaction; the ordered list its sequence."""
    uni, bi = LEX[domain]
    toks = WORD.findall(text.lower())
    seen, seq = set(), []
    for i, t in enumerate(toks):
        a = uni.get(t)
        if a is None and i + 1 < len(toks):
            a = bi.get((t, toks[i + 1]))
        if a is not None and a not in seen:
            seen.add(a)
            seq.append(a)
    return seq


# =================================================================================================
# Sentiment: VADER, averaged over sentences, scored in parallel worker processes
# =================================================================================================
def sentiment(texts, workers=6, chunk=2000):
    """Score a list of texts in parallel (worker code in vader_worker.py); returns a numpy
    array of mean sentence compound scores in [-1, 1]."""
    import nltk
    import vader_worker
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)          # ~100 KB lexicon, downloaded once
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    with ProcessPoolExecutor(max_workers=workers, initializer=vader_worker.init) as ex:
        parts = list(ex.map(vader_worker.score_chunk, chunks))
    return np.array([x for p in parts for x in p])
