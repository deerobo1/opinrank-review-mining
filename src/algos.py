"""
algos.py - from-scratch frequent-pattern algorithms for the ASBD lab exam (OpinRank reviews).

Contents
  1. apriori_bitset()     Apriori on a vertical TID-bitset layout; cross-checks mlxtend / Spark
  2. fi_closed_maximal()  splits frequent itemsets into FI / CFI / MFI
  3. aclose()             A-Close: minimal generators -> Galois closures = closed frequent itemsets
  4. support_from_cfis()  lossless property of CFIs: sup(X) = max support of a closed superset
  5. closed_rules()       association rules generated from the CFIs (min-max / generic basis)
  6. gsp()                Apriori-based (AprioriAll / GSP-style) sequential-pattern mining
  7. sequential_rules()   "prefix => suffix" rules from the frequent sequences
  8. self-tests on standard textbook toy databases  (run: python algos.py)

Conventions
  * A transaction is any iterable of hashable, sortable items (strings here).
  * min_sup may be an absolute count (int >= 1) or a fraction of the database (0 < float < 1).
  * Supports returned are COUNTS; divide by n for the fraction.
"""
from collections import defaultdict
import math

import numpy as np
import pandas as pd


def _min_count(min_sup, n):
    """Support threshold -> absolute count. A float in (0, 1) is a fraction of n."""
    if isinstance(min_sup, float) and 0 < min_sup < 1:
        return max(1, math.ceil(min_sup * n))
    return int(min_sup)


def _tid_bitsets(transactions):
    """Vertical data format: item -> TID-set.

    The TID-set is stored as a Python int used as a bitset (bit t is set when transaction t
    contains the item). Intersecting two TID-sets is then a single '&' and the support is a
    popcount (int.bit_count), so no database rescan is ever needed.
    The bitsets are built from TID lists with numpy.packbits, which is linear in the data size
    (OR-ing bits into a growing Python int one at a time would be quadratic).
    """
    tidlists, n = {}, 0
    for t, items in enumerate(transactions):
        for item in set(items):
            tidlists.setdefault(item, []).append(t)
        n = t + 1
    tids = {}
    for item, tl in tidlists.items():
        bits = np.zeros(n, dtype=np.uint8)
        bits[tl] = 1
        tids[item] = int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little")
    return tids


# --------------------------------------------------------------------------------------------
# 1. Apriori (level-wise, candidate generation with join + prune)
# --------------------------------------------------------------------------------------------
def apriori_bitset(transactions, min_sup, max_len=None):
    """Classic Apriori. Returns {frozenset(itemset): support_count} for every frequent itemset.

    Join : two frequent k-itemsets that share their first k-1 items (sorted order).
    Prune: drop a candidate if any of its k-subsets is not frequent (Apriori property).
    Count: AND the two parents' TID-bitsets (equivalent to one DB scan per level).
    """
    T = list(transactions)
    n = len(T)
    mc = _min_count(min_sup, n)
    tid = _tid_bitsets(T)
    level = {frozenset([i]): b for i, b in tid.items() if b.bit_count() >= mc}   # L1
    freq = {s: b.bit_count() for s, b in level.items()}
    k = 1
    while level and (max_len is None or k < max_len):
        keys = sorted(tuple(sorted(s)) for s in level)
        nxt = {}
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                a, b = keys[x], keys[y]
                if a[:-1] != b[:-1]:          # sorted list: once the prefix differs, stop
                    break
                cand = frozenset(a + (b[-1],))                       # JOIN
                if all((cand - {i}) in level for i in cand):          # PRUNE
                    bits = level[frozenset(a)] & level[frozenset(b)]  # COUNT
                    if bits.bit_count() >= mc:
                        nxt[cand] = bits
        level = nxt
        freq.update({s: b.bit_count() for s, b in level.items()})
        k += 1
    return freq


def rules_from_freq(freq, n, min_conf=0.6, consequent=None):
    """Strong rules X => Y from a complete frequent-itemset table:
    for every frequent l and every non-empty proper subset s, output s => l - s when
    conf = sup(l) / sup(s) >= min_conf.
        lift       = conf / P(Y)                       (> 1: X and Y occur together more than chance)
        conviction = (1 - P(Y)) / (1 - conf)           (inf for exact rules)
    consequent: optional item; keep only rules whose consequent is exactly {consequent}."""
    from itertools import combinations
    rows = []
    for l, sl in freq.items():
        if len(l) < 2 or (consequent is not None and consequent not in l):
            continue
        items = sorted(l)
        for r in range(1, len(l)):
            for ante in combinations(items, r):
                a = frozenset(ante)
                c = l - a
                if consequent is not None and c != {consequent}:
                    continue
                conf = sl / freq[a]
                if conf < min_conf:
                    continue
                pc = freq[c] / n
                rows.append(dict(antecedent=_fmt(a), consequent=_fmt(c), support=sl / n, confidence=conf,
                                 lift=conf / pc, conviction=(1 - pc) / (1 - conf) if conf < 1 else float("inf"),
                                 size=len(l)))
    df = pd.DataFrame(rows, columns=["antecedent", "consequent", "support", "confidence", "lift", "conviction", "size"])
    return df.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# 2. FI -> closed (CFI) and maximal (MFI)
# --------------------------------------------------------------------------------------------
def fi_closed_maximal(freq):
    """freq: {frozenset: count} holding ALL frequent itemsets (e.g. Apriori output).

    closed  : no immediate superset has the SAME support         (lossless summary)
    maximal : no immediate superset is frequent at all           (lossy summary)
    Each (k+1)-itemset marks its k-subsets, so this is linear in the number of itemsets.
    Returns (closed, maximal) as dicts. MFI is a subset of CFI, which is a subset of FI.
    """
    has_frequent_superset, has_equal_superset = set(), set()
    for t, ct in freq.items():
        for x in t:
            s = t - {x}
            if s in freq:
                has_frequent_superset.add(s)
                if freq[s] == ct:
                    has_equal_superset.add(s)
    closed = {s: c for s, c in freq.items() if s not in has_equal_superset}
    maximal = {s: c for s, c in freq.items() if s not in has_frequent_superset}
    return closed, maximal


# --------------------------------------------------------------------------------------------
# 3. A-Close (Pasquier et al. 1999)
# --------------------------------------------------------------------------------------------
def aclose(transactions, min_sup):
    """A-Close: find the closed frequent itemsets through their minimal generators.

    Galois connection: t(X) = transactions containing X, i(Y) = items common to all of Y,
    closure f(X) = i(t(X)). X is closed iff f(X) = X. A generator g of a closed set C has
    f(g) = C; it is MINIMAL if no proper subset of g has the same closure (= the same support).

    Phase 1 (level-wise, Apriori style) keeps only minimal generators:
        - drop infrequent candidates;
        - drop a candidate whose support equals that of one of its k-subsets
          (its closure is already produced by that smaller generator);
        - Apriori prune: every k-subset must itself be a minimal generator.
    Phase 2 computes the closure of every generator: f(g) = items whose TID-set contains t(g).

    Returns (cfis, table, gens)
        cfis : {frozenset(closed itemset): support_count}
        table: DataFrame with one row per minimal generator -> closure (for the report)
        gens : list of (generator, closure, support_count) frozenset tuples (input to closed_rules)
    """
    T = [frozenset(t) for t in transactions]
    n = len(T)
    mc = _min_count(min_sup, n)
    tid = _tid_bitsets(T)
    items = sorted(tid)
    all_bits = (1 << n) - 1                     # t(empty set) = every transaction

    def closure(bits):
        """f(X) = i(t(X)): every item present in ALL transactions of the TID-set 'bits'."""
        return frozenset(i for i in items if tid[i] & bits == bits)

    # Level 1: frequent single items. An item present in every transaction has the same
    # support as the empty set, so it is not a minimal generator (it lies in f(empty set)).
    level = {}
    for i in items:
        s = tid[i].bit_count()
        if mc <= s < n:
            level[frozenset([i])] = (s, tid[i])
    generators = dict(level)

    while level:
        keys = sorted(tuple(sorted(g)) for g in level)
        nxt = {}
        for x in range(len(keys)):
            for y in range(x + 1, len(keys)):
                a, b = keys[x], keys[y]
                if a[:-1] != b[:-1]:
                    break
                cand = frozenset(a + (b[-1],))
                subs = [cand - {i} for i in cand]
                if not all(s in level for s in subs):               # Apriori-style prune
                    continue
                bits = level[frozenset(a)][1] & level[frozenset(b)][1]
                sup = bits.bit_count()
                if sup < mc:                                         # infrequent
                    continue
                if any(level[s][0] == sup for s in subs):            # not minimal
                    continue
                nxt[cand] = (sup, bits)
        generators.update(nxt)
        level = nxt

    cfis, gens = {}, []
    base = closure(all_bits)                 # items occurring in every transaction (often empty)
    if base and n >= mc:
        cfis[base] = n
        gens.append((frozenset(), base, n))
    for g, (sup, bits) in generators.items():
        c = closure(bits)
        cfis[c] = sup
        gens.append((g, c, sup))
    table = pd.DataFrame([(_fmt(g), _fmt(c), s) for g, c, s in gens],
                         columns=["minimal_generator", "closure (CFI)", "support_count"])
    table["support"] = table["support_count"] / n
    table = table.sort_values(["support_count", "closure (CFI)"], ascending=[False, True])
    return cfis, table.reset_index(drop=True), gens


def _fmt(itemset):
    return "{" + ", ".join(sorted(map(str, itemset))) + "}"


# --------------------------------------------------------------------------------------------
# 4-5. Using the CFIs: lossless support recovery and closed association rules
# --------------------------------------------------------------------------------------------
def support_from_cfis(itemset, cfis):
    """Support of ANY frequent itemset X recovered from the CFIs alone (lossless property):
    sup(X) = max{ sup(C) : C closed and X is a subset of C }. Returns 0 if X is not frequent."""
    X = frozenset(itemset)
    return max((s for c, s in cfis.items() if X <= c), default=0)


def closed_rules(n, cfis, gens, min_conf=0.6):
    """Association rules generated from the closed frequent itemsets (CFIs) found by A-Close.

    Generic (min-max) basis, Bastide et al. 2000. The premise of every rule is a minimal
    generator g (antecedent as small as possible) and the conclusion comes from a closed set
    (consequent as large as possible), so the rule set is non-redundant; every other valid rule
    can be derived from it:
        exact rules        g => f(g) \\ g                         confidence = 100%
        approximate rules  g => C \\ g   for closed C strictly containing f(g)
                                                                  confidence = sup(C) / sup(g)
    n    : number of transactions;  cfis, gens : the first and third outputs of aclose().
    Lift needs sup(consequent); it is recovered from the CFIs (lossless property).
    """
    rows = []
    for g, c, s_g in gens:
        if not g:
            continue                       # the empty premise gives no rule
        if g != c:
            rows.append((g, c - g, s_g, 1.0))                       # exact rule
        for C, s_C in cfis.items():
            if c < C:
                conf = s_C / s_g
                if conf >= min_conf:
                    rows.append((g, C - g, s_C, conf))              # approximate rule
    out = []
    for g, cons, s_rule, conf in rows:
        s_cons = support_from_cfis(cons, cfis)
        out.append(dict(antecedent=_fmt(g), consequent=_fmt(cons),
                        support=s_rule / n, confidence=conf,
                        lift=conf / (s_cons / n) if s_cons else float("nan"),
                        type="exact" if conf == 1.0 else "approximate"))
    df = pd.DataFrame(out, columns=["antecedent", "consequent", "support", "confidence", "lift", "type"])
    return df.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# 6-7. Sequential patterns (Apriori-based, GSP / AprioriAll style)
# --------------------------------------------------------------------------------------------
def gsp(sequences, min_sup, max_len=None):
    """Level-wise (Apriori-based) sequential-pattern mining.

    sequences : list of item lists in the order the items occur (here: the aspects in the order a
                reviewer first mentions them). Each item is kept once per sequence (first mention),
                so <a, b> means "a is mentioned before b" (gaps allowed).
    Join      : s1 + last(s2) when s1 without its first item == s2 without its last item (GSP).
    Prune     : every subsequence with one item removed must be frequent (anti-monotone).
    Count     : vectorised with a position matrix P[seq, item] (-1 = absent):
                <a, b, c> is contained iff P[a] >= 0 and P[a] < P[b] < P[c].
    Returns {tuple(pattern): support_count}.
    """
    n = len(sequences)
    mc = _min_count(min_sup, n)
    items = sorted({i for s in sequences for i in s})
    col = {it: j for j, it in enumerate(items)}
    P = np.full((n, len(items)), -1, dtype=np.int32)
    for r, s in enumerate(sequences):
        for pos, it in enumerate(s):
            j = col[it]
            if P[r, j] < 0:
                P[r, j] = pos                  # first mention only

    def count(pattern):
        cols = [col[i] for i in pattern]
        mask = P[:, cols[0]] >= 0
        for a, b in zip(cols, cols[1:]):
            mask &= P[:, a] < P[:, b]          # chaining also guarantees b is present
        return int(mask.sum())

    level = {}
    for it in items:
        c = int((P[:, col[it]] >= 0).sum())
        if c >= mc:
            level[(it,)] = c
    freq = dict(level)
    k = 1
    while level and (max_len is None or k < max_len):
        by_prefix = defaultdict(list)
        for s in level:
            by_prefix[s[:-1]].append(s)
        cands = set()
        for s1 in level:
            for s2 in by_prefix.get(s1[1:], []):
                cand = s1 + (s2[-1],)
                if len(set(cand)) < len(cand):        # an item can occur once per sequence
                    continue
                if all((cand[:i] + cand[i + 1:]) in level for i in range(len(cand))):
                    cands.add(cand)
        level = {}
        for cand in cands:
            c = count(cand)
            if c >= mc:
                level[cand] = c
        freq.update(level)
        k += 1
    return freq


def sequential_rules(freq_seqs, n, min_conf=0.5):
    """Rules  <s[:j]> => <s[j:]>  from each frequent sequence s (length >= 2), read as
    "if a review mentions s[:j] in this order, it later goes on to mention s[j:]".
        support    = sup(s) / n
        confidence = sup(s) / sup(s[:j])         (s[:j] is frequent by the Apriori property)
        lift       = confidence / (sup(s[j:]) / n)
    """
    rows = []
    for s, c in freq_seqs.items():
        for j in range(1, len(s)):
            pre, post = s[:j], s[j:]
            conf = c / freq_seqs[pre]
            if conf >= min_conf:
                rows.append(dict(antecedent=" -> ".join(pre), consequent=" -> ".join(post),
                                 length=len(s), support=c / n, confidence=conf,
                                 lift=conf / (freq_seqs[post] / n)))
    df = pd.DataFrame(rows, columns=["antecedent", "consequent", "length", "support", "confidence", "lift"])
    return df.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# 8. Self-tests on standard textbook toy databases
# --------------------------------------------------------------------------------------------
def _selftest():
    import random

    # Han & Kamber's 9-transaction DB: 13 frequent itemsets at min_sup = 2
    D = [["I1", "I2", "I5"], ["I2", "I4"], ["I2", "I3"], ["I1", "I2", "I4"], ["I1", "I3"],
         ["I2", "I3"], ["I1", "I3"], ["I1", "I2", "I3", "I5"], ["I1", "I2", "I3"]]
    F = apriori_bitset(D, 2)
    assert len(F) == 13, len(F)
    C, M = fi_closed_maximal(F)
    assert {tuple(sorted(m)) for m in M} == {("I1", "I2", "I3"), ("I1", "I2", "I5"), ("I2", "I4")}
    print(f"Apriori  (Han DB, min_sup=2): {len(F)} FIs, {len(C)} CFIs, {len(M)} MFIs   [expected 13 FIs, MFIs I1I2I3 / I1I2I5 / I2I4]")

    # 5-transaction A-Close worked example: CFIs = {C:4, AC:3, BE:4, BCE:3}
    DB11 = ["ACD", "BCE", "ABCE", "BE", "ABCE"]
    cfis, table, gens = aclose([list(t) for t in DB11], 3)
    got = {"".join(sorted(k)): v for k, v in cfis.items()}
    assert got == {"C": 4, "AC": 3, "BE": 4, "BCE": 3}, got
    print("A-Close  (5-tx DB, min_sup=3):", got, "  [expected {C:4, AC:3, BE:4, BCE:3}]")
    print(table.to_string(index=False))

    # Property test: A-Close must return exactly the closed sets among Apriori's FIs
    rng = random.Random(7)
    for trial in range(30):
        db = [[x for x in "ABCDEFGH" if rng.random() < 0.45] for _ in range(rng.randint(20, 120))]
        ms = rng.choice([2, 3, 5, 0.1, 0.2])
        closed_ref, _ = fi_closed_maximal(apriori_bitset(db, ms))
        assert aclose(db, ms)[0] == closed_ref, trial
    print("A-Close == closed(Apriori) on 30 random databases: OK")

    # Closed rules from these CFIs
    print(closed_rules(len(DB11), cfis, gens, min_conf=0.6).to_string(index=False))

    # GSP toy example: <a,b>, <a,c>, <b,c> frequent (2 each); <a,b,c> only once
    S = [["a", "b", "c"], ["a", "c"], ["b", "c"], ["a", "b"]]
    fs = gsp(S, 2)
    expected = {("a",): 3, ("b",): 3, ("c",): 3, ("a", "b"): 2, ("a", "c"): 2, ("b", "c"): 2}
    assert fs == expected, fs
    print("GSP      (toy, min_sup=2):", fs)
    print(sequential_rules(fs, len(S), 0.5).to_string(index=False))
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _selftest()
