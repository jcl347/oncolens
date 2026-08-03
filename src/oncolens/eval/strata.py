"""Query **strata** — because one benchmark measures one kind of question.

**The gap this exists to close.** The citation-context benchmark has 2,225 expert-authored
queries and is the best label source available here. It also has a shape problem that was
invisible until measured:

===============================  ==========
citation-context query length    words
===============================  ==========
minimum                          8
median                           27
maximum                          59
under 8 words                    **0 (0.0%)**
===============================  ==========

**Nobody types a 27-word query.** Biomedical search logs are 2–5 terms: ``EGFR C797S``,
``osimertinib resistance``, ``CAR-T exhaustion solid tumor``. So a system tuned on that
benchmark is tuned for a query distribution its users never produce — and the direction of
the error is predictable, because long queries carry enough context for semantic matching
to shine while short ones lean on exact lexical match. A conclusion like "the dense arm is
worth +0.088" may simply not transfer.

Three strata, measuring three genuinely different questions:

===============  =========================  ==================  ==============================
stratum          query shape                judge               what it tests
===============  =========================  ==================  ==============================
``claim``        27-word sentence           the citing author   claim-level specificity
``concept``      2–4 word MeSH term         NLM indexers        topical recall, real query shape
``identifier``   1–3 token gene/variant     derived, exact      precision on rare literals
===============  =========================  ==================  ==============================

**A change must not be promoted on an aggregate mean across strata.** The failure this
prevents is concrete and was anticipated in ``docs/MEASUREMENT.md``: an aggregate rises
while exact-identifier lookup collapses, because identifier queries are a small share of
the pool and semantic smoothing helps everything else. Per-stratum gating makes that
trade visible instead of averaging it away.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: MeSH descriptors so generic that every document carries them. A query for "Humans"
#: is answered identically by every system and measures nothing.
GENERIC_DESCRIPTORS = frozenset({
    "Humans", "Animals", "Male", "Female", "Adult", "Aged", "Middle Aged",
    "Young Adult", "Adolescent", "Child", "Aged, 80 and over", "Mice",
    "Retrospective Studies", "Prospective Studies", "Treatment Outcome",
    "Follow-Up Studies", "Time Factors", "Reproducibility of Results",
    "Cell Line, Tumor", "Cell Line", "Animals, Newborn", "Rats",
})

#: A concept query must have at least this many relevant documents. Below it the metric
#: is dominated by which single paper happened to be indexed with the term.
MIN_DOCS_PER_CONCEPT = 3
#: And at most this many, or the descriptor is a topic label rather than a query.
MAX_DOCS_PER_CONCEPT = 60

#: Gene symbols, protein variants, and trial identifiers — the literals dense retrieval
#: is worst at and users most expect to work exactly.
_IDENTIFIER = re.compile(
    r"\b("
    r"[A-Z][A-Z0-9]{1,6}\s+[A-Z]\d{2,4}[A-Z]"      # EGFR C797S, KRAS G12C
    r"|[A-Z][A-Z0-9]{2,6}-[A-Za-z]?\d{1,3}"         # PD-L1, HER-2, IL-6
    r"|NCT\d{8}"                                     # trial registration
    r"|rs\d{4,}"                                     # dbSNP
    r")\b"
)


@dataclass
class StratifiedQuery:
    query_id: str
    query: str
    stratum: str
    judgments: dict[str, int] = field(default_factory=dict)
    exclude_doc: str | None = None
    note: str = ""


def concept_queries(
    doc_descriptors: dict[str, list[tuple[str, bool]]],
    *,
    min_docs: int = MIN_DOCS_PER_CONCEPT,
    max_docs: int = MAX_DOCS_PER_CONCEPT,
) -> list[StratifiedQuery]:
    """Short topical queries from NLM's human MeSH indexing.

    ``doc_descriptors`` maps doc_id -> [(descriptor, is_major_topic), ...].

    * query    = the descriptor's preferred term, which is 2–4 words — **the shape users
      actually type**, and the reason this stratum exists
    * relevant = every document NLM indexed with it
    * grade    = 3 for a major topic, 1 otherwise, so "the paper is about this" and
      "the paper mentions this" are not collapsed

    **Not circular**, but only because of where the labels live. MeSH descriptors are
    stored on ``documents.descriptors``; the retrievable text is ``chunks.text``, which is
    the article's own prose. A descriptor matching a passage is the passage genuinely
    using the term, not the label leaking into the index. If descriptors were ever
    concatenated into the indexed text this stratum would become meaningless overnight,
    which is why ``assert_no_descriptor_leakage`` exists below.
    """
    by_desc: dict[str, dict[str, int]] = {}
    for doc_id, descs in doc_descriptors.items():
        for name, is_major in descs:
            clean = name.replace("MESH:", "").strip()
            if not clean or clean in GENERIC_DESCRIPTORS:
                continue
            by_desc.setdefault(clean, {})[doc_id] = 3 if is_major else 1

    out: list[StratifiedQuery] = []
    for desc, judgments in sorted(by_desc.items()):
        if not (min_docs <= len(judgments) <= max_docs):
            continue
        # A concept nobody flagged as a major topic is incidental vocabulary, not a topic
        # anyone would search for.
        if not any(g == 3 for g in judgments.values()):
            continue
        out.append(StratifiedQuery(
            query_id=f"concept:{desc}",
            query=desc,
            stratum="concept",
            judgments=judgments,
            note=f"{sum(1 for g in judgments.values() if g == 3)} major / {len(judgments)} total",
        ))
    return out


def identifier_queries(claim_queries: list[StratifiedQuery]) -> list[StratifiedQuery]:
    """Exact-literal lookups mined from the claim stratum.

    Takes the identifier out of a sentence that cites a paper *for* that identifier, and
    keeps the same target. The query becomes ``EGFR C797S`` instead of the 27-word
    sentence around it — which is what a user types, and the case where a dense model is
    most likely to smooth a rare literal into something merely similar.

    The judgment is inherited rather than invented: the citing author asserted the cited
    paper is about this claim, and the identifier is the claim's subject.
    """
    out: list[StratifiedQuery] = []
    seen: set[str] = set()
    for q in claim_queries:
        for m in _IDENTIFIER.finditer(q.query):
            token = m.group(1)
            key = f"{token.lower()}|{sorted(q.judgments)[0] if q.judgments else ''}"
            if key in seen:
                continue
            seen.add(key)
            out.append(StratifiedQuery(
                query_id=f"ident:{token}:{q.query_id.split(':')[-1]}",
                query=token,
                stratum="identifier",
                judgments=dict(q.judgments),
                exclude_doc=q.exclude_doc,
                note=f"extracted from: {q.query[:70]}",
            ))
    return out


#: Longest surface form the miner will consider, in whitespace tokens. NCIt contains very
#: long descriptive concept names ("Recurrent Adult Diffuse Large B-Cell Lymphoma") which
#: are real concepts but are not what anyone types as a literal lookup.
MAX_MINE_TOKENS = 3

#: Shortest literal worth mining, after cleaning. Same reasoning as
#: ``terminology.MIN_TERM_CHARS``: below this a token cannot identify anything.
MIN_MINE_CHARS = 3

#: Ordinary English that NCIt also holds as concept names. Without this the miner turns
#: every claim sentence into a dozen spurious identifier queries carrying real judgments,
#: which would inflate the stratum with noise and look like progress on the power problem.
#: This is a stop list against dictionary FALSE POSITIVES, not a hand-built vocabulary of
#: what to search for — the distinction §4.4 draws about biasing a benchmark.
_MINE_STOPWORDS = frozenset({
    "and", "or", "not", "the", "for", "with", "was", "were", "are", "can", "may", "all",
    "this", "that", "these", "those", "from", "into", "than", "then", "when", "where",
    "cell", "cells", "gene", "genes", "protein", "proteins", "study", "studies", "group",
    "groups", "patient", "patients", "result", "results", "method", "methods", "data",
    "level", "levels", "effect", "effects", "type", "types", "case", "cases", "control",
    "controls", "human", "mouse", "mice", "rat", "tumor", "tumour", "cancer", "disease",
    "one", "two", "three", "high", "low", "new", "old", "same", "other", "such", "both",
    "each", "more", "most", "less", "only", "also", "well", "very", "much", "many",
    "some", "any", "fig", "figure", "table", "ref", "via", "per", "using", "used", "show",
    "shown", "shows", "increase", "decrease", "expression", "activity", "response",
    "treatment", "therapy", "model", "models", "assay", "line", "lines", "mrna", "dna",
    "rna", "vivo", "vitro", "sensitivity", "specificity", "survival", "growth", "risk",
})


def gazetteer_identifier_queries(
    claim_queries: list["StratifiedQuery"],
    gazetteer: dict[str, dict],
    *,
    max_per_target: int = 4,
) -> list["StratifiedQuery"]:
    """Mine identifier queries with a **dictionary** instead of a pattern.

    ``_IDENTIFIER`` recognises four character-class shapes. It structurally cannot match a
    bare gene symbol (``BRCA1``, ``TP53``) or any drug name (``osimertinib``,
    ``pembrolizumab``), because those look like ordinary words — and looking like an
    ordinary word is not something a regex can distinguish from being one. That limit, not
    the corpus, is why the identifier stratum was 335 queries while the eval needed
    thousands to resolve a 0.02 effect (§4.14, §4.8).

    A gazetteer built from NCIt knows ``osimertinib`` is a Pharmacologic Substance and
    ``patients`` is not, so recognition and typing are one lookup. Matching is
    **leftmost-longest** over a normalised key, so ``EGFR T790M`` is taken whole rather than
    split into ``EGFR`` — the failure that used to *broaden* a resistance-mutation query
    into every EGFR paper.

    Judgments are inherited from the claim the literal was mined out of, exactly as
    ``identifier_queries`` does: the citing author asserted the cited paper supports this
    sentence, and the literal is the sentence's subject. Nothing new is judged here.

    ``max_per_target`` mirrors §4.4's ``MAX_PER_TARGET`` — without it a landmark paper
    cited 90 times contributes 90 near-identical queries and the stratum measures that one
    paper.

    ⚠️ **Use with** ``identifier_queries``, **not instead of it.** Measured: the gazetteer
    finds 1,849 literals the regex cannot, and the regex finds **138** the gazetteer does
    not — trial acronyms (``BOLERO-2``, ``BATTLE-2``) that are not NCIt concepts at all,
    and hyphenated forms whose sense fan-out exceeds ``MAX_MINE_SENSES``. Neither is
    sufficient alone, which is the same conclusion §4.14 reached about the registries
    themselves. ``build_strata`` unions them.
    """
    out: list[StratifiedQuery] = []
    seen: set[str] = set()
    per_target: dict[str, int] = {}

    for q in claim_queries:
        target = sorted(q.judgments)[0] if q.judgments else ""
        if per_target.get(target, 0) >= max_per_target:
            continue
        for token, entry in _mine_spans(q.query, gazetteer):
            key = f"{token.casefold()}|{target}"
            if key in seen:
                continue
            seen.add(key)
            per_target[target] = per_target.get(target, 0) + 1
            out.append(StratifiedQuery(
                query_id=f"ident:{token.replace(' ', '_')}:{q.query_id.split(':')[-1]}",
                query=token,
                stratum="identifier",
                judgments=dict(q.judgments),
                exclude_doc=q.exclude_doc,
                note=f"{entry['sem'][0] if entry.get('sem') else 'concept'} "
                     f"{entry.get('code', '')}; from: {q.query[:60]}",
            ))
            if per_target[target] >= max_per_target:
                break
    return out


#: A span may match this many NCIt concepts and still be mined.
#:
#: The first version required exactly one, reasoning from §4.14's ``ER`` case. That was
#: wrong, and it cost 154 real identifiers — ``BCL-2``, ``CALAA-01``, ``B16-F10``. NCIt
#: routinely holds a gene, its protein and its wild-type allele as three concepts sharing
#: one surface form; those are the *same* entity in three guises, not competing senses, and
#: expanding any of them retrieves the same papers. ``ER``'s problem was 14 senses spanning
#: an organelle, a country and an emergency room. The distinction is the size of the fan-out.
MAX_MINE_SENSES = 4

#: Semantic types where an all-lowercase single token is a genuine literal. INN drug names
#: are lowercase by convention (``osimertinib``, ``pembrolizumab``), while gene symbols and
#: cell lines carry capitals. Outside these types a bare lowercase word is far more likely
#: to be an ordinary English word that NCIt happens to contain as a concept name.
_LOWERCASE_OK = {
    "Pharmacologic Substance", "Organic Chemical", "Antibiotic",
    "Biologically Active Substance", "Nucleic Acid, Nucleoside, or Nucleotide",
}

_HTML_ENTITY = re.compile(r"&[a-z]+;|&#\d+;")
#: What survives as a query literal: letters, digits, hyphens, dots and internal spaces.
#: Anything else is sentence punctuation that leaked into the span.
_MINE_CLEAN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9\-./ ]*[A-Za-z0-9]$|^[A-Za-z0-9]{3,}$")


def _clean_span(span: str) -> str:
    """Strip the sentence away from the literal.

    Without this the miner emitted ``( fig``, ``+ cd45ra +`` and ``&gt; a`` as queries:
    ``norm_key`` folds punctuation away for *lookup*, so the raw span still matched, and
    the raw span was what got stored. Normalising for matching but not for emission is the
    bug, and it produced queries no user would type against judgments that were real.
    """
    s = _HTML_ENTITY.sub(" ", span)
    s = s.strip(" \t\"'`()[]{}<>.,;:!?*+/\\|—–-")
    return " ".join(s.split())


def _mine_spans(text: str, gazetteer: dict[str, dict]):
    """Leftmost-longest gazetteer matches over ``text``.

    Longest-first is what keeps ``EGFR T790M`` intact: a greedy left-to-right scan that
    tried one token first would consume ``EGFR`` and never see the variant.
    """
    from oncolens.terminology import norm_key

    words = text.split()
    i = 0
    while i < len(words):
        hit = None
        for n in range(min(MAX_MINE_TOKENS, len(words) - i), 0, -1):
            span = _clean_span(" ".join(words[i:i + n]))
            if len(span) < MIN_MINE_CHARS or not _MINE_CLEAN.match(span):
                continue
            if span.casefold() in _MINE_STOPWORDS:
                continue
            # A multi-token span whose parts are function words is a sentence fragment
            # that happened to normalise onto a concept: "a to" folds to "ato" and hits
            # Arsenic Trioxide. Normalisation is what makes the dictionary robust to
            # spelling; it is also what lets nonsense in, so the raw tokens are checked.
            toks = span.split()
            if len(toks) > 1 and any(len(t) < 2 or t.casefold() in _MINE_STOPWORDS
                                     for t in toks):
                continue
            entry = gazetteer.get(norm_key(span))
            if not entry or entry.get("senses", 1) > MAX_MINE_SENSES:
                continue
            sem = set(entry.get("sem") or ())
            # An all-lowercase single word is only a literal if the dictionary says it is
            # a drug. This uses the CONCEPT'S TYPE as the arbiter, with case as a prior —
            # not a pattern deciding on shape alone.
            if span.islower() and " " not in span and not (sem & _LOWERCASE_OK):
                continue
            hit = (span, entry, n)
            break
        if hit:
            yield hit[0], hit[1]
            i += hit[2]
        else:
            i += 1


def merge_duplicate_queries(queries: list["StratifiedQuery"]) -> list["StratifiedQuery"]:
    """Collapse identical query strings, unioning their judgments.

    **The defect this fixes makes a metric unwinnable.** Identifier literals are mined out
    of claim sentences and inherit that claim's single cited paper. Many sentences mention
    ``CTLA-4``, so ``CTLA-4`` appeared as **66 separate queries**, each with one *different*
    relevant document. A retrieval system is a function of the query string — it returns
    one ranking for all 66 — so at most one of them can have its judged document at rank 1.
    The other 65 score zero however good retrieval is.

    Measured ceilings before this ran:

    ========== ================== ==================
    stratum    max success@1      queries sharing a string
    ========== ================== ==================
    identifier **0.4985**         63%
    claim      1.0000             0%
    concept    1.0000             0%
    synthesis  1.0000             3%
    ========== ================== ==================

    So identifier's 0.1483 was **29.8% of what was achievable**, not 14.8%, and it was
    being quoted next to three unbounded numbers as "the worst in the system". That framing
    is what motivated two rounds of query expansion. It is the §6.5 error again: a number
    read in a context it does not describe.

    Merging is not a convenience, it is the correct semantics. A user typing ``CTLA-4``
    wants *any* paper substantively about CTLA-4, and all 68 judged papers qualify. This
    makes identifier consistent with ``concept``, the other short-string topical stratum,
    which already carries a median of 11 relevant documents per query.

    ⚠️ **Scores will rise when this lands, and the rise is not a retrieval improvement.**
    It is a measurement correction. Any comparison across this change is invalid; the
    baseline must be re-measured, which ``improve_loop`` does on every run anyway.
    """
    by_text: dict[str, StratifiedQuery] = {}
    for q in queries:
        key = f"{q.stratum}|{q.query.strip().casefold()}"
        first = by_text.get(key)
        if first is None:
            by_text[key] = StratifiedQuery(
                query_id=q.query_id, query=q.query, stratum=q.stratum,
                judgments=dict(q.judgments), exclude_doc=q.exclude_doc, note=q.note)
            continue
        for doc, grade in q.judgments.items():
            # Keep the strongest grade any source assigned.
            if grade > first.judgments.get(doc, 0):
                first.judgments[doc] = grade
        # `exclude_doc` guards §4.4's leakage rule: the citing paper must never be
        # returnable. Merging must not drop one, so a merged query keeps the first and
        # the rest are recorded for the caller to exclude too.
        if q.exclude_doc and q.exclude_doc != first.exclude_doc:
            extra = first.note.split("|excl:")[-1] if "|excl:" in first.note else ""
            if q.exclude_doc not in extra:
                first.note += f"|excl:{q.exclude_doc}"
    return list(by_text.values())


def assert_no_descriptor_leakage(sample_texts: list[str], descriptors: list[str]) -> None:
    """Fail if indexed text looks like it carries the MeSH labels themselves.

    Guards the **artifact, not the accessor**. An earlier version of this project scanned
    source code for reads of ``descriptors`` and passed cleanly while all 140 corpus
    documents carried the labels inline. Checking whether the code *could* leak is not the
    same as checking whether the data *does*.
    """
    if not sample_texts or not descriptors:
        return
    marker = re.compile(r"\bMESH:", re.I)
    hits = sum(1 for t in sample_texts if marker.search(t))
    if hits:
        raise AssertionError(
            f"{hits}/{len(sample_texts)} sampled passages contain a 'MESH:' marker — the "
            f"MeSH labels appear to be inside the indexed text, which makes the concept "
            f"stratum measure label lookup rather than retrieval."
        )


def summarize(queries: list[StratifiedQuery]) -> dict:
    from collections import Counter

    by_stratum = Counter(q.stratum for q in queries)
    out: dict = {"total": len(queries), "by_stratum": dict(by_stratum)}
    for s in by_stratum:
        qs = [q for q in queries if q.stratum == s]
        words = [len(q.query.split()) for q in qs]
        rel = [len(q.judgments) for q in qs]
        out[s] = {
            "queries": len(qs),
            "median_words": sorted(words)[len(words) // 2] if words else 0,
            "median_relevant_docs": sorted(rel)[len(rel) // 2] if rel else 0,
        }
    return out
