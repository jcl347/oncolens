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
