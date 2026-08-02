"""Relevance labels mined from citation contexts — found data, not annotation.

**The idea.** When an author writes

    "Acquired resistance to osimertinib is frequently driven by MET amplification [12]."

they have written a technical, passage-level, natural-language description of what
reference [12] contains — and they wrote it after reading it. That sentence is a *query*
for which the cited paper is a *relevant answer*, and the judgment came free, from a domain
expert, at the scale of the literature itself.

This matters because the alternative labels available here are weak in a specific way. MeSH
headings are **document-level and topical**: they say an article is *about* "Drug Resistance,
Neoplasm", which cannot distinguish a paper that measures a resistance mechanism from one
that mentions it in passing. Citation contexts are **claim-level and specific**, which is
what the product actually does — find the passage where a concept is used and say what it
contributed.

**Why this is not circular.** The labels come from JATS ``<xref>`` markup, which the
retrieval system never sees: we index the plain-text rendition, where citation markers have
already been flattened away. The label source and the indexed representation are disjoint.

Validity hazards, and what is done about each
---------------------------------------------

1. **The citing document contains the query verbatim.** The query sentence *is* a passage
   of the citing paper, so that paper would rank first on an exact-substring match and the
   score would measure nothing. Every query therefore records ``source_doc_id`` and the
   evaluator must exclude it. :func:`assert_source_excluded` enforces this rather than
   trusting the caller to remember.

2. **Judgments are incomplete.** Other corpus papers may be equally relevant but simply
   were not the one cited. This is the classic pooled-judgment problem, so results must be
   read with ``bpref`` and the unjudged@k diagnostic alongside nDCG — never nDCG alone.

3. **Popularity bias.** A few landmark papers are cited constantly, so an unfiltered set
   rewards a system that always returns them. Labels are capped per cited document.

4. **Diffuse attribution.** "Several studies have shown X [3,7,11,14]" does not assert that
   any *one* of those papers is a good answer for X. Grade falls with the number of works
   cited, and sentences citing more than ``MAX_COCITED`` are dropped entirely.

5. **Uninformative sentences.** "This has been reported previously [9]" describes nothing.
   Such sentences are filtered on content, not just length.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field

#: A sentence citing more works than this asserts nothing specific about any of them.
MAX_COCITED = 3
#: Cap labels per cited document so a handful of landmark papers cannot dominate.
MAX_PER_TARGET = 4
#: Cap labels contributed by any single citing paper, so one review with 300 references
#: does not become the benchmark.
MAX_PER_SOURCE = 12

MIN_QUERY_CHARS = 60
MAX_QUERY_CHARS = 320

#: Sentences that cite without describing. Matched against the whole (lowercased) sentence.
_CONTENTLESS = re.compile(
    r"^(as )?(previously |recently )?(described|reported|reviewed|shown|discussed|"
    r"published|noted|summarized)\b.{0,40}$|"
    r"^(see|reviewed in|for a review|refer to)\b|"
    r"^(this|these|it) (has|have) been\b.{0,60}$",
    re.I,
)

#: A citing sentence is only useful as a query if it makes a technical assertion. Requiring
#: a verb of finding/mechanism is a cheap proxy that removes bare pointer sentences without
#: hand-listing biomedical vocabulary (which would bias the benchmark toward terms we
#: happened to think of).
_ASSERTIVE = re.compile(
    r"\b(show|shows|showed|shown|demonstrat|report|reveal|identif|find|found|observ|"
    r"associat|correlat|induc|inhibit|activat|suppress|promot|mediat|regulat|express|"
    r"increas|decreas|reduc|enhanc|confer|drive|driven|caus|result|lead|improv|"
    r"predict|target|bind|mutat|amplif|deleti|resist|sensiti)", re.I)


@dataclass
class CitationLabel:
    """One (query, relevant document) judgment mined from a citation."""

    query_id: str
    query: str
    source_doc_id: str          #: the CITING paper — must be excluded at evaluation time
    target_doc_id: str          #: the CITED paper — the relevant answer
    grade: int                  #: 3 = sole citation, 2 = one of two, 1 = one of three
    n_cocited: int
    section: str = ""

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id, "query": self.query,
            "source_doc_id": self.source_doc_id, "target_doc_id": self.target_doc_id,
            "grade": self.grade, "n_cocited": self.n_cocited, "section": self.section,
        }


def is_useful_query(sentence: str) -> bool:
    """Would this sentence be a sensible thing for a user to search?"""
    s = sentence.strip()
    if not (MIN_QUERY_CHARS <= len(s) <= MAX_QUERY_CHARS):
        return False
    if _CONTENTLESS.search(s):
        return False
    if not _ASSERTIVE.search(s):
        return False
    # A sentence that is mostly numbers or symbols is a table fragment, not prose.
    letters = sum(c.isalpha() for c in s)
    return letters / max(len(s), 1) >= 0.6


def grade_for(n_cocited: int) -> int:
    """Confidence that *this* cited work is what the sentence is about.

    Sole attribution is a strong judgment; co-citation dilutes it. The mapping is
    deliberately coarse — the ordering is defensible, finer gradations would not be.
    """
    if n_cocited <= 1:
        return 3
    if n_cocited == 2:
        return 2
    return 1


def _qid(source_doc_id: str, sentence: str) -> str:
    h = hashlib.sha1(sentence.encode("utf-8")).hexdigest()[:10]
    return f"cite:{source_doc_id}:{h}"


def build_labels(
    per_source: dict[str, list[tuple[str, list[str], str]]],
    corpus_doc_ids: set[str],
    *,
    max_per_target: int = MAX_PER_TARGET,
    max_per_source: int = MAX_PER_SOURCE,
) -> tuple[list[CitationLabel], dict[str, int]]:
    """Turn extracted citation contexts into labels.

    ``per_source`` maps a citing ``doc_id`` to ``(sentence, [cited_doc_id, ...], section)``
    tuples. Only cited documents that are actually in ``corpus_doc_ids`` can become labels;
    the rest are counted as ``out_of_corpus``, which is the number that tells you whether
    snowball ingestion is worth running.
    """
    stats: dict[str, int] = defaultdict(int)
    per_target: dict[str, int] = defaultdict(int)
    per_src: dict[str, int] = defaultdict(int)
    seen_queries: set[str] = set()
    out: list[CitationLabel] = []

    for source_doc_id, contexts in sorted(per_source.items()):
        for sentence, cited, section in contexts:
            stats["contexts_seen"] += 1
            in_corpus = [c for c in cited if c in corpus_doc_ids]
            if not in_corpus:
                stats["out_of_corpus"] += 1
                continue
            if len(cited) > MAX_COCITED:
                stats["too_many_cocited"] += 1
                continue
            if not is_useful_query(sentence):
                stats["not_useful_query"] += 1
                continue
            # A paper citing itself gives a query whose answer is its own text.
            in_corpus = [c for c in in_corpus if c != source_doc_id]
            if not in_corpus:
                stats["self_citation"] += 1
                continue
            key = re.sub(r"\W+", " ", sentence.lower()).strip()
            if key in seen_queries:
                stats["duplicate_sentence"] += 1
                continue
            if per_src[source_doc_id] >= max_per_source:
                stats["source_cap"] += 1
                continue

            kept = [c for c in in_corpus if per_target[c] < max_per_target]
            if not kept:
                stats["target_cap"] += 1
                continue

            seen_queries.add(key)
            per_src[source_doc_id] += 1
            qid = _qid(source_doc_id, sentence)
            for target in kept:
                per_target[target] += 1
                out.append(CitationLabel(
                    query_id=qid, query=sentence, source_doc_id=source_doc_id,
                    target_doc_id=target, grade=grade_for(len(cited)),
                    n_cocited=len(cited), section=section,
                ))
            stats["labels_kept"] += 1

    stats["queries"] = len({lb.query_id for lb in out})
    stats["judgments"] = len(out)
    stats["distinct_targets"] = len({lb.target_doc_id for lb in out})
    return out, dict(stats)


@dataclass
class CitationQrels:
    """Query set + judgments, in the shape the existing evaluator consumes."""

    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)
    #: query_id -> citing doc_id that must be excluded from that query's results.
    exclude: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_labels(cls, labels: list[CitationLabel]) -> "CitationQrels":
        q = cls()
        for lb in labels:
            q.queries[lb.query_id] = lb.query
            q.qrels.setdefault(lb.query_id, {})[lb.target_doc_id] = lb.grade
            q.exclude[lb.query_id] = lb.source_doc_id
        return q

    def as_dict(self) -> dict:
        return {"queries": self.queries, "qrels": self.qrels, "exclude": self.exclude}


def assert_source_excluded(query_id: str, ranked_doc_ids: list[str],
                           exclude: dict[str, str]) -> None:
    """Fail loudly if the citing document appears in a result list.

    The citing paper contains the query sentence verbatim, so leaving it in inflates every
    metric while measuring nothing. Making this an assertion rather than a documented
    convention means the mistake cannot be made silently — which is the only kind of
    evaluation bug that actually survives.
    """
    src = exclude.get(query_id)
    if src and src in ranked_doc_ids:
        raise AssertionError(
            f"query {query_id}: citing document {src} was not excluded from its own "
            f"results. It contains the query text verbatim, so any metric computed "
            f"including it is meaningless."
        )
