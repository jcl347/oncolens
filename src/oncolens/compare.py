"""Comparative retrieval: finding *technical comparisons between papers*.

Standard RAG retrieval is the wrong shape for this goal, and the failure is not subtle.

Ask "how do these studies measure ctDNA clearance?" and a top-k relevance ranking happily
returns ten passages drawn from two papers, several of which discuss ctDNA without ever
stating a *method*. You cannot build a comparison from that: a comparison needs **the same
technical dimension, across different papers**, and top-k optimises for neither.

Three changes make it work:

1. **Aspect conditioning.** Retrieve passages that actually discuss the dimension being
   compared (cohort, assay, endpoint, effect size, model system…), not passages that merely
   mention the topic. A passage that says "ctDNA was informative" is on-topic and useless
   for a methods comparison.

2. **Document diversification.** Cap passages per paper and maximise distinct papers.
   Relevance ranking is actively hostile here: the most relevant passages cluster inside
   whichever paper is most on-topic, which is the opposite of what a comparison needs.
   Implemented as MMR with a hard per-document cap.

3. **Comparability gating.** Two papers are only comparable on an aspect if both actually
   report it. Silently comparing a paper that reports a hazard ratio against one that
   reports nothing produces a table with a confident empty cell, which reads as "no effect"
   rather than "not measured". Those are opposite claims, so the distinction is explicit.

Every cell in a resulting comparison carries the passage and offsets it came from, so a
claim can always be traced back to source text rather than to a model's paraphrase.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .retrieval.pipeline import Evidence, Retriever
from .retrieval.text import tokenize


# ---------------------------------------------------------------------------
# Technical aspects — the dimensions along which oncology papers get compared
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Aspect:
    key: str
    label: str
    question: str            # used to steer retrieval toward this dimension
    cues: tuple[str, ...]    # surface markers that a passage reports this aspect
    numeric: bool = False    # does a well-formed answer contain a number?


ASPECTS: tuple[Aspect, ...] = (
    Aspect(
        "cohort", "Cohort / sample",
        "how many patients or samples were studied, and what population",
        ("patients", "participants", "cohort", "enrolled", "n =", "n=", "samples",
         "subjects", "cases", "consecutive", "median age", "eligible"),
        numeric=True,
    ),
    Aspect(
        "assay", "Assay / platform",
        "what assay, sequencing platform, or measurement technology was used",
        ("sequencing", "ngs", "ddpcr", "digital pcr", "qpcr", "rna-seq", "wes", "wgs",
         "immunohistochemistry", "ihc", "flow cytometry", "mass spectrometry", "elisa",
         "panel", "assay", "platform", "illumina", "single-cell", "spatial"),
    ),
    Aspect(
        "endpoint", "Endpoint",
        "what clinical or biological endpoint was measured",
        ("progression-free survival", "overall survival", "pfs", "os", "response rate",
         "orr", "objective response", "duration of response", "endpoint", "primary outcome",
         "disease-free", "event-free", "time to", "clearance"),
    ),
    Aspect(
        "effect", "Effect size / statistics",
        "what effect size, hazard ratio, or statistical result was reported",
        ("hazard ratio", "hr ", "odds ratio", "95% ci", "confidence interval", "p =",
         "p<", "p =", "median pfs", "median os", "ic50", "fold change", "significant",
         "log-rank", "auc", "sensitivity", "specificity"),
        numeric=True,
    ),
    Aspect(
        "model", "Model system",
        "what experimental model was used",
        ("cell line", "xenograft", "pdx", "patient-derived", "organoid", "mouse", "mice",
         "in vitro", "in vivo", "knockout", "crispr", "transgenic", "syngeneic"),
    ),
    Aspect(
        "intervention", "Intervention / agent",
        "what drug, dose, or intervention was administered",
        ("mg", "dose", "administered", "treated with", "monotherapy", "combination",
         "regimen", "cycle", "intravenous", "oral", "twice daily", "once daily"),
    ),
    Aspect(
        "resistance", "Resistance mechanism",
        "what mechanism of resistance or escape was identified",
        ("resistance", "resistant", "escape", "bypass", "acquired", "refractory",
         "relapse", "progression on", "mutation", "amplification", "loss of"),
    ),
    Aspect(
        "limitation", "Limitations",
        "what limitations or caveats did the authors state",
        ("limitation", "caveat", "retrospective", "single-center", "small sample",
         "not powered", "confounding", "selection bias", "should be interpreted"),
    ),
)

ASPECTS_BY_KEY = {a.key: a for a in ASPECTS}

_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def aspect_score(text: str, aspect: Aspect) -> float:
    """How strongly a passage *reports* this aspect (not merely mentions the topic).

    Cue density, not a binary flag: a Methods paragraph naming three platforms is a far
    better assay-comparison source than a Discussion sentence that says "sequencing".
    Numeric aspects additionally require a number, because "the cohort was large" cannot
    populate a comparison cell.
    """
    low = text.lower()
    hits = sum(1 for cue in aspect.cues if cue in low)
    if hits == 0:
        return 0.0
    score = math.log1p(hits) / math.log1p(len(aspect.cues))
    if aspect.numeric:
        n_numbers = len(_NUM.findall(text))
        if n_numbers == 0:
            return 0.0                      # unquantified: useless for this aspect
        score *= min(1.0, 0.4 + 0.15 * n_numbers)
    return min(1.0, score)


def detect_aspects(text: str, *, threshold: float = 0.25) -> dict[str, float]:
    return {a.key: s for a in ASPECTS if (s := aspect_score(text, a)) >= threshold}


def infer_aspect(query: str) -> Aspect | None:
    """Guess which dimension a natural-language question is asking to compare."""
    q = query.lower()
    best, best_score = None, 0.0
    for a in ASPECTS:
        s = sum(1 for cue in a.cues if cue in q) + sum(
            1 for w in tokenize(a.question) if w in q
        ) * 0.5
        if s > best_score:
            best, best_score = a, s
    return best if best_score >= 1.0 else None


# ---------------------------------------------------------------------------
# Diversified, aspect-conditioned retrieval
# ---------------------------------------------------------------------------

@dataclass
class ComparisonCell:
    doc_id: str
    aspect: str
    passage: str
    section: str
    start_char: int
    end_char: int
    chunk_id: str
    aspect_strength: float
    reported: bool = True          # False = this paper does not report the aspect

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "aspect": self.aspect, "reported": self.reported,
            "passage": self.passage, "section": self.section,
            "start_char": self.start_char, "end_char": self.end_char,
            "chunk_id": self.chunk_id, "aspect_strength": round(self.aspect_strength, 4),
        }


@dataclass
class ComparisonTable:
    query: str
    aspects: list[str]
    doc_ids: list[str]
    cells: dict[str, dict[str, ComparisonCell]] = field(default_factory=dict)  # doc -> aspect -> cell
    notes: list[str] = field(default_factory=list)

    def coverage(self) -> float:
        """Fraction of (paper, aspect) pairs actually supported by a passage.

        Reported because a sparse table is a weak comparison, and the caller should know
        that before presenting it as though every cell were evidence.
        """
        total = len(self.doc_ids) * len(self.aspects)
        if total == 0:
            return 0.0
        filled = sum(1 for d in self.cells.values() for c in d.values() if c.reported)
        return filled / total

    def as_dict(self) -> dict:
        return {
            "query": self.query, "aspects": self.aspects, "doc_ids": self.doc_ids,
            "coverage": round(self.coverage(), 4), "notes": self.notes,
            "cells": {d: {a: c.as_dict() for a, c in row.items()} for d, row in self.cells.items()},
        }


def _mmr(
    candidates: Sequence[tuple[str, float, str]],   # (chunk_id, score, doc_id)
    *, k: int, per_doc: int, lambda_: float = 0.5,
) -> list[tuple[str, float, str]]:
    """Maximal Marginal Relevance with a hard per-document cap.

    The per-document cap is what actually produces a comparison. Pure relevance ranking
    concentrates inside the single most on-topic paper; capping forces breadth across
    papers, which is the whole point. ``lambda_`` trades relevance against novelty.
    """
    selected: list[tuple[str, float, str]] = []
    doc_counts: dict[str, int] = {}
    pool = list(candidates)
    while pool and len(selected) < k:
        best_i, best_val = None, -1e9
        for i, (cid, score, doc) in enumerate(pool):
            if doc_counts.get(doc, 0) >= per_doc:
                continue
            # Novelty = penalise documents already represented.
            penalty = doc_counts.get(doc, 0) / max(per_doc, 1)
            val = lambda_ * score - (1 - lambda_) * penalty
            if val > best_val:
                best_i, best_val = i, val
        if best_i is None:
            break
        chosen = pool.pop(best_i)
        selected.append(chosen)
        doc_counts[chosen[2]] = doc_counts.get(chosen[2], 0) + 1
    return selected


def comparative_search(
    retriever: Retriever,
    query: str,
    *,
    aspects: Sequence[str] | None = None,
    n_papers: int = 6,
    per_doc: int = 2,
    candidate_depth: int = 400,
) -> ComparisonTable:
    """Retrieve a paper x aspect grid for a comparison question.

    Unlike ordinary search this does not return "the best passages" — it returns the
    passages that let distinct papers be set side by side on the same dimensions.
    """
    chosen = [ASPECTS_BY_KEY[a] for a in aspects if a in ASPECTS_BY_KEY] if aspects else None
    if not chosen:
        inferred = infer_aspect(query)
        chosen = [inferred] if inferred else list(ASPECTS[:4])

    table = ComparisonTable(query=query, aspects=[a.key for a in chosen], doc_ids=[])

    # Stage 1 — topical candidate pool, deliberately deep. Comparison needs breadth of
    # documents, and the diversification step can only work with candidates to choose from.
    saved_k = retriever.config.top_k
    try:
        object.__setattr__(retriever.config, "top_k", candidate_depth)
        base = retriever.search(query)
    finally:
        object.__setattr__(retriever.config, "top_k", saved_k)

    if not base.ranking:
        table.notes.append("no documents matched the query")
        return table

    # Stage 2 — score every candidate passage per aspect, then diversify per aspect.
    chunk_by_id = retriever.chunk_by_id
    doc_of = retriever.unit_to_doc
    ranked_docs = set(base.ranking[: max(n_papers * 4, 24)])

    per_aspect_selected: dict[str, list[tuple[str, float, str]]] = {}
    for aspect in chosen:
        cands: list[tuple[str, float, str]] = []
        for cid, chunk in chunk_by_id.items():
            doc = doc_of.get(cid)
            if doc not in ranked_docs:
                continue
            s = aspect_score(chunk.text, aspect)
            if s > 0:
                cands.append((cid, s, doc))
        cands.sort(key=lambda x: -x[1])
        per_aspect_selected[aspect.key] = _mmr(
            cands, k=n_papers * per_doc, per_doc=per_doc
        )

    # Stage 3 — choose the papers that are comparable across the most aspects. A paper
    # reporting one of four dimensions makes a mostly-empty row and weakens the table.
    doc_aspect_count: dict[str, int] = {}
    for sel in per_aspect_selected.values():
        for _cid, _s, doc in sel:
            doc_aspect_count[doc] = doc_aspect_count.get(doc, 0) + 1
    docs = [d for d, _ in sorted(doc_aspect_count.items(), key=lambda x: (-x[1], x[0]))][:n_papers]
    table.doc_ids = docs

    # Stage 4 — fill the grid; mark genuinely-absent aspects rather than leaving a blank.
    for doc in docs:
        row: dict[str, ComparisonCell] = {}
        for aspect in chosen:
            best = None
            for cid, s, d in per_aspect_selected[aspect.key]:
                if d == doc and (best is None or s > best[1]):
                    best = (cid, s)
            if best:
                c = chunk_by_id[best[0]]
                row[aspect.key] = ComparisonCell(
                    doc_id=doc, aspect=aspect.key, passage=c.text, section=c.section,
                    start_char=c.start_char, end_char=c.end_char, chunk_id=c.chunk_id,
                    aspect_strength=best[1],
                )
            else:
                # Explicit "not reported". Distinct from a null result — conflating the
                # two turns a missing measurement into a claim of no effect.
                row[aspect.key] = ComparisonCell(
                    doc_id=doc, aspect=aspect.key, passage="", section="",
                    start_char=0, end_char=0, chunk_id="", aspect_strength=0.0,
                    reported=False,
                )
        table.cells[doc] = row

    cov = table.coverage()
    if cov < 0.5:
        table.notes.append(
            f"sparse comparison: only {cov:.0%} of paper x aspect cells are supported by a "
            f"passage. Treat empty cells as 'not reported', never as 'no effect'."
        )
    return table


def contrast_prompt(table: ComparisonTable, doc_titles: Mapping[str, str]) -> str:
    """Build a grounded prompt for the synthesis step.

    Deliberately passes the *passages*, not a summary of them, and instructs the model to
    refuse to compare on aspects a paper does not report — which is the failure mode that
    makes automated literature comparison untrustworthy.
    """
    lines = [
        "Compare the following papers on the technical dimensions shown.",
        "",
        "Rules:",
        "- Ground every statement in the quoted passages. Do not use outside knowledge.",
        "- Where a cell is marked NOT REPORTED, say the paper does not report it. Never",
        "  infer a value, and never treat a missing measurement as a null result.",
        "- Prefer exact numbers, units and assay names over paraphrase.",
        "- Call out where papers are not directly comparable (different endpoints,",
        "  populations, or measurement methods) rather than forcing a comparison.",
        "",
        f"Question: {table.query}",
        "",
    ]
    for doc in table.doc_ids:
        lines.append(f"### {doc} — {doc_titles.get(doc, '')}")
        for aspect_key in table.aspects:
            cell = table.cells[doc][aspect_key]
            label = ASPECTS_BY_KEY[aspect_key].label
            if cell.reported:
                lines.append(f"- **{label}** [{cell.section} {cell.start_char}-{cell.end_char}]: "
                             f"{cell.passage[:600]}")
            else:
                lines.append(f"- **{label}**: NOT REPORTED")
        lines.append("")
    return "\n".join(lines)
