"""Clause-level match spans — *where exactly* a query matched inside a passage.

A passage is too coarse for the product requirement. A 900-character Methods paragraph
that "matches" a query gives the reader no idea which sentence earned the hit, and asking
them to re-read the paragraph defeats the purpose of retrieval.

This module narrows a hit to the **clause** that actually matched, and returns character
offsets that are absolute within the source document — so the UI can scroll to and
highlight the exact span in the original article, not a re-rendered copy.

Offsets composition, which is the part that is easy to get wrong:

    document section text
      └── chunk           (chunk.start_char .. chunk.end_char, relative to the section)
            └── clause    (clause offsets relative to the chunk)
                  └── term span (relative to the clause)

Everything returned here is resolved back to **section-absolute** offsets, because that is
what a viewer needs. Keeping them chunk-relative is a bug that only shows up as
mysteriously-off highlights once the UI is built.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from collections.abc import Sequence

from .retrieval.text import is_rare_literal, tokenize

# Clause boundaries: sentence enders, plus semicolons and colons, which in scientific
# prose usually separate independently-quotable claims.
_CLAUSE = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(\[])|(?<=[.;:!?])\s*$")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/+]*")


@dataclass(frozen=True)
class TermSpan:
    start: int          # section-absolute
    end: int
    text: str
    is_literal: bool    # identifier-shaped (gene variant, trial id) rather than a word


@dataclass(frozen=True)
class Clause:
    text: str
    start: int          # section-absolute
    end: int
    score: float
    matched_terms: list[str]
    spans: list[TermSpan]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["spans"] = [asdict(s) for s in self.spans]
        return d


def _split_clauses(text: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    pos = 0
    for part in _CLAUSE.split(text):
        if part is None:
            continue
        stripped = part.strip()
        if not stripped:
            pos += len(part) if part else 0
            continue
        idx = text.find(stripped, pos)
        if idx < 0:
            idx = pos
        out.append((stripped, idx, idx + len(stripped)))
        pos = idx + len(stripped)
    return out or [(text, 0, len(text))]


def find_clauses(
    passage: str,
    query: str,
    *,
    base_offset: int = 0,
    max_clauses: int = 3,
    min_score: float = 0.01,
) -> list[Clause]:
    """Rank the clauses inside a passage by how well they answer the query.

    Scoring deliberately over-weights identifier-shaped terms: a clause containing
    ``EGFR C797S`` is a far better citation for that query than one containing only
    ``resistance``, even though both "match". Rare literals are the whole reason a user
    trusts a retrieval hit in this domain.
    """
    q_terms = set(tokenize(query))
    if not q_terms:
        return []
    literals = {t for t in q_terms if is_rare_literal(t)}

    scored: list[Clause] = []
    for ctext, cstart, cend in _split_clauses(passage):
        c_tokens = tokenize(ctext)
        if not c_tokens:
            continue
        matched = q_terms & set(c_tokens)
        if not matched:
            continue

        lit_hits = matched & literals
        # Coverage of the query, with a strong bonus for identifier hits, damped by
        # clause length so a long paragraph does not win purely by containing more words.
        coverage = len(matched) / len(q_terms)
        lit_bonus = 1.5 * (len(lit_hits) / max(len(literals), 1)) if literals else 0.0
        length_damp = 1.0 / (1.0 + math_log(len(c_tokens) / 25.0))
        score = (coverage + lit_bonus) * length_damp

        spans: list[TermSpan] = []
        for m in _WORD.finditer(ctext):
            w = m.group(0).lower()
            variants = set(tokenize(w))
            hit = variants & q_terms
            if hit:
                spans.append(TermSpan(
                    start=base_offset + cstart + m.start(),
                    end=base_offset + cstart + m.end(),
                    text=m.group(0),
                    is_literal=any(is_rare_literal(v) for v in hit),
                ))

        if score >= min_score:
            scored.append(Clause(
                text=ctext, start=base_offset + cstart, end=base_offset + cend,
                score=round(score, 4), matched_terms=sorted(matched), spans=spans,
            ))

    scored.sort(key=lambda c: -c.score)
    return scored[:max_clauses]


def math_log(x: float) -> float:
    import math
    return math.log(max(x, 1.0) + 1.0)


def annotate_evidence(
    query: str,
    passages: Sequence[tuple[str, str, int, int, str]],
    *,
    max_clauses: int = 2,
) -> list[dict]:
    """Attach clause-level highlights to retrieved passages.

    ``passages`` items are ``(doc_id, text, start_char, end_char, section)``, where the
    offsets are the chunk's position inside its section. Returned offsets are
    section-absolute and ready for a viewer to highlight directly.
    """
    out: list[dict] = []
    for doc_id, text, start_char, _end_char, section in passages:
        clauses = find_clauses(text, query, base_offset=start_char, max_clauses=max_clauses)
        out.append({
            "doc_id": doc_id,
            "section": section,
            "passage_start": start_char,
            "passage_text": text,
            "clauses": [c.as_dict() for c in clauses],
            "best_clause": clauses[0].as_dict() if clauses else None,
        })
    return out
