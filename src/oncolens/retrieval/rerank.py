"""Feature-based reranking of the fused candidate list.

A cross-encoder is the standard second stage and is unavailable offline, so this
implements the classic feature signals a cross-encoder learns to approximate — the ones
neither BM25 nor a bag-of-words dense model can express:

* **coverage** — what fraction of the *distinct* query concepts appear at all. BM25 will
  happily rank a passage that repeats one query term twenty times above a passage that
  mentions all five terms once; coverage inverts that.
* **proximity** — the tightest window containing the matched terms. "EGFR" and "resistance"
  in one clause is a different claim from the two words 400 characters apart.
* **exact phrase** — a contiguous match of a multi-word query, which bag-of-words scoring
  cannot represent at all.
* **section prior** — an Abstract or Specific Aims passage states a document's thesis;
  a Methods passage mentioning the same term usually does not. This is a genuine
  domain prior for grants and papers specifically.

Weights are configuration, not constants, so the loop measures whether each feature earns
its place instead of assuming the priors are correct.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .chunking import Chunk
from .text import is_rare_literal, tokenize

#: Sections that typically carry a document's central claim.
THESIS_SECTIONS = frozenset({"abstract", "specific aims", "discussion", "results"})


@dataclass(frozen=True)
class RerankWeights:
    coverage: float = 1.0
    proximity: float = 0.5
    phrase: float = 1.0
    literal: float = 1.0
    section: float = 0.25
    base: float = 1.0          # weight on the incoming fused score (rank-normalised)


def _coverage(q_terms: Sequence[str], d_terms: set[str]) -> float:
    if not q_terms:
        return 0.0
    distinct = set(q_terms)
    return sum(1 for t in distinct if t in d_terms) / len(distinct)


def _proximity(q_terms: Sequence[str], doc_tokens: Sequence[str]) -> float:
    """1.0 when all matched query terms sit in a tight window; decays as the span grows."""
    wanted = set(q_terms)
    positions: dict[str, list[int]] = {}
    for i, tok in enumerate(doc_tokens):
        if tok in wanted:
            positions.setdefault(tok, []).append(i)
    if len(positions) < 2:
        return 0.0
    # Smallest window containing one occurrence of each matched term (greedy sweep).
    hits = sorted((p, t) for t, ps in positions.items() for p in ps)
    need = len(positions)
    best = None
    counts: dict[str, int] = {}
    left = 0
    for right in range(len(hits)):
        counts[hits[right][1]] = counts.get(hits[right][1], 0) + 1
        while len(counts) == need:
            span = hits[right][0] - hits[left][0] + 1
            best = span if best is None else min(best, span)
            t = hits[left][1]
            counts[t] -= 1
            if counts[t] == 0:
                del counts[t]
            left += 1
    if best is None:
        return 0.0
    ideal = need
    return ideal / max(best, ideal)


def _phrase(query: str, text: str) -> float:
    """Longest contiguous query n-gram present verbatim, normalised by query length."""
    q = tokenize(query)
    if len(q) < 2:
        return 0.0
    low = text.lower()
    best = 0
    for n in range(len(q), 1, -1):
        for i in range(len(q) - n + 1):
            if " ".join(q[i : i + n]) in low:
                best = n
                break
        if best:
            break
    return best / len(q)


def _literal_hits(q_terms: Sequence[str], d_terms: set[str]) -> float:
    """Fraction of identifier-shaped query tokens actually present.

    Weighted separately from coverage because missing 'C797S' is a categorically worse
    failure than missing 'resistance'.
    """
    lits = [t for t in set(q_terms) if is_rare_literal(t)]
    if not lits:
        return 0.0
    return sum(1 for t in lits if t in d_terms) / len(lits)


def rerank(
    query: str,
    candidates: Sequence[tuple[str, float]],
    chunk_by_id: Mapping[str, Chunk],
    *,
    weights: RerankWeights | None = None,
    depth: int = 100,
) -> list[tuple[str, float]]:
    """Rescore the top ``depth`` candidates; leave the tail in its original order.

    Only the head is rescored because reranking is the expensive stage and the tail rarely
    reaches a user. ``depth`` is exposed so the loop can measure the recall/cost trade
    rather than assume a value.
    """
    w = weights or RerankWeights()
    q_terms = tokenize(query)
    head = list(candidates[:depth])
    tail = list(candidates[depth:])
    if not head:
        return list(candidates)

    n = len(head)
    out: list[tuple[str, float]] = []
    for rank, (unit_id, _score) in enumerate(head):
        chunk = chunk_by_id.get(unit_id)
        if chunk is None:
            out.append((unit_id, (n - rank) / n))
            continue
        text = chunk.indexable_text(include_heading=True)
        d_tokens = tokenize(text)
        d_terms = set(d_tokens)
        s = (
            w.base * ((n - rank) / n)
            + w.coverage * _coverage(q_terms, d_terms)
            + w.proximity * _proximity(q_terms, d_tokens)
            + w.phrase * _phrase(query, text)
            + w.literal * _literal_hits(q_terms, d_terms)
            + w.section * (1.0 if chunk.section.strip().lower() in THESIS_SECTIONS else 0.0)
        )
        out.append((unit_id, s))

    out.sort(key=lambda x: (-x[1], x[0]))
    # Tail keeps its relative order, pushed below every reranked item.
    floor = out[-1][1] if out else 0.0
    out.extend((d, floor - 1e-6 * (i + 1)) for i, (d, _) in enumerate(tail))
    return out
