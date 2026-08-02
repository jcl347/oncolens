"""LLM reranking of retrieved passages, and LLM extraction of *how* a passage is used.

Two distinct jobs, deliberately kept apart because they have different failure modes and
different costs.

**1. Reranking.** First-stage retrieval (BM25 + dense fusion) is recall-oriented: it is good
at getting the right passage into the top 50 and mediocre at putting it first. A reranker
reads query and passage *together*, so it can judge things a bi-encoder structurally cannot
— whether the passage actually *measures* the thing asked about or merely mentions it. That
distinction is the entire difference between a useful oncology search and a keyword match.

Scoring is **pointwise** (each passage judged on its own 0–3 scale) rather than listwise
(rank these 20). Listwise usually scores a little better, but its output ordering depends on
the order the candidates were presented, which makes the evaluation non-deterministic in a
way that is easy to mistake for a real improvement. Pointwise scores are stable and
independently auditable, which matters more here than the last point of nDCG.

**2. Usage extraction.** The product requirement is not just "which papers" but *how the
concept is used and what impact it had*. That is a reading-comprehension task over a passage
already retrieved, so it runs on the top few results only — never on the candidate set.

Both are optional. Retrieval degrades to fusion-only if no key is configured, because an
outage in a third-party API must not take search down.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

#: Small model by default: reranking is the high-volume call, and the judgment ("does this
#: passage answer this question") is far easier than generation. Configurable because that
#: trade-off is worth measuring rather than assuming.
DEFAULT_RERANK_MODEL = "gpt-4o-mini"
DEFAULT_EXTRACT_MODEL = "gpt-4o-mini"

_RERANK_SYSTEM = """You score how well a passage from an oncology paper answers a query.

Scale:
3 - the passage directly reports the queried finding, mechanism, or result
2 - the passage is about the queried topic and contributes relevant evidence
1 - the passage mentions the topic only in passing, or as background/citation
0 - the passage does not address the query

Judge only what the passage itself states. Do not reward a passage for being from a
plausible-sounding paper, and do not penalise it for lacking context you would expect
elsewhere in the article. A methods paragraph that measures the queried quantity scores
higher than a discussion paragraph that speculates about it.

Return strict JSON: {"scores": [{"id": <int>, "score": <0-3>}, ...]} with one entry per
passage, in the order given."""

_EXTRACT_SYSTEM = """You explain how a specific concept is used in a passage from an
oncology paper, for a researcher scanning results.

Return strict JSON with these keys:
  "usage"   - one sentence: what role the concept plays in THIS work (e.g. it is the
              intervention, the resistance mechanism measured, the assay used, background)
  "impact"  - one sentence: what the passage reports as the consequence or finding
              involving the concept. Use the paper's own numbers where present.
  "claim"   - the single most load-bearing sentence from the passage, quoted VERBATIM.
              It must appear character-for-character in the passage.
  "reported" - true if the passage states an actual finding about the concept, false if it
              only mentions it.

If the passage does not really discuss the concept, set reported=false and say so plainly
in "usage" rather than inventing a connection. Never state a number that is not in the
passage."""


def _client():
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    return OpenAI()


@dataclass(frozen=True)
class RerankedPassage:
    index: int
    score: float
    llm_score: int | None


def rerank(query: str, passages: list[str], *, model: str = DEFAULT_RERANK_MODEL,
           max_passages: int = 24, max_chars: int = 1200,
           blend: float = 0.7) -> list[RerankedPassage]:
    """Rescore ``passages`` for ``query``. Returns entries sorted best-first.

    ``blend`` mixes the LLM score with the first-stage rank rather than discarding the
    latter. A pure-LLM ordering throws away the retriever's evidence entirely and, because
    the LLM emits only four distinct values, leaves large ties that then resolve
    arbitrarily. Keeping a fraction of the original rank breaks those ties with the
    retriever's own confidence, which is information, not noise.

    Falls back to the identity ordering when no API key is configured — search must not
    depend on a third-party service being reachable.
    """
    client = _client()
    n = min(len(passages), max_passages)
    identity = [RerankedPassage(i, 1.0 / (i + 1), None) for i in range(len(passages))]
    if client is None or n == 0:
        return identity

    numbered = [{"id": i, "passage": passages[i][:max_chars]} for i in range(n)]
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": json.dumps(
                    {"query": query, "passages": numbered}, ensure_ascii=False)},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:  # noqa: BLE001
        return identity

    llm: dict[int, int] = {}
    for row in data.get("scores", []):
        try:
            idx, sc = int(row["id"]), int(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < n:
            llm[idx] = max(0, min(3, sc))

    out: list[RerankedPassage] = []
    for i in range(len(passages)):
        prior = 1.0 / (i + 1)
        if i in llm:
            score = blend * (llm[i] / 3.0) + (1 - blend) * prior
        else:
            # Not judged (beyond max_passages, or the model omitted it): keep it strictly
            # below everything judged relevant, but preserve its first-stage order.
            score = (1 - blend) * prior
        out.append(RerankedPassage(i, score, llm.get(i)))
    out.sort(key=lambda r: (-r.score, r.index))
    return out


@dataclass(frozen=True)
class Usage:
    usage: str
    impact: str
    claim: str
    reported: bool
    verified_span: tuple[int, int] | None

    def as_dict(self) -> dict:
        return {"usage": self.usage, "impact": self.impact, "claim": self.claim,
                "reported": self.reported,
                "span": list(self.verified_span) if self.verified_span else None}


def explain_usage(concept: str, passage: str, *,
                  model: str = DEFAULT_EXTRACT_MODEL) -> Usage | None:
    """What role does ``concept`` play in ``passage``, and what did it yield?

    The returned ``claim`` is **verified to occur verbatim** in the passage, and its
    character offsets are returned so the UI can highlight it. A quote the model
    paraphrased is discarded rather than displayed: an interface that highlights text the
    user cannot find in the source destroys exactly the trust that showing sources is meant
    to build.
    """
    client = _client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": json.dumps(
                    {"concept": concept, "passage": passage[:4000]}, ensure_ascii=False)},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:  # noqa: BLE001
        return None

    claim = (data.get("claim") or "").strip()
    span = None
    if claim:
        pos = passage.find(claim)
        if pos < 0:
            # Tolerate whitespace-only differences before giving up on the quote.
            import re
            flat = re.sub(r"\s+", " ", passage)
            fclaim = re.sub(r"\s+", " ", claim)
            fpos = flat.find(fclaim)
            pos = -1 if fpos < 0 else passage.find(fclaim.split(" ")[0], 0)
        if pos >= 0:
            span = (pos, pos + len(claim))
        else:
            claim = ""          # unverifiable quote is dropped, not shown
    return Usage(
        usage=(data.get("usage") or "").strip(),
        impact=(data.get("impact") or "").strip(),
        claim=claim,
        reported=bool(data.get("reported")),
        verified_span=span,
    )
