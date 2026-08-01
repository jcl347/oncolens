"""Chunk contextualization (the offline stand-in for Contextual Retrieval).

Anthropic's Contextual Retrieval prepends an LLM-generated situating sentence to each chunk
before indexing, which removes the "this aim tests resistance mechanisms" problem where a
chunk is unintelligible out of context. No LLM is reachable here, so this module builds the
same *shape* of signal deterministically from the document's own text.

**Label-leakage guard — the most important thing in this file.** A corpus document carries
a ``descriptors`` field, which is the gold label. Using it to build index-time context
would leak the answer key into the retriever and produce a spectacular, entirely fake
improvement. ``build_contexts`` reads only ``title``, ``doc_type``, ``meta`` and section
text, and ``assert_no_label_leakage`` is run in CI to keep it that way.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from .retrieval.chunking import Chunk

#: Fields a retriever or context builder must never read. These are answer-key fields.
FORBIDDEN_FIELDS = ("descriptors",)

_FIRST_SENT = re.compile(r"^(.{40,320}?[.!?])(?:\s|$)", re.S)


def _gist(doc: Mapping) -> str:
    """One-sentence gist of a document, taken from its own abstract."""
    for section in doc.get("sections", []):
        if section.get("name", "").lower() in ("abstract",):
            text = (section.get("text") or "").strip()
            m = _FIRST_SENT.match(text)
            if m:
                return m.group(1).strip()
            return text[:220]
    return ""


def build_contexts(docs: Sequence[Mapping], chunks: Sequence[Chunk], *, mode: str = "structural") -> dict[str, str]:
    """chunk_id -> situating sentence.

    ``mode``:
      * ``none``        — no contextualization (control arm).
      * ``structural``  — document title + type + section role.
      * ``gist``        — structural, plus the document's own opening sentence. Closest
                          offline analogue to the LLM-generated version.
    """
    if mode == "none":
        return {}
    by_id = {d["doc_id"]: d for d in docs}
    out: dict[str, str] = {}
    for c in chunks:
        doc = by_id.get(c.doc_id)
        if doc is None:
            continue
        kind = "Grant application" if doc.get("doc_type") == "grant" else "Research paper"
        meta = doc.get("meta") or {}
        tag = meta.get("activity_code") or meta.get("journal") or ""
        head = f"{kind}{f' ({tag})' if tag else ''}: {doc.get('title','')}. Section: {c.section}."
        if mode == "gist":
            g = _gist(doc)
            out[c.chunk_id] = f"{head} {g}".strip() if g else head
        else:
            out[c.chunk_id] = head
    return out


def assert_no_label_leakage(source_files: Sequence[str]) -> list[str]:
    """Scan retrieval-path source for reads of answer-key fields.

    Returns a list of violations. A non-empty list means any measured improvement is
    suspect and the run must be discarded, not explained away.
    """
    violations: list[str] = []
    for path in source_files:
        try:
            text = open(path, "r", encoding="utf-8").read()
        except OSError:
            continue
        for field in FORBIDDEN_FIELDS:
            for m in re.finditer(rf"""\[["']{field}["']\]|\.get\(\s*["']{field}["']""", text):
                line = text[: m.start()].count("\n") + 1
                violations.append(f"{path}:{line}: retrieval path reads answer-key field '{field}'")
    return violations
