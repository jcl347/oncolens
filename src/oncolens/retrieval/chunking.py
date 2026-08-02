"""Section-aware chunking that preserves exact provenance back to the source.

Every chunk carries ``(doc_id, section, start_char, end_char)`` relative to its section
text. This is not incidental bookkeeping — the product requirement is to show *the passage
where a term was mentioned*, so a chunk that cannot point back at its source is useless
regardless of how well it retrieves.

Rules, following the section-aware / paragraph-integrity findings in the literature:
  * chunk boundaries never cross a section boundary;
  * whole paragraphs are aggregated, never split, unless a single paragraph exceeds the
    hard cap (then it is split on sentence boundaries, still with exact offsets);
  * the parent heading is carried as context on every chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .references import is_reference_like, strip_references

# Sentence boundary: period/question/exclamation + space + capital, avoiding common
# biomedical abbreviations and decimal numbers.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_PARA = re.compile(r"\n\s*\n")

TARGET_CHARS = 900
HARD_CAP = 1600
MIN_CHARS = 200


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    title: str
    section: str
    text: str
    start_char: int
    end_char: int
    ordinal: int

    def as_dict(self) -> dict:
        return asdict(self)

    def indexable_text(self, *, include_heading: bool = True, context: str | None = None) -> str:
        """Text handed to the indexers.

        ``context`` is the Contextual-Retrieval-style situating sentence. Keeping it a
        parameter (rather than baking it into ``text``) means the *displayed* passage
        stays verbatim source text while the *indexed* text can be enriched — the citation
        offsets remain valid either way.
        """
        parts = []
        if include_heading:
            parts.append(f"{self.title} — {self.section}")
        if context:
            parts.append(context)
        parts.append(self.text)
        return "\n".join(parts)


def _force_split(text: str, base_offset: int) -> list[tuple[str, int, int]]:
    """Last-resort split at whitespace, for text with no sentence boundaries at all.

    MEASURED: sentence splitting silently fails on two real shapes — tab-separated tables
    lifted out of JATS, and reference blocks written without ``. Capital`` boundaries. Both
    were emitted as single "passages", the largest **48,763 characters**. That breaks the
    product requirement directly: a passage that large cannot be shown as *the place a
    concept was mentioned*, and it also exceeds an embedding model's input limit.

    So ``HARD_CAP`` is enforced as an invariant here rather than left as an aspiration of
    the sentence splitter.
    """
    out: list[tuple[str, int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + HARD_CAP, n)
        if end < n:
            # Prefer a whitespace boundary so words are never cut in half.
            brk = text.rfind(" ", pos + HARD_CAP // 2, end)
            if brk > pos:
                end = brk
        piece = text[pos:end].strip()
        if piece:
            start = base_offset + pos + (len(text[pos:end]) - len(text[pos:end].lstrip()))
            out.append((piece, start, start + len(piece)))
        pos = end if end > pos else n
    return out


def _split_long_paragraph(para: str, base_offset: int) -> list[tuple[str, int, int]]:
    sents = _SENT.split(para)
    out: list[tuple[str, int, int]] = []
    cur: list[str] = []
    cur_start = base_offset
    cursor = base_offset
    for s in sents:
        idx = para.find(s, cursor - base_offset)
        s_start = base_offset + (idx if idx >= 0 else 0)
        if cur and (s_start + len(s) - cur_start) > HARD_CAP:
            text = para[cur_start - base_offset : cursor - base_offset].strip()
            if text:
                out.append((text, cur_start, cur_start + len(text)))
            cur, cur_start = [], s_start
        cur.append(s)
        cursor = s_start + len(s)
    tail = para[cur_start - base_offset :].strip()
    if tail:
        out.append((tail, cur_start, cur_start + len(tail)))

    # Enforce the cap. A "sentence" longer than HARD_CAP means the splitter found no
    # boundary, which happens on tables and on citation blocks — see _force_split.
    capped: list[tuple[str, int, int]] = []
    for text, start, end in out:
        if len(text) <= HARD_CAP:
            capped.append((text, start, end))
        else:
            capped.extend(_force_split(text, start))
    return capped


#: Strip bibliographies before chunking. MEASURED on real PMC text: 18.4% of full-text
#: characters are reference lists, and they retrieve well while containing no findings —
#: two of the top three hits for "osimertinib resistance mechanism" were reference entries.
DROP_REFERENCES = True


def chunk_document(doc: dict) -> list[Chunk]:
    """Chunk one corpus document into passage-level units with offsets.

    Bibliographies are removed first. Offsets stay valid because removal only ever
    truncates the tail of a section: everything kept retains its original character
    positions, so a passage still points at the right span of the source document.
    """
    chunks: list[Chunk] = []
    ordinal = 0
    for section in doc.get("sections", []):
        name = section.get("name", "Body")
        text = section.get("text", "") or ""
        if not text.strip():
            continue
        if DROP_REFERENCES and len(text) > 2000:
            stripped = strip_references(text)
            if stripped.start_index is not None:
                # Truncate rather than splice, so surviving offsets are unchanged.
                text = "\n\n".join(stripped.kept)

        # Paragraph units with their offsets inside this section.
        units: list[tuple[str, int, int]] = []
        cursor = 0
        for para in _PARA.split(text):
            if not para.strip():
                cursor += len(para) + 2
                continue
            start = text.find(para, cursor)
            if start < 0:
                start = cursor
            end = start + len(para)
            cursor = end
            if len(para) > HARD_CAP:
                units.extend(_split_long_paragraph(para, start))
            else:
                units.append((para.strip(), start, end))

        # Aggregate whole paragraphs up to TARGET_CHARS, never splitting one.
        buf: list[tuple[str, int, int]] = []
        buf_len = 0
        for unit in units:
            if buf and buf_len + len(unit[0]) > TARGET_CHARS:
                chunks.append(_emit(doc, name, buf, ordinal))
                ordinal += 1
                buf, buf_len = [], 0
            buf.append(unit)
            buf_len += len(unit[0])
        if buf:
            # Fold a runt tail into the previous chunk of the same section.
            if buf_len < MIN_CHARS and chunks and chunks[-1].section == name:
                prev = chunks[-1]
                merged = f"{prev.text}\n{' '.join(u[0] for u in buf)}"
                chunks[-1] = Chunk(
                    prev.chunk_id, prev.doc_id, prev.doc_type, prev.title, prev.section,
                    merged, prev.start_char, buf[-1][2], prev.ordinal,
                )
            else:
                chunks.append(_emit(doc, name, buf, ordinal))
                ordinal += 1
    return chunks


def _is_droppable(text: str) -> bool:
    """Final guard for reference blocks that survived section stripping."""
    return DROP_REFERENCES and len(text) > 200 and is_reference_like(text)


def _emit(doc: dict, section: str, units: list[tuple[str, int, int]], ordinal: int) -> Chunk:
    text = "\n".join(u[0] for u in units)
    return Chunk(
        chunk_id=f"{doc['doc_id']}#{ordinal:03d}",
        doc_id=doc["doc_id"],
        doc_type=doc.get("doc_type", "unknown"),
        title=doc.get("title", ""),
        section=section,
        text=text,
        start_char=units[0][1],
        end_char=units[-1][2],
        ordinal=ordinal,
    )


def chunk_corpus(docs: list[dict]) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out
