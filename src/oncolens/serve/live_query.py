"""Serving path that queries the live store, rather than a bundled snapshot.

**Why the bundled artifact is no longer sufficient.** `api/search.py` was written against
an offline-built artifact containing every posting list and chunk vector. That is a good
design for a small fixed corpus — zero infrastructure, numpy-only runtime — and it stops
working at scale: the corpus is now 59,306 passages, whose vectors alone are 45 MB before
any text or postings, against Vercel's function bundle limits. Past roughly 10k chunks the
artifact must be replaced by a query against the store.

**The failure this module is built to prevent.** A query vector produced by one embedding
model, compared against document vectors produced by another, does not raise: cosine
distance happily compares two unrelated 192-dimension spaces and returns a confident,
meaningless ranking. This corpus was first embedded with LSA and later re-embedded with
``text-embedding-3-small`` at the *same* dimensionality, so nothing about the column shape
reveals a mismatch. Every query therefore checks ``index_config`` first and refuses to
answer if the encoders disagree.

**Which configuration this serves, and why.** Measured on 2,225 citation-context queries,
paired permutation test, Bonferroni-corrected within the iteration:

    hybrid-openai   nDCG@10 0.4526   +0.0878 vs shipping   CI [+0.0762, +0.0995]
    openai          nDCG@10 0.4116   +0.0469
    bm25            nDCG@10 0.3888   +0.0241   <- beats the shipping hybrid ALONE
    hybrid-lsa      nDCG@10 0.3647   (what shipped)
    lsa             nDCG@10 0.3088   -0.0560

The LSA dense arm was measurably *harmful*: BM25 on its own outranked the hybrid that
included it, 466 wins to 432 with p < 0.0001. So the default here is lexical + OpenAI,
and ``dense_weight=0`` degrades cleanly to the BM25-only arm that still beats the old
default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import neon_store

DEFAULT_BACKEND = os.environ.get("ONCOLENS_EMBED_BACKEND", "openai")
DEFAULT_DIM = int(os.environ.get("ONCOLENS_EMBED_DIM", "192"))


@dataclass
class LiveIndex:
    """Holds the connection and query encoder for the lifetime of a warm container.

    Both are expensive to create and safe to reuse: a serverless container handles many
    requests, and reconnecting per request would dominate latency on a cold pool.
    """

    dsn: str
    backend_name: str = DEFAULT_BACKEND
    dim: int = DEFAULT_DIM
    _conn: object | None = field(default=None, repr=False)
    _backend: object | None = field(default=None, repr=False)
    _checked: bool = field(default=False, repr=False)

    def conn(self):
        import psycopg

        if self._conn is None or getattr(self._conn, "closed", 1):
            # Neon's pooler endpoint is what makes this viable from serverless; a direct
            # endpoint exhausts connection slots as concurrency rises.
            self._conn = psycopg.connect(self.dsn, connect_timeout=10, autocommit=True)
            self._checked = False
        if not self._checked:
            neon_store.assert_embedding_matches(self._conn, self.backend_name, self.dim)
            self._checked = True
        return self._conn

    def encoder(self):
        if self._backend is None:
            from ..retrieval.dense import make_backend

            self._backend = make_backend(self.backend_name, dim=self.dim)
        return self._backend

    def encode_query(self, query: str) -> list[float]:
        return [float(x) for x in self.encoder().encode_queries([query])[0]]

    def search(self, query: str, *, top_k: int = 10, candidates: int = 200,
               bm25_weight: float = 1.0, dense_weight: float = 1.0) -> dict:
        conn = self.conn()
        vec = self.encode_query(query) if dense_weight > 0 else [0.0] * self.dim
        rows = neon_store.hybrid_search(
            conn, query, vec, candidates=candidates, top_k=top_k,
            bm25_weight=bm25_weight, dense_weight=dense_weight,
        )
        results = [_shape(r, query) for r in rows]
        # SAY SO WHEN THE WORDS WERE NEVER FOUND. The dense arm has no relevance
        # threshold, so it always returns its nearest neighbours and this endpoint can
        # never come back empty. Without this flag a misspelling returns ten confident
        # results and the reader concludes the corpus holds work on a term it has never
        # seen. Absence of evidence has to be reported as absence, not as ten hits.
        any_lexical = any(r.get("matched_by") != "semantic" for r in results)
        out = {
            "query": query,
            "backend": self.backend_name,
            "source": "neon",
            "lexical_match": any_lexical,
            "results": results,
        }
        if results and not any_lexical:
            out["notes"] = [
                "No passage in this corpus contains your search terms. These results are "
                "the nearest matches by meaning, which may be useful, but nothing below "
                "is a literal hit. Check the spelling, or treat these as leads rather "
                "than as evidence the corpus covers this term."
            ]
        return out


def _shape(row: dict, query: str) -> dict:
    """Attach clause offsets so the UI can highlight the matched span.

    The offsets are section-absolute and computed here rather than stored, because they
    depend on the query. ``start_char``/``end_char`` on the row locate the passage inside
    its section; ``base_offset`` rebases the clause offsets onto the same scale.

    **This must emit the SAME SHAPE as the artifact path in ``api/search.py``, and for a
    long time it did not.** Production takes this branch whenever ``POSTGRES_URL`` is set,
    so the divergence was the only thing users ever saw. It dropped ``best_clause``,
    ``doc_type`` and ``meta``, hand-rolled the clause dicts without ``matched_terms`` or
    ``spans``, and computed clause offsets with no ``base_offset``. The client reads
    ``passage.best_clause``, got ``undefined``, and fell back to a raw 320-character
    truncation: **no highlight rendered on any query, on the only path production uses**,
    and the PubMed link never appeared because the PMID was buried in a ``source`` object
    the client does not read.

    That is §4.11's compare-view drift repeated here. The lesson produced a contract test
    for ``/api/compare`` and was not extended to ``/api/search``;
    ``tests/test_search_contract.py`` now covers both sides.
    """
    from ..spans import find_clauses

    # neon_store.hybrid_search already nests the passage fields; reading them from the top
    # level yields None for every offset, which silently removes the provenance that is the
    # entire point of the product rather than raising.
    p = row.get("passage") or {}
    text = p.get("text") or row.get("text") or ""
    # base_offset is load-bearing: without it the clause offsets are passage-relative
    # while passage.start_char is section-absolute, so the UI prints a character range
    # that does not exist in the source article.
    base = p.get("start_char") or 0
    try:
        found = find_clauses(text, query, base_offset=base, max_clauses=3)
    except Exception:  # noqa: BLE001 — highlighting must never break a result
        found = []
    clauses = [c.as_dict() for c in found]
    meta = row.get("meta") or {}
    pmid = (row.get("doc_id") or "").replace("PAPER:PMID", "") or None
    return {
        "doc_id": row.get("doc_id"),
        "title": row.get("title"),
        "doc_type": row.get("doc_type") or "paper",
        "year": row.get("year"),
        # The client reads meta.pmid / meta.journal directly. Carry the stored metadata
        # through and guarantee pmid, which is derivable from doc_id even when the row
        # has no meta at all.
        "meta": {**meta, "pmid": meta.get("pmid") or pmid},
        "score": round(float(row.get("score") or 0.0), 6),
        # Which arm retrieved this. `lexical` means the query's words are literally in the
        # passage; `semantic` means only the embedding matched, which is the honest label
        # for a result the reader might otherwise take as a term hit.
        "matched_by": (
            "lexical+semantic" if row.get("lex_rank") is not None and row.get("dense_rank") is not None
            else "lexical" if row.get("lex_rank") is not None
            else "semantic"
        ),
        "passage": {
            "chunk_id": p.get("chunk_id"),
            "section": p.get("section"),
            "start_char": p.get("start_char"),
            "end_char": p.get("end_char"),
            "text": text,
            "clauses": clauses,
            "best_clause": clauses[0] if clauses else None,
        },
        # Retained for any consumer that already reads it; `meta` above is what the UI uses.
        "source": {
            "pmid": pmid,
            "pmcid": meta.get("pmcid"),
            "license": meta.get("license_code"),
            "blob_url": meta.get("blob_url"),
        },
    }


#: A cell must clear this to count as "reports this dimension". Below it the honest answer
#: is NOT REPORTED — a blank cell reads as *no effect* when it means *not measured*, and
#: that misreading is worse than an empty table.
CELL_MIN_SCORE = 0.08


def _cues_to_tsquery(cues: tuple[str, ...]) -> str:
    """Build a valid ``to_tsquery`` string from human-written cue phrases.

    The cue lists are written for readability, so they contain things Postgres rejects:
    trailing spaces (``"hr "``), bare operators (``"p<"``, ``"n ="``), and punctuation
    (``"95% ci"``). Naively substituting ``<->`` for spaces produced
    ``"hr <-> | odds <-> ratio"`` and a syntax error. Cues are therefore reduced to
    alphanumeric tokens, multi-word phrases become adjacency groups, and anything left
    empty is dropped rather than emitted as a dangling operator.
    """
    import re as _re

    parts: list[str] = []
    for cue in cues:
        tokens = [t for t in _re.split(r"[^A-Za-z0-9]+", cue.lower()) if t]
        if not tokens:
            continue
        phrase = " <-> ".join(tokens)
        parts.append(f"({phrase})" if len(tokens) > 1 else phrase)
    # Deduplicate while preserving order; repeated terms only inflate the parse.
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return " | ".join(uniq)


def aspect_catalogue() -> list[dict]:
    """Every dimension the comparison CAN fill, not just the four it defaults to.

    Eight aspects are defined and four were reachable, because the client hardcoded the
    default tuple and never sent ``?aspect=`` even though the endpoint has always parsed
    it. The unreachable four included ``resistance`` — "what mechanism of resistance or
    escape was identified" — in an oncology tool whose own search placeholder invites
    exactly that question.
    """
    from ..aspects import ASPECTS, DEFAULT_ASPECT_KEYS

    return [{"key": a.key, "label": a.label, "question": a.question,
             "numeric": a.numeric, "default": a.key in DEFAULT_ASPECT_KEYS}
            for a in ASPECTS]


def compare(index: LiveIndex, query: str, *, n_papers: int = 5,
            aspect_keys: tuple[str, ...] | None = None,
            doc_ids: tuple[str, ...] | None = None) -> dict:
    """Papers x technical dimensions, every cell carrying its own citation.

    **Why this is not just top-k.** Asking "how do these studies measure X" with a plain
    top-k returns passages clustered inside the single most on-topic paper, several of
    which state no method at all. So the query selects *documents* first, then each cell is
    filled by searching **within that document** for a passage that actually reports the
    dimension — full-text ranked against the aspect's cue vocabulary, and for numeric
    aspects (cohort size, effect size) required to contain a digit, because a cohort
    sentence without a number is not a cohort answer.

    Runs entirely in SQL against the live store. The previous implementation imported the
    offline ``Retriever``, which needs a fitted in-process index and scipy — neither
    present in a serverless function, so it returned 500 in production.
    """
    from ..aspects import ASPECTS_BY_KEY, DEFAULT_ASPECT_KEYS

    keys = tuple(aspect_keys or DEFAULT_ASPECT_KEYS)
    aspects = [ASPECTS_BY_KEY[k] for k in keys if k in ASPECTS_BY_KEY]
    if not aspects:
        aspects = [ASPECTS_BY_KEY[k] for k in DEFAULT_ASPECT_KEYS]

    # EXPLICIT PAPERS BEAT RE-GUESSING THE QUERY. Without this the only way to tabulate
    # three specific papers was to reword the query until the ranking happened to surface
    # them, which is not a workflow, it is a slot machine.
    if doc_ids:
        conn0 = index.conn()
        with conn0.cursor() as cur:
            cur.execute("SELECT doc_id, title, year FROM documents WHERE doc_id = ANY(%s)",
                        (list(doc_ids),))
            rows0 = cur.fetchall()
        found = {r[0]: (r[1], r[2]) for r in rows0}
        # Preserve the caller's order; drop ids this corpus does not hold.
        docs = [d for d in doc_ids if d in found]
        titles = {d: found[d][0] for d in docs}
        years = {d: found[d][1] for d in docs}
    else:
        top = index.search(query, top_k=n_papers)
        docs = [r["doc_id"] for r in top["results"]]
        titles = {r["doc_id"]: r["title"] for r in top["results"]}
        years = {r["doc_id"]: r.get("year") for r in top["results"]}
    if not docs:
        # SAME SHAPE as the success path. This branch previously returned `list(keys)` —
        # a list of strings — where the success path returns {key,label,numeric} objects,
        # so the field's type depended on whether anything matched. A client written
        # against one branch is broken by the other, and the empty case is exactly the one
        # nobody exercises by hand.
        return {"query": query,
                "aspects": [{"key": a.key, "label": a.label, "numeric": a.numeric}
                            for a in aspects],
                "doc_ids": [], "coverage": 0.0,
                "cells": {}, "titles": {}, "years": {}, "source": "neon",
                "notes": ["no documents matched the query"]}

    conn = index.conn()
    cells: dict[str, dict] = {}
    filled = 0
    with conn.cursor() as cur:
        for asp in aspects:
            tsquery = _cues_to_tsquery(asp.cues)
            if not tsquery:
                continue
            cur.execute(
                """
                SELECT DISTINCT ON (c.doc_id)
                       c.doc_id, c.chunk_id, c.section, c.start_char, c.end_char, c.text,
                       ts_rank_cd(c.tsv, to_tsquery('english', %(tsq)s)) AS rank
                FROM chunks c
                WHERE c.doc_id = ANY(%(docs)s)
                  -- Body passages only. Figure rows exist for reading, not comparison:
                  -- a caption answering an aspect would look like the paper reported it
                  -- in text, which is the "not reported" over-claim §4.15 corrected.
                  AND c.kind = 'passage'
                  AND c.tsv @@ to_tsquery('english', %(tsq)s)
                  AND (NOT %(numeric)s OR c.text ~ '[0-9]')
                ORDER BY c.doc_id, rank DESC
                """,
                {"tsq": tsquery, "docs": docs, "numeric": asp.numeric},
            )
            found = {r[0]: r for r in cur.fetchall()}
            for doc in docs:
                row = found.get(doc)
                key = f"{doc}|{asp.key}"
                if row is None or float(row[6]) < CELL_MIN_SCORE:
                    # SAY WHICH KIND OF EMPTY THIS IS. "No chunk in this paper contains
                    # any of the aspect's cue words" and "a chunk matched but scored
                    # 0.079 against a 0.08 threshold" are different facts, and neither is
                    # "the paper does not report this dimension". Emitting the near-miss
                    # score lets the reader see how close the call was instead of
                    # inheriting a verdict.
                    cells[key] = {
                        "reported": False,
                        "text": None,
                        "reason": "no_cue_match" if row is None else "below_threshold",
                        "score": None if row is None else round(float(row[6]), 4),
                        "threshold": CELL_MIN_SCORE,
                    }
                    continue
                filled += 1
                # SEND THE PASSAGE WHOLE. It used to be clipped to 700 characters while
                # start_char/end_char still described the full span (chunks target 900 and
                # cap at 1600), and the viewer captioned it "verbatim, at the offsets
                # above". The offsets therefore did not bound the text shown, which is the
                # one rule broken silently. Worse for numeric aspects: the digit that
                # qualified the cell could sit in the discarded tail, so a cohort cell
                # could contain no cohort number.
                cells[key] = {
                    "reported": True,
                    "chunk_id": row[1],
                    "section": row[2],
                    "start_char": row[3],
                    "end_char": row[4],
                    "text": row[5] or "",
                    "score": round(float(row[6]), 4),
                }

    total = len(docs) * len(aspects)
    return {
        "query": query,
        "aspects": [{"key": a.key, "label": a.label, "numeric": a.numeric} for a in aspects],
        # The full menu travels with the result so the UI can offer the dimensions it is
        # not currently showing, rather than hardcoding a subset of a subset.
        "all_aspects": aspect_catalogue(),
        "doc_ids": docs,
        "titles": titles,
        "years": years,
        "coverage": round(filled / total, 4) if total else 0.0,
        "cells": cells,
        "source": "neon",
        # WHAT AN EMPTY CELL ACTUALLY MEANS, stated as the retrieval fact it is.
        #
        # This used to read "the paper does not report that dimension", which the
        # mechanism cannot support: a cell is filled when some chunk clears
        # ts_rank_cd >= 0.08 against an OR of the aspect's cue words, with numeric aspects
        # needing only a digit somewhere in the chunk. So an empty cell can be a paper
        # that used different wording, or one that scored 0.079. A researcher assembling a
        # review table writes "not reported" down as a finding about the paper, and that
        # is the tool putting a claim in their notes that it never established.
        "notes": [
            "An empty cell means no passage in that paper matched this dimension's cue "
            "vocabulary above the retrieval threshold. That is a statement about this "
            "search, not about the paper: the authors may have reported the dimension in "
            "wording the cue list does not cover. Open a filled cell to check the passage "
            "before recording anything from this table.",
        ],
    }


def get_document(index: LiveIndex, doc_id: str, *, highlight: str = "") -> dict | None:
    """A whole paper, in reading order, with every passage's own offsets.

    **Why the passages rather than a stored blob.** The passage rows ARE the article as
    this system holds it: reference-stripped, chunked at offsets that retrieval already
    cites. Rendering from them means the page a reader scrolls is exactly the text that was
    searched, so a highlighted result cannot point at something the reader cannot find.
    Serving a separate full-text copy would let the two drift.

    ``ordinal`` is the chunk order within a section; ``ORDER BY section, ordinal`` puts the
    article back together. ``start_char`` is kept on every passage so a deep link from a
    search result can scroll to the exact clause.
    """
    conn = index.conn()
    with conn.cursor() as cur:
        cur.execute(
            """SELECT doc_id, title, year, descriptors, meta
               FROM documents WHERE doc_id = %s""", (doc_id,))
        row = cur.fetchone()
        if not row:
            return None
        did, title, year, descriptors, meta = row

        cur.execute(
            """SELECT chunk_id, section, ordinal, start_char, end_char, text
               FROM chunks WHERE doc_id = %s AND kind = 'passage'
               ORDER BY section, ordinal""", (doc_id,))
        passages = [
            {"chunk_id": r[0], "section": r[1], "ordinal": r[2],
             "start_char": r[3], "end_char": r[4], "text": r[5]}
            for r in cur.fetchall()
        ]

        # Figures are read alongside the text, so the reading page can show what the
        # passages refer to. `image_uri` points at NCBI's own public object, which is the
        # artifact a reader verifies against — a caption has no character range worth
        # citing, so the picture IS the provenance (§1).
        cur.execute(
            """SELECT figure_id, figure_label, text, image_uri, figure_type, figure_type_src
               FROM chunks WHERE doc_id = %s AND kind = 'figure'
               ORDER BY figure_id""", (doc_id,))
        figures = [
            {"figure_id": r[0], "label": r[1] or "", "caption": r[2],
             "image_uri": r[3], "figure_type": r[4],
             # 'caption' means the publisher's own words named the type; anything else is
             # inferred and must not be shown as though the paper said it.
             "figure_type_source": r[5]}
            for r in cur.fetchall() if r[3]
        ]

        # Tables are shipped as PARSED ROWS, not markup. 95.3% of this corpus's tables
        # carry a machine-readable <table>, so the numbers are already structured and the
        # page can render a real grid without ever injecting publisher HTML.
        cur.execute(
            """SELECT chunk_id, figure_label, text, table_rows
               FROM chunks WHERE doc_id = %s AND kind = 'table'
               ORDER BY chunk_id""", (doc_id,))
        tables = []
        for r in cur.fetchall():
            data = r[3] if isinstance(r[3], dict) else {}
            rows = data.get("rows") or []
            if not rows:
                continue
            tables.append({
                "table_id": r[0].split("#tbl:")[-1],
                "label": r[1] or "",
                "caption": r[2],
                "rows": rows,
                # The real size, so a truncated table says so rather than quietly
                # presenting 40 rows as the whole thing.
                "n_rows": data.get("n_rows", len(rows)),
                "n_cols": data.get("n_cols", 0),
                "truncated": bool(data.get("truncated")),
                # Abbreviation key and significance markers: a table's units live here.
                "foot": data.get("foot") or "",
            })

    meta = meta if isinstance(meta, dict) else {}
    mesh = [m for m in (meta.get("mesh") or []) if isinstance(m, dict)]
    return {
        "doc_id": did,
        "title": title or "",
        "year": year,
        "pmid": meta.get("pmid") or (did or "").replace("PAPER:PMID", ""),
        "pmcid": meta.get("pmcid"),
        "journal": meta.get("journal") or "",
        "license_code": meta.get("license_code"),
        "full_text_chars": meta.get("full_text_chars"),
        "blob_url": meta.get("blob_url"),
        # Major topics first: NLM's own statement of what the paper is centrally about.
        "mesh_major": [m["descriptor"] for m in mesh if m.get("major") and m.get("descriptor")],
        "mesh_minor": [m["descriptor"] for m in mesh
                       if not m.get("major") and m.get("descriptor")],
        "descriptors": [d.replace("MESH:", "") for d in (descriptors or [])],
        "grants": meta.get("grants") or [],
        "n_passages": len(passages),
        "passages": passages,
        "n_figures": len(figures),
        "figures": figures,
        "n_tables": len(tables),
        "tables": tables,
        "highlight": highlight,
        "source": "neon",
    }


_LIVE: LiveIndex | None = None


def get_live_index() -> LiveIndex | None:
    """Return a warm live index, or None when no database is configured.

    Returning None rather than raising lets the caller fall back to the bundled artifact,
    which is the right behaviour for a preview deployment that has no store attached.
    """
    global _LIVE
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    if _LIVE is None:
        _LIVE = LiveIndex(dsn=dsn)
    return _LIVE
