#!/usr/bin/env python
"""Make figures first-class rows so a paper can be read with its images.

**What this does and deliberately does not do.** It adds one `chunks` row per JATS `<fig>`,
carrying the verbatim caption and a URL to the real image. It does **not** put figures into
retrieval: `kind='figure'` rows are excluded by the search CTEs until the effect is measured
(docs/PLAN_FIGURES.md §3.9, Path A). So this changes what a reader can *see* and provably
cannot change any ranking metric.

**Images are not copied anywhere.** NCBI publishes them on a public S3 bucket, which the
metadata JSON already names in `media_urls`, and the URL is constructible as
``{HTTPS_BASE}/{pmcid}.{version}/{filename}`` — verified to serve without the md5 query
parameter. Re-hosting 16,792 images would add a storage bill, a sync problem and a licence
question for no benefit, and §2's Blob store rejects REST uploads anyway.

**Provenance for something with no character offsets.** §1 requires a result to point back
at its source. A caption has real offsets *into itself*, and the artifact a reader actually
verifies against is the picture, so `image_uri` is the provenance — not a character range
invented to satisfy a NOT NULL constraint.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.sources import pmc_cloud as pc  # noqa: E402

XLINK = "{http://www.w3.org/1999/xlink}href"

#: Caption patterns -> figure type. Captions are PRECISE when they fire: a caption saying
#: "Kaplan-Meier" is near-certain. Measured coverage 59.5%; the residue is left `NULL` and
#: reported as unknown rather than guessed (§4.15's "not reported" lesson). A BiomedCLIP
#: pass can fill it later at ~61% precision, tagged `figure_type_src='biomedclip'`.
FIGURE_TYPES = [
    ("kaplan-meier", r"\b(kaplan[\s-]?meier|survival curve)"),
    ("forest-plot", r"\bforest plot"),
    ("western-blot", r"\b(western blot|immunoblot|gel electrophor)"),
    ("flow-cytometry", r"\b(flow cytometr|facs\b|gating)"),
    ("microscopy", r"\b(immunohistochem|micrograph|h&e\b|histolog|confocal|fluoresc)"),
    ("heatmap", r"\bheat ?map|clustergram"),
    ("volcano-plot", r"\bvolcano plot"),
    ("roc-curve", r"\b(roc curve|receiver operating)"),
    ("dose-response", r"\b(dose[\s-]response|ic50|ec50)"),
    ("schematic", r"\b(schematic|flow ?chart|study design|consort|prisma|workflow)"),
    ("bar-chart", r"\bbar (?:chart|plot|graph)"),
    ("scatter-plot", r"\bscatter ?plot"),
]

SCHEMA_SQL = """
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS kind            TEXT NOT NULL DEFAULT 'passage';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_id       TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS image_uri       TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_label    TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_type     TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_type_src TEXT;
-- Tables are stored as PARSED ROWS, never as raw markup. Rendering publisher HTML would
-- mean injecting untrusted markup into the page; a JSON grid renders as a real table with
-- no such exposure, and it is also what makes the cells addressable later.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS table_rows      JSONB;
-- Every type the caption names. 36.5% of figures match more than one, so a single
-- column made the label arbitrary for a third of the corpus.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS figure_types    JSONB;
CREATE INDEX IF NOT EXISTS chunks_kind_idx ON chunks (doc_id, kind);
"""

#: Beyond this a table is a data dump, not something to read on a page. Truncated rather
#: than dropped, with the true size recorded so the page can say what it is not showing.
MAX_TABLE_ROWS = 40
MAX_TABLE_COLS = 12


def txt(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def classify_all(caption: str) -> list[str]:
    """EVERY type the publisher's caption names, not just the first.

    ⚠️ **The single-label version was discarding information it had already computed.**
    Measured on 1,224 held-out figures, **36.5% match more than one pattern** — a composite
    panel genuinely is both a western blot and a bar chart, and ~50% of biomedical figures
    are multi-panel. Returning the first match made the label arbitrary for a third of the
    corpus, and it is what made the type look hard to predict: a semantic classifier scored
    50.7% against the single label and 58.7% against any valid one, and 8 of those points
    were the evaluation being wrong rather than the model.

    So the fix for "the regex is too crude" turned out not to be a better classifier. It
    was to stop throwing away what the regex already knew.
    """
    return [name for name, pat in FIGURE_TYPES if re.search(pat, caption, re.I)]


def classify(caption: str) -> str | None:
    """The primary type: first match in FIGURE_TYPES order. Kept for the page's single
    chip and for callers that want one answer; `classify_all` is the honest set."""
    all_ = classify_all(caption)
    return all_[0] if all_ else None


def figures_from_jats(path: Path) -> list[dict]:
    """Every `<fig>` with a caption and an image, plus the sentences that cite it."""
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []
    pmcid = "PMC" + re.sub(r"\D", "", path.stem)

    # In-text references: the author's own description of what the figure shows. Indexed
    # (not displayed) so a figure is findable by how the paper talks about it.
    cites: dict[str, list[str]] = {}
    for p in root.findall(".//p"):
        xs = [x for x in p.iter("xref") if x.get("ref-type") == "fig"]
        if not xs:
            continue
        body = " ".join("".join(p.itertext()).split())
        for s in re.split(r"(?<=[.!?])\s+", body):
            if not re.search(r"\b[Ff]ig", s):
                continue
            for x in xs:
                for rid in (x.get("rid") or "").split():
                    cites.setdefault(rid, []).append(s)

    out = []
    for i, fg in enumerate(root.findall(".//fig")):
        cap = txt(fg.find("./caption"))
        g = fg.find(".//graphic")
        href = g.get(XLINK) if g is not None else None
        if not cap or not href:
            continue
        fid = fg.get("id") or f"fig{i+1}"
        label = txt(fg.find("./label"))
        ctx = " ".join(dict.fromkeys(cites.get(fid, [])))[:2000]
        out.append({
            "pmcid": pmcid, "fig_id": fid, "label": label, "caption": cap,
            "file": href.split("/")[-1], "cites": ctx,
            "figure_type": classify(cap),
            "figure_types": classify_all(cap),
        })
    return out


def tables_from_jats(path: Path) -> list[dict]:
    """Every `<table-wrap>` whose data is real markup rather than a picture of a table.

    Measured on this corpus: **95.3% of 4,051 tables carry a machine-readable `<table>`**,
    so the numbers are already structured and no vision model is involved. §4.14's "table
    extraction first" principle, and here it costs one XML walk.

    Tables that are only an image are skipped: they belong to the figure path, and
    pretending a picture is a grid would put an empty table on the page.
    """
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return []
    out = []
    for i, tw in enumerate(root.findall(".//table-wrap")):
        tbl = tw.find(".//table")
        if tbl is None:
            continue
        rows: list[list[str]] = []
        for tr in tbl.iter("tr"):
            cells = [txt(c) for c in tr if c.tag in ("td", "th")]
            if any(c for c in cells):
                rows.append(cells[:MAX_TABLE_COLS])
        if not rows:
            continue
        cap = txt(tw.find("./caption"))
        label = txt(tw.find("./label"))
        # The footer carries the abbreviation key and significance markers, which is where
        # a table's units and p-value conventions live. Indexed, not rendered as caption.
        foot = txt(tw.find(".//table-wrap-foot"))
        out.append({
            "table_id": tw.get("id") or f"tbl{i+1}",
            "label": label,
            "caption": cap,
            "rows": rows[:MAX_TABLE_ROWS],
            "n_rows": len(rows),
            "n_cols": max(len(r) for r in rows),
            "foot": foot,
        })
    return out


def resolve_image(fig: dict, session, versions=(1, 2, 3)) -> str | None:
    """Constructed URL, verified with a HEAD. Storing a URL that 404s would show a reader
    a broken image and call it provenance."""
    for v in versions:
        url = f"{pc.HTTPS_BASE}/{fig['pmcid']}.{v}/{fig['file']}"
        try:
            r = session.head(url, timeout=25)
            if r.status_code == 200:
                return url
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env(REPO)
    import psycopg
    import requests

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL not set")

    cache = local_data_dir() / "jats_cache"
    files = sorted(cache.glob("*.xml"))
    if args.limit_docs:
        files = files[: args.limit_docs]
    print(f"JATS cached: {len(files)} documents")

    # pmcid -> doc_id, so figures attach to documents that are actually indexed
    with psycopg.connect(dsn) as cx, cx.cursor() as cur:
        cur.execute("SELECT doc_id, meta->>'pmcid' FROM documents WHERE meta->>'pmcid' IS NOT NULL")
        by_pmc = {re.sub(r"\D", "", (p or "")): d for d, p in cur.fetchall() if p}
    print(f"indexed documents with a pmcid: {len(by_pmc):,}")

    figs: list[dict] = []
    tbls: list[dict] = []
    for f in files:
        doc = by_pmc.get(re.sub(r"\D", "", f.stem))
        if not doc:
            continue
        for fig in figures_from_jats(f):
            fig["doc_id"] = doc
            figs.append(fig)
        for t in tables_from_jats(f):
            t["doc_id"] = doc
            tbls.append(t)
    print(f"figures parsed from indexed documents: {len(figs):,}")
    print(f"tables parsed (machine-readable <table> only): {len(tbls):,}")
    if not figs and not tbls:
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = "oncolens/1.0 (jcl347@cornell.edu)"
    print(f"verifying image URLs with {args.workers} workers ...")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        urls = list(ex.map(lambda f: resolve_image(f, session), figs))
    ok = [(f, u) for f, u in zip(figs, urls) if u]
    print(f"  resolved {len(ok):,}/{len(figs):,} ({len(ok)/len(figs):.1%})")

    typed = sum(1 for f, _ in ok if f["figure_type"])
    print(f"  typed from caption: {typed:,} ({typed/max(len(ok),1):.1%}); "
          f"{len(ok)-typed:,} left NULL rather than guessed")

    if args.dry_run:
        print("\n--- dry run, nothing written. sample figures: ---")
        for f, u in ok[:3]:
            print(f"  {f['doc_id']} {f['fig_id']} [{f['figure_type']}] {u}")
            print(f"     {f['caption'][:110]}")
        print("\n--- sample tables: ---")
        for t in tbls[:3]:
            print(f"  {t['doc_id']} {t['table_id']} {t['n_rows']}x{t['n_cols']}  "
                  f"{t['label']} {t['caption'][:70]}")
            for r in t["rows"][:2]:
                print(f"     | {' | '.join(c[:18] for c in r[:6])}")
        return 0

    rows = []
    for f, u in ok:
        # `text` is the caption VERBATIM and nothing else. The label lives in its own
        # column: prepending it here produced "Figure 1." twice on the page, once as the
        # chip and once at the head of the caption, and more importantly it would mean the
        # stored text is not what the publisher wrote.
        cap = f["caption"].strip()
        rows.append((
            f"{f['doc_id']}#fig:{f['fig_id']}", f["doc_id"], "Figure", 0,
            0, len(cap), cap,
            # indexed_text carries the label and the citing sentences as well; `text` stays
            # verbatim so what a reader sees is only ever the publisher's caption.
            f"{f['label']} {cap} {f['cites']}".strip(),
            "figure", f"{f['pmcid']}:{f['fig_id']}", u, f["label"] or None,
            f["figure_type"], "caption" if f["figure_type"] else None,
            json.dumps(f["figure_types"]) if f["figure_types"] else None,
        ))

    # Tables. `text` is the caption verbatim; the grid goes in `table_rows` as JSON so the
    # page renders a real table without ever injecting publisher markup. `indexed_text`
    # additionally carries the flattened cells and the footer, because a table's numbers
    # and its abbreviation key are exactly what someone searches for.
    trows = []
    for t in tbls:
        cap = t["caption"].strip()
        flat = " ".join(" ".join(r) for r in t["rows"])
        trows.append((
            f"{t['doc_id']}#tbl:{t['table_id']}", t["doc_id"], "Table", 0,
            0, max(len(cap), 1), cap,
            f"{t['label']} {cap} {flat} {t['foot']}".strip()[:8000],
            "table", None, None, t["label"] or None, None, None,
            json.dumps({"rows": t["rows"], "n_rows": t["n_rows"],
                        "n_cols": t["n_cols"], "foot": t["foot"],
                        "truncated": t["n_rows"] > len(t["rows"])}),
        ))

    with psycopg.connect(dsn) as cx:
        with cx.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        cx.commit()
        with cx.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, section, ordinal, start_char,
                        end_char, text, indexed_text, kind, figure_id, image_uri,
                        figure_label, figure_type, figure_type_src, figure_types)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (chunk_id) DO UPDATE SET
                        text=EXCLUDED.text, indexed_text=EXCLUDED.indexed_text,
                        image_uri=EXCLUDED.image_uri, figure_label=EXCLUDED.figure_label,
                        figure_type=EXCLUDED.figure_type,
                        figure_type_src=EXCLUDED.figure_type_src,
                        figure_types=EXCLUDED.figure_types""", rows)
            cur.executemany(
                """INSERT INTO chunks (chunk_id, doc_id, section, ordinal, start_char,
                        end_char, text, indexed_text, kind, figure_id, image_uri,
                        figure_label, figure_type, figure_type_src, table_rows)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (chunk_id) DO UPDATE SET
                        text=EXCLUDED.text, indexed_text=EXCLUDED.indexed_text,
                        figure_label=EXCLUDED.figure_label,
                        table_rows=EXCLUDED.table_rows""", trows)
        cx.commit()
        with cx.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE kind='figure'")
            print(f"\nfigure rows now in chunks: {cur.fetchone()[0]:,}")
            cur.execute("SELECT count(*) FROM chunks WHERE kind='passage' OR kind IS NULL")
            print(f"passage rows (unchanged): {cur.fetchone()[0]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
