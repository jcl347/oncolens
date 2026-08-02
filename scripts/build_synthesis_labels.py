#!/usr/bin/env python
"""Build the **synthesis** stratum: R&D questions with expert-curated answer *sets*.

Every other stratum in this project is known-item lookup — one query, one right paper.
Researchers do not work that way. They ask *"what are the known resistance mechanisms to
osimertinib?"* and expect a set of papers covering the field.

A review article already contains that mapping, made by someone who read the literature:

    review section heading   ->  the R&D question
    works cited beneath it   ->  the answer set

Measured on real cached articles, headings look like *"Lactate-hepcidin axis in
ferroptosis"* (9 citations) and *"MCTs and GPR81/HCAR1"* (16 citations). No annotation was
required and no model was involved in choosing them.

**Why this is a different measurement, not a bigger one.** With a 7-to-20-document answer
set, ranking quality finally means something: nDCG has a real ideal ordering to compare
against, recall@k measures coverage of a field rather than a coin flip, and a system that
returns one excellent paper and nothing else is correctly scored as incomplete.

**Validity hazards, each with a guard:**

1. *The review itself is a perfect answer.* It contains every cited claim and would rank
   first on lexical overlap. It is excluded per query, and the exclusion is asserted.
2. *Structural headings.* "Introduction" is not a research question, and "Conclusion"
   cites everything indiscriminately. Both are filtered.
3. *One prolific review dominating.* Capped per source article.
4. *Sets too small to be sets.* A heading citing fewer than three held papers is a lookup
   question wearing a synthesis costume; it is dropped.

    python scripts/build_synthesis_labels.py --email you@org.edu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.sources import jats  # noqa: E402

#: A heading citing fewer than this many *held* papers is not a set task.
MIN_IN_CORPUS = 3
#: Cap per review, so one 300-reference survey cannot become the benchmark.
MAX_SECTIONS_PER_REVIEW = 8
#: Beyond this the "set" is really a chapter bibliography.
MAX_IN_CORPUS = 40


def corpus_doc_ids() -> tuple[set[str], dict[str, str]]:
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, meta->>'pmcid' FROM documents")
        rows = cur.fetchall()
    return {r[0] for r in rows}, {r[1]: r[0] for r in rows if r[1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-in-corpus", type=int, default=MIN_IN_CORPUS)
    args = ap.parse_args()
    load_env()

    ids, by_pmcid = corpus_doc_ids()
    cache = local_data_dir() / "jats_cache"
    xmls = sorted(cache.glob("*.xml"))
    print(f"{len(ids)} corpus documents, {len(xmls)} cached JATS files")

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    exclude: dict[str, str] = {}
    notes: dict[str, str] = {}
    stats = Counter()

    for f in xmls:
        pmcid = f.stem
        source_doc = by_pmcid.get(pmcid)
        try:
            xml = f.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        refs = jats.parse_references(xml)
        if not refs:
            continue
        sections = jats.extract_section_citations(xml)
        if not sections:
            continue
        title = jats.article_title(xml) or ""
        kept_here = 0
        for sec in sections:
            if kept_here >= MAX_SECTIONS_PER_REVIEW:
                stats["source_cap"] += 1
                break
            targets: dict[str, int] = {}
            for rid in sec.rids:
                ref = refs.get(rid)
                if not ref or not ref.pmid:
                    continue
                doc = f"PAPER:PMID{ref.pmid}"
                if doc in ids and doc != source_doc:
                    targets[doc] = 3
            stats["sections_seen"] += 1
            if len(targets) < args.min_in_corpus:
                stats["too_few_in_corpus"] += 1
                continue
            if len(targets) > MAX_IN_CORPUS:
                stats["too_many"] += 1
                continue
            qid = f"synth:{pmcid}:{abs(hash(sec.heading)) % 100000}"
            # The heading alone can be ambiguous out of context ("Clinical trials");
            # the review's own topic is what disambiguates it, exactly as it does for a
            # reader scanning the table of contents.
            topic = title.split(":")[0][:60].strip()
            queries[qid] = f"{sec.heading} in {topic}" if topic else sec.heading
            qrels[qid] = targets
            if source_doc:
                exclude[qid] = source_doc
            notes[qid] = f"{len(targets)} held / {len(sec.rids)} cited | {title[:60]}"
            kept_here += 1
            stats["kept"] += 1

    print(f"\nsections examined      {stats['sections_seen']}")
    print(f"  too few held papers  {stats['too_few_in_corpus']}")
    print(f"  too many (chapter)   {stats['too_many']}")
    print(f"  source cap           {stats['source_cap']}")
    print(f"  KEPT                 {stats['kept']}")

    if not queries:
        print("\nno synthesis questions built — the corpus holds too few cited papers "
              "per review section. Snowball ingestion along the citation graph is the fix.")
        return 1

    sizes = sorted(len(v) for v in qrels.values())
    print(f"\nanswer-set size: min {sizes[0]}, median {sizes[len(sizes)//2]}, max {sizes[-1]}")
    print(f"distinct source reviews: {len(set(q.split(':')[1] for q in queries))}")

    print("\nsample R&D questions (heading + review topic -> held papers):")
    for qid in list(queries)[:10]:
        print(f"  [{len(qrels[qid]):>2} papers] {queries[qid][:96]}")

    out = Path(args.out) if args.out else local_data_dir() / "qrels_synthesis.json"
    out.write_text(json.dumps({"queries": queries, "qrels": qrels,
                               "exclude": exclude, "notes": notes}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
