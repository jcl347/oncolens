"""Are figure captions ALREADY in the index? Measured, after I asserted they were not.

I claimed 17.4M characters of caption text were absent, reasoning that figures are images
so nothing survives the plain-text rendition. Wrong: NCBI inlines the caption into the body
flow. This measures the real rate instead of reasoning about it again.

Method: for each figure in the JATS, take a distinctive interior window of its caption and
look for it in the concatenated chunk text of that document. Interior window, not prefix,
because "Figure 1." prefixes are formatted inconsistently.
"""
import sys, os, re, random, collections
from pathlib import Path
from xml.etree import ElementTree as ET
import psycopg

ROOT = Path(r"c:\Users\jcl34\OneDrive\Documents\GitHub\oncolens-1")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from oncolens.env import load_env, local_data_dir
load_env(ROOT)

dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
cache = local_data_dir() / "jats_cache"


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


# map pmcid -> doc_id
with psycopg.connect(dsn) as cx, cx.cursor() as cur:
    cur.execute("SELECT doc_id, meta->>'pmcid' FROM documents WHERE meta->>'pmcid' IS NOT NULL")
    by_pmc = {re.sub(r"\D", "", p or ""): d for d, p in cur.fetchall() if p}
print(f"documents with a pmcid: {len(by_pmc):,}")

files = [f for f in sorted(cache.glob("*.xml")) if re.sub(r"\D", "", f.stem) in by_pmc]
random.seed(11)
sample = random.sample(files, min(250, len(files)))
print(f"sampling {len(sample)} documents that are BOTH cached and indexed\n")

present = absent = 0
tbl_present = tbl_absent = 0
missing_examples = []

with psycopg.connect(dsn) as cx, cx.cursor() as cur:
    for f in sample:
        doc = by_pmc[re.sub(r"\D", "", f.stem)]
        cur.execute("SELECT string_agg(text, ' ' ORDER BY ordinal) FROM chunks "
                    "WHERE doc_id=%s AND kind='passage'",
                    (doc,))
        row = cur.fetchone()
        body = norm(row[0] or "")
        if not body:
            continue
        try:
            root = ET.parse(f).getroot()
        except Exception:
            continue

        for fg in root.findall(".//fig"):
            cap_el = fg.find("./caption")
            cap = norm("".join(cap_el.itertext())) if cap_el is not None else ""
            if len(cap) < 80:
                continue
            probe = cap[30:110]           # interior window, skips the label prefix
            if probe in body:
                present += 1
            else:
                absent += 1
                if len(missing_examples) < 4:
                    missing_examples.append((f.stem, cap[:150]))

        for tb in root.findall(".//table-wrap"):
            cap_el = tb.find("./caption")
            cap = norm("".join(cap_el.itertext())) if cap_el is not None else ""
            if len(cap) < 40:
                continue
            if cap[10:70] in body:
                tbl_present += 1
            else:
                tbl_absent += 1

tot = present + absent
print("=== FIGURE captions ===")
print(f"  already present in indexed text : {present:,} / {tot:,}  ({present/max(tot,1):.1%})")
print(f"  genuinely absent                : {absent:,} / {tot:,}  ({absent/max(tot,1):.1%})")
ttot = tbl_present + tbl_absent
print("\n=== TABLE captions ===")
print(f"  already present : {tbl_present:,} / {ttot:,}  ({tbl_present/max(ttot,1):.1%})")
print(f"  absent          : {tbl_absent:,} / {ttot:,}  ({tbl_absent/max(ttot,1):.1%})")

if missing_examples:
    print("\n--- captions that really are missing ---")
    for pmc, cap in missing_examples:
        print(f"  PMC{pmc}: {cap}")
