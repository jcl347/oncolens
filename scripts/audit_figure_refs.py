"""Can a figure-retrieval benchmark be built from FOUND data, the way 4.4's was?

The existing labels are citation contexts: a sentence citing a paper is an expert's
description of that paper, written after reading it. Adding figures to the index cannot
move that benchmark, because the judgment is which DOCUMENT is right and the document is
the same either way. So the current eval is structurally incapable of showing figure value
-- the 4.13 fault at the scale of a whole subsystem.

The same trick should work one level down. JATS marks in-text figure references as
<xref ref-type="fig" rid="F3">, so the sentence around it is the AUTHOR'S OWN description
of what that figure shows: "Tumour volume was reduced by 60% (Fig. 3B)". Query = the
sentence, answer = the figure. Free, expert, disjoint from the caption, and not generated
by any model we also evaluate.

This measures whether enough of them exist.
"""
import sys, re, collections, statistics
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(r"c:\Users\jcl34\OneDrive\Documents\GitHub\oncolens-1")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from oncolens.env import load_env, local_data_dir
load_env(ROOT)

cache = local_data_dir() / "jats_cache"
files = sorted(cache.glob("*.xml"))

n_docs = n_xref = n_docs_with = 0
per_doc = []
sentences = []
panel_refs = 0
PANEL = re.compile(r"\b(?:fig(?:ure)?s?\.?\s*\d+\s*[A-Ha-h]\b|\([A-Ha-h]\))")

for f in files:
    try:
        root = ET.parse(f).getroot()
    except Exception:
        continue
    n_docs += 1
    # figure ids present in this document
    fig_ids = {fg.get("id") for fg in root.findall(".//fig") if fg.get("id")}
    xrefs = [x for x in root.findall(".//xref") if x.get("ref-type") == "fig"]
    if xrefs:
        n_docs_with += 1
    n_xref += len(xrefs)
    per_doc.append(len(xrefs))

    # Pull the sentence around each reference, from the paragraph that contains it.
    for p in root.findall(".//p"):
        has = [x for x in p.iter("xref") if x.get("ref-type") == "fig"]
        if not has:
            continue
        text = " ".join("".join(p.itertext()).split())
        for s in re.split(r"(?<=[.!?])\s+", text):
            if not re.search(r"\b[Ff]ig", s):
                continue
            if PANEL.search(s):
                panel_refs += 1
            if len(sentences) < 8 and 80 < len(s) < 300:
                rid = has[0].get("rid")
                sentences.append((f.stem, rid, s))

print(f"documents parsed            : {n_docs}")
print(f"documents with a fig xref   : {n_docs_with}  ({n_docs_with/max(n_docs,1):.1%})")
print(f"total in-text figure refs   : {n_xref:,}")
print(f"median refs per document    : {statistics.median(per_doc):.0f}")
print(f"sentences naming a PANEL    : {panel_refs:,}  "
      f"(these are panel-level labels, e.g. 'Fig 3B')")
print()
print("A citation-context label set of 7,056 took the corpus this far. This is the same")
print("kind of found data one level down, and there are more of them.")
print("\n=== sample author-written figure descriptions (query <- answer) ===")
for pmc, rid, s in sentences:
    print(f"\n  PMC{pmc}  -> {rid}")
    print(f"    {s}")
