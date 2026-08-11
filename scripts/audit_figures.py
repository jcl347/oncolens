"""How much figure and table content is the index currently throwing away?

The corpus is indexed from NCBI's plain-text rendition of JATS. Figures are images and
tables are markup, so neither survives that rendition — but the JATS itself carries both,
with captions, labels and image filenames. Before planning any detector, measure what is
already structurally available and being discarded.

This is the §4.1 question again: PMC's own XML states where things are, so anything that
re-derives that from pixels is solving a problem this corpus does not have.
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
print(f"JATS cached: {len(files)} documents\n")

XLINK = "{http://www.w3.org/1999/xlink}href"


def txt(el):
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


figs_per, tbls_per = [], []
n_docs = n_with_fig = n_with_tbl = 0
n_fig = n_tbl = n_fig_caption = n_fig_graphic = n_tbl_inline = 0
caption_lens = []
samples = []

for f in files:
    try:
        root = ET.parse(f).getroot()
    except Exception:
        continue
    n_docs += 1
    figs = root.findall(".//fig")
    tbls = root.findall(".//table-wrap")
    figs_per.append(len(figs))
    tbls_per.append(len(tbls))
    if figs:
        n_with_fig += 1
    if tbls:
        n_with_tbl += 1
    n_fig += len(figs)
    n_tbl += len(tbls)

    for fg in figs:
        cap = txt(fg.find("./caption"))
        if cap:
            n_fig_caption += 1
            caption_lens.append(len(cap))
        g = fg.find(".//graphic")
        if g is not None and g.get(XLINK):
            n_fig_graphic += 1
        if len(samples) < 6 and cap and len(cap) > 120:
            samples.append((f.stem, txt(fg.find("./label")), cap[:300],
                            (g.get(XLINK) if g is not None else None)))
    for tb in tbls:
        # <table> present means the DATA is in the XML, not only a picture of it.
        if tb.find(".//table") is not None:
            n_tbl_inline += 1

print("=== what the JATS already contains ===")
print(f"  documents parsed          : {n_docs}")
print(f"  with >=1 figure           : {n_with_fig}  ({n_with_fig/n_docs:.1%})")
print(f"  with >=1 table            : {n_with_tbl}  ({n_with_tbl/n_docs:.1%})")
print(f"  total figures             : {n_fig:,}   median/doc {statistics.median(figs_per):.0f}")
print(f"  total tables              : {n_tbl:,}   median/doc {statistics.median(tbls_per):.0f}")
print()
print(f"  figures WITH a caption    : {n_fig_caption:,} ({n_fig_caption/max(n_fig,1):.1%})")
print(f"  figures WITH an image ref : {n_fig_graphic:,} ({n_fig_graphic/max(n_fig,1):.1%})")
print(f"  tables with MACHINE-READABLE <table> markup (not a picture): "
      f"{n_tbl_inline:,} ({n_tbl_inline/max(n_tbl,1):.1%})")
if caption_lens:
    print(f"\n  caption length: median {statistics.median(caption_lens):.0f} chars, "
          f"mean {statistics.mean(caption_lens):.0f}, max {max(caption_lens):,}")
    print(f"  total caption text currently NOT indexed: "
          f"{sum(caption_lens):,} chars")

print("\n=== sample captions (these are real prose, and none of it is in the index) ===")
for pmc, label, cap, href in samples:
    print(f"\n  PMC{pmc}  [{label}]  image={href}")
    print(f"    {cap}")
