"""How much of the figure opportunity does FREE caption indexing already capture?

This is the number that ranks the design space. For each in-text figure reference we have
a (sentence -> figure) pair the author wrote. The question for each pair is whether the
caption alone would let text retrieval find that figure:

  overlap high  -> indexing captions (no model, no GPU) already serves this query
  overlap low   -> only the IMAGE distinguishes this figure; that is the VLM/pixel budget

A second, sharper cut: numeric literals. If the sentence cites "56%" against Figure 1 and
"56%" is absent from the caption, the caption does not carry the quantitative content even
if the topic words match.

IMPORTANT SCOPE NOTE. In-text references are, by construction, the parts of a figure the
author DID write down in prose. So they measure *retrieval* (can we surface the right
figure) and they cannot measure pixel-only QA (the answer is in the sentence). Sizing the
pixel-only opportunity needs a different instrument; this script does not claim to.
"""
import sys, re, statistics, collections
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(r"c:\Users\jcl34\OneDrive\Documents\GitHub\oncolens-1")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from oncolens.env import load_env, local_data_dir
load_env(ROOT)

STOP = set("the a an of in on for to and or with by from as at is are was were be been "
           "this that these those we our it its their they which than then when while "
           "not no also both each per vs versus using used shown show shows figure fig "
           "table between among after before during over under more most less least".split())
NUM = re.compile(r"\d+(?:\.\d+)?%?")


def content(s):
    return {w for w in re.findall(r"[a-z0-9\-]{3,}", s.lower()) if w not in STOP}


cache = local_data_dir() / "jats_cache"
files = sorted(cache.glob("*.xml"))

pairs = 0
overlaps, num_pairs, num_covered = [], 0, 0
buckets = collections.Counter()
examples = {"low": [], "high": []}

for f in files:
    try:
        root = ET.parse(f).getroot()
    except Exception:
        continue
    caps = {}
    for fg in root.findall(".//fig"):
        fid = fg.get("id")
        if not fid:
            continue
        cap = " ".join("".join((fg.find("./caption") or ET.Element("x")).itertext()).split())
        caps[fid] = cap
    if not caps:
        continue

    for p in root.findall(".//p"):
        xs = [x for x in p.iter("xref") if x.get("ref-type") == "fig"]
        if not xs:
            continue
        text = " ".join("".join(p.itertext()).split())
        for s in re.split(r"(?<=[.!?])\s+", text):
            if not re.search(r"\b[Ff]ig", s):
                continue
            rids = {r for x in xs for r in (x.get("rid") or "").split() if r in caps}
            if len(rids) != 1:          # diffuse reference; 4.4 drops these
                continue
            rid = rids.pop()
            cap = caps[rid]
            if len(cap) < 40 or len(s) < 40:
                continue
            cs, cc = content(s), content(cap)
            if not cs:
                continue
            ov = len(cs & cc) / len(cs)
            overlaps.append(ov)
            pairs += 1
            b = ("0-20%" if ov < .2 else "20-40%" if ov < .4 else "40-60%" if ov < .6
                 else "60-80%" if ov < .8 else "80-100%")
            buckets[b] += 1

            ns = set(NUM.findall(s))
            if ns:
                num_pairs += 1
                if ns & set(NUM.findall(cap)):
                    num_covered += 1

            key = "low" if ov < 0.2 else ("high" if ov > 0.7 else None)
            if key and len(examples[key]) < 3:
                examples[key].append((f.stem, rid, round(ov, 2), s[:190], cap[:190]))

print(f"usable (sentence -> single figure) pairs: {pairs:,}\n")
print("=== how much of the reference sentence is already in the CAPTION ===")
for b in ("0-20%", "20-40%", "40-60%", "60-80%", "80-100%"):
    c = buckets[b]
    print(f"  overlap {b:<8} {c:>7,}  {c/max(pairs,1):>6.1%}  {'#'*int(52*c/max(pairs,1))}")
print(f"\n  median overlap: {statistics.median(overlaps):.1%}")
low = sum(v for k, v in buckets.items() if k in ("0-20%", "20-40%"))
print(f"  pairs where the caption shares <40% of the query's content words: "
      f"{low:,} ({low/max(pairs,1):.1%})")
print(f"  <- this is the slice captions CANNOT serve. It bounds every pixel-based approach.")

print(f"\n=== quantitative content ===")
print(f"  reference sentences containing a number : {num_pairs:,}")
print(f"  ...whose number also appears in the caption: {num_covered:,} "
      f"({num_covered/max(num_pairs,1):.1%})")
print(f"  ...number NOT in the caption               : {num_pairs-num_covered:,} "
      f"({1-num_covered/max(num_pairs,1):.1%})")
print("  <- the caption names the topic but not the value; the value is in the image")

for k, label in (("high", "CAPTION ALREADY SERVES THIS"), ("low", "ONLY THE IMAGE SERVES THIS")):
    print(f"\n=== {label} ===")
    for pmc, rid, ov, s, cap in examples[k]:
        print(f"\n  PMC{pmc} {rid}  overlap={ov}")
        print(f"    query   : {s}")
        print(f"    caption : {cap}")
