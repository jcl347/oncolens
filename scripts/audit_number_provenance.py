"""The 18.9%: numbers cited against a figure that are NOT in its caption. Where ARE they?

Before reaching for a vision model, find out whether the value is recoverable from text
somewhere else in the same paper. "Table extraction first" says: if the number exists in a
table, never read it off a chart. This measures how often that applies.

For each (sentence -> figure) pair whose numeric literals are absent from the caption, look
for those literals in:
  1. the paper's <table-wrap> markup  (free, structured, exact)
  2. any other caption in the paper   (free)
  3. the body text outside this sentence (free)
  4. nowhere in the XML at all        <- ONLY THIS needs pixels
"""
import sys, re, collections, random
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(r"c:\Users\jcl34\OneDrive\Documents\GitHub\oncolens-1")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from oncolens.env import load_env, local_data_dir
load_env(ROOT)

# Values worth chasing: not years, not tiny counts, not reference numbers.
NUM = re.compile(r"\b\d{1,4}(?:\.\d+)?\b")


def nums(s):
    out = set()
    for m in NUM.finditer(s):
        v = m.group(0)
        f = float(v)
        if 1900 <= f <= 2035 and "." not in v:   # a year
            continue
        if f < 2 and "." not in v:               # 0/1, usually structural
            continue
        out.add(v)
    return out


cache = local_data_dir() / "jats_cache"
files = sorted(cache.glob("*.xml"))
random.seed(7)
files = random.sample(files, min(700, len(files)))

where = collections.Counter()
total = 0
pixel_examples = []

for f in files:
    try:
        root = ET.parse(f).getroot()
    except Exception:
        continue
    caps, tbls = {}, []
    for fg in root.findall(".//fig"):
        fid = fg.get("id")
        ce = fg.find("./caption")
        if fid and ce is not None:
            caps[fid] = " ".join("".join(ce.itertext()).split())
    if not caps:
        continue
    for tw in root.findall(".//table-wrap"):
        tbls.append(" ".join("".join(tw.itertext()).split()))
    table_nums = set()
    for t in tbls:
        table_nums |= nums(t)
    all_caption_nums = set()
    for c in caps.values():
        all_caption_nums |= nums(c)

    body_all = " ".join("".join(p.itertext()) for p in root.findall(".//p"))

    for p in root.findall(".//p"):
        xs = [x for x in p.iter("xref") if x.get("ref-type") == "fig"]
        if not xs:
            continue
        text = " ".join("".join(p.itertext()).split())
        for s in re.split(r"(?<=[.!?])\s+", text):
            if not re.search(r"\b[Ff]ig", s):
                continue
            rids = {r for x in xs for r in (x.get("rid") or "").split() if r in caps}
            if len(rids) != 1:
                continue
            rid = rids.pop()
            own = nums(caps[rid])
            for v in nums(s):
                if v in own:
                    continue                      # already covered by 81.1%
                total += 1
                if v in table_nums:
                    where["in a TABLE in the same paper"] += 1
                elif v in all_caption_nums:
                    where["in ANOTHER figure's caption"] += 1
                elif body_all.count(v) > 1:
                    where["elsewhere in the body text"] += 1
                else:
                    where["NOWHERE in the XML -> needs pixels"] += 1
                    if len(pixel_examples) < 5 and len(s) < 210:
                        pixel_examples.append((f.stem, rid, v, s))

print(f"numeric literals cited against a figure but ABSENT from its caption: {total:,}\n")
for k, n in where.most_common():
    print(f"  {n:>7,}  {n/max(total,1):>6.1%}  {k}")

pix = where["NOWHERE in the XML -> needs pixels"]
print(f"\n  recoverable from TEXT already in the XML : {total-pix:,} ({1-pix/max(total,1):.1%})")
print(f"  genuinely pixel-only                     : {pix:,} ({pix/max(total,1):.1%})")

print("\n--- values that exist nowhere but in the image ---")
for pmc, rid, v, s in pixel_examples:
    print(f"\n  PMC{pmc} {rid}  value={v}")
    print(f"    {s[:200]}")
