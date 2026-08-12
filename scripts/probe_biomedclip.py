"""Can BiomedCLIP classify a figure's TYPE from the image alone?

The argument so far is that BiomedCLIP cannot read values off a plot (224x224 destroys the
glyphs) but might still extract *type*, which is the gist task CLIP is built for. That is an
empirical claim and this session has repeatedly shown my inferences need checking.

Ground truth is caption-derived and therefore noisy: a caption saying "kaplan-meier" is
strong evidence the figure IS one, but a caption that does not say so is weak evidence it is
not. So only figures whose caption matches EXACTLY ONE type pattern are used, and the score
is reported as agreement with a noisy reference rather than as accuracy.

If BiomedCLIP can type figures, that is a real capability captions cover only partially --
my caption regex left 40.5% unclassified.
"""
import sys, re, io, json, random, collections, time
from pathlib import Path
from xml.etree import ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(r"c:\Users\jcl34\OneDrive\Documents\GitHub\oncolens-1")
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from oncolens.env import load_env, local_data_dir
load_env(ROOT)
from oncolens.sources import pmc_cloud as pc
import requests
from PIL import Image

# ---- caption patterns -> the reference label -------------------------------------------
TYPES = {
    "kaplan-meier survival curve": r"\b(kaplan[\s-]?meier|survival curve)",
    "western blot":                r"\b(western blot|immunoblot)",
    "flow cytometry plot":         r"\b(flow cytometr|facs)",
    "forest plot":                 r"\bforest plot",
    "bar chart":                   r"\bbar (?:chart|plot|graph)",
    "heatmap":                     r"\bheat ?map",
    "microscopy image":            r"\b(immunohistochem|micrograph|h&e|histolog)",
    "schematic diagram":           r"\b(schematic|flow ?chart|study design|graphical abstract)",
}
# Prompts fed to the text tower. Deliberately plain -- prompt engineering here would be
# tuning on the test set.
PROMPTS = {k: f"a {k} from a biomedical research paper" for k in TYPES}

cache = local_data_dir() / "jats_cache"
XLINK = "{http://www.w3.org/1999/xlink}href"

# ---- build a labelled sample -----------------------------------------------------------
items = []
files = sorted(cache.glob("*.xml"))
random.seed(3)
random.shuffle(files)
for f in files:
    if len(items) >= 400:
        break
    try:
        root = ET.parse(f).getroot()
    except Exception:
        continue
    pmcid = "PMC" + re.sub(r"\D", "", f.stem)
    for fg in root.findall(".//fig"):
        ce = fg.find("./caption")
        g = fg.find(".//graphic")
        if ce is None or g is None or not g.get(XLINK):
            continue
        cap = " ".join("".join(ce.itertext()).split())
        hits = [k for k, p in TYPES.items() if re.search(p, cap, re.I)]
        if len(hits) != 1:          # ambiguous or unlabelled -> unusable as reference
            continue
        items.append({"pmcid": pmcid, "file": g.get(XLINK), "label": hits[0],
                      "caption": cap[:120]})
print(f"caption-unambiguous figures found: {len(items)}")
by = collections.Counter(i["label"] for i in items)
print("  reference distribution:", dict(by.most_common()))

# cap per class so one type cannot dominate the score
per, capped = collections.Counter(), []
for i in items:
    if per[i["label"]] < 18:
        per[i["label"]] += 1
        capped.append(i)
items = capped
print(f"  balanced sample: {len(items)}  {dict(collections.Counter(i['label'] for i in items))}\n")

# ---- fetch the images ------------------------------------------------------------------
S = requests.Session()
S.headers["User-Agent"] = "oncolens/1.0 (jcl347@cornell.edu)"
meta_cache = {}


def media_url(pmcid, fname):
    if pmcid not in meta_cache:
        m = None
        for v in (1, 2, 3):
            m = pc.fetch_metadata(pmcid, v)
            if m:
                break
        meta_cache[pmcid] = m or {}
    for u in (meta_cache[pmcid].get("media_urls") or []):
        if fname.split("/")[-1] in u:
            return u.replace("s3://pmc-oa-opendata/", f"{pc.HTTPS_BASE}/")
    return None


def grab(i):
    try:
        u = media_url(i["pmcid"], i["file"])
        if not u:
            return None
        r = S.get(u, timeout=40)
        if not r.ok:
            return None
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        return {**i, "image": im}
    except Exception:
        return None


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as ex:
    got = [x for x in ex.map(grab, items) if x]
print(f"fetched {len(got)}/{len(items)} images in {time.perf_counter()-t0:.0f}s\n")
if len(got) < 30:
    print("too few images to score"); raise SystemExit(0)

# ---- BiomedCLIP zero-shot --------------------------------------------------------------
import torch, open_clip

MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
print(f"loading {MODEL} ...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL)
tok = open_clip.get_tokenizer(MODEL)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev).eval()

labels = list(PROMPTS)
with torch.no_grad():
    tfeat = model.encode_text(tok([PROMPTS[k] for k in labels], context_length=256).to(dev))
    tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    print(f"  text embedding dim: {tfeat.shape[1]}")

    px = torch.stack([preprocess(g["image"]) for g in got]).to(dev)
    ifeat = model.encode_image(px)
    ifeat = ifeat / ifeat.norm(dim=-1, keepdim=True)
    sims = (ifeat @ tfeat.T).cpu()

pred = [labels[i] for i in sims.argmax(dim=1).tolist()]
top2 = [[labels[j] for j in row.topk(2).indices.tolist()] for row in sims]

correct = sum(p == g["label"] for p, g in zip(pred, got))
top2ok = sum(g["label"] in t for t, g in zip(top2, got))
n = len(got)
print(f"\n=== zero-shot figure typing, n={n}, {len(labels)} classes (chance = {1/len(labels):.1%}) ===")
print(f"  top-1 agreement with caption reference : {correct}/{n} = {correct/n:.1%}")
print(f"  top-2 agreement                         : {top2ok}/{n} = {top2ok/n:.1%}")

print("\n  per class:")
cm = collections.defaultdict(collections.Counter)
for p, g in zip(pred, got):
    cm[g["label"]][p] += 1
for lab in labels:
    tot = sum(cm[lab].values())
    if not tot:
        continue
    hit = cm[lab][lab]
    worst = [f"{k}:{v}" for k, v in cm[lab].most_common(3) if k != lab]
    print(f"    {lab:<30} {hit:>2}/{tot:<3} {hit/tot:>6.0%}   confused with: {', '.join(worst) or '-'}")
