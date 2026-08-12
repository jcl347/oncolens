#!/usr/bin/env python
"""Can a VLM actually READ a scientific figure? Measured against the author's own reading.

**The argument this tests.** Kaplan-Meier, dose-response and forest plots encode their
meaning *positionally*: the result is where the lines sit relative to each other. Nothing in
the file says "the treatment arm had better survival". A text pipeline extracts
"Figure 3. Overall survival by treatment arm" and stops — it knows the figure exists and not
what it says. Layout detection finds the figure; reading it is a separate problem.

**Ground truth, found rather than written (§4.4).** When an author writes
"HR 0.62 (95% CI 0.48-0.81) (Figure 3)", 0.62 IS what Figure 3 shows — they read it off
their own plot. So the citing sentence is expert ground truth for the image, produced with
no annotation and no model.

**The control that makes it mean anything.** A VLM asked for "the hazard ratio" on an
oncology forest plot could answer 0.62 without looking, because 0.6-0.8 is where hazard
ratios live. So every figure is scored TWICE:

  * ``vision``   — the model sees the image and the caption
  * ``blind``    — the model sees ONLY the caption, and must guess

The vision-minus-blind gap is the only number that means anything. If they match, the model
is reading the prior, not the plot, and §3.13 applies: an invented number in the index
misroutes retrieval rather than merely failing to help.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oncolens.env import load_env, local_data_dir  # noqa: E402

MODEL = "gpt-4o-mini"
#: Values worth testing: not years, not 0/1, not single digits that appear everywhere.
NUM = re.compile(r"\b\d{1,4}\.\d+\b|\b\d{2,4}\b")

ASK = (
    "You are reading a figure from an oncology paper. Report ONLY values you can actually "
    "see plotted, printed, or labelled in the image. Do not infer, do not use typical "
    "values from the literature, and do not repeat numbers that appear only in the caption "
    "without confirming them in the image.\n"
    "Return strict JSON: {\"values\": [numbers you can read], \"direction\": \"<which "
    "group/arm is better, or 'none'>\", \"readable\": true|false}. "
    "If the image is too low-resolution or has no plotted numbers, set readable=false and "
    "values=[]."
)


def numbers(s: str) -> set[str]:
    out = set()
    for m in NUM.finditer(s or ""):
        v = m.group(0)
        f = float(v)
        if 1900 <= f <= 2035 and "." not in v:
            continue
        out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def call(key: str, caption: str, img_b64: str | None) -> dict:
    import requests

    content = [{"type": "text", "text": f"{ASK}\n\nCaption: {caption[:700]}"}]
    if img_b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
    else:
        content[0]["text"] += ("\n\nNOTE: the image is NOT provided. Give your best guess "
                               "of the values from the caption alone.")
    try:
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL, "max_tokens": 300, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": content}]}, timeout=120)
        if not r.ok:
            return {}
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--types", default="kaplan-meier,forest-plot,dose-response,roc-curve")
    args = ap.parse_args()

    load_env(REPO)
    import psycopg
    import requests

    key = os.environ["OPENAI_API_KEY"]
    want = [t.strip() for t in args.types.split(",")]
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")

    with psycopg.connect(dsn) as cx, cx.cursor() as cur:
        cur.execute("""SELECT chunk_id, figure_type, text, indexed_text, image_uri
                       FROM chunks
                       WHERE kind='figure' AND image_uri IS NOT NULL
                         AND figure_type = ANY(%s)""", (want,))
        rows = cur.fetchall()
    print(f"candidate figures of type {want}: {len(rows):,}")

    # Usable = the CITING text states a number the caption does not. That number is the
    # author reading their own plot, and the caption cannot have leaked it.
    usable = []
    for cid, ftype, cap, idx, uri in rows:
        cited = numbers((idx or "")[len(cap or ""):])
        target = cited - numbers(cap or "")
        target = {t for t in target if "." in t}     # effect sizes, not counts
        if target:
            usable.append((cid, ftype, cap, uri, target))
    print(f"  with a cited decimal absent from the caption: {len(usable):,}")
    if not usable:
        return 0

    random.seed(13)
    sample = random.sample(usable, min(args.n, len(usable)))
    print(f"  sampling {len(sample)}  ({dict(Counter(t for _c, t, *_ in sample))})\n")

    S = requests.Session()
    S.headers["User-Agent"] = "oncolens/1.0"

    def one(item):
        cid, ftype, cap, uri, target = item
        try:
            img = S.get(uri, timeout=60).content
            b64 = base64.b64encode(img).decode()
        except Exception:  # noqa: BLE001
            return None
        vis = call(key, cap, b64)
        blind = call(key, cap, None)
        return {"cid": cid, "type": ftype, "target": target,
                "vision": vis, "blind": blind}

    print("querying the VLM twice per figure (vision, then blind) ...")
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = [r for r in ex.map(one, sample) if r]
    print(f"  completed {len(res)}/{len(sample)}\n")

    def hit(out, target):
        got = {str(v).rstrip("0").rstrip(".") for v in (out.get("values") or [])
               if isinstance(v, (int, float))}
        return bool(got & target)

    per = defaultdict(lambda: {"n": 0, "vis": 0, "blind": 0, "unreadable": 0})
    for r in res:
        d = per[r["type"]]
        d["n"] += 1
        d["vis"] += hit(r["vision"], r["target"])
        d["blind"] += hit(r["blind"], r["target"])
        d["unreadable"] += (r["vision"].get("readable") is False)
    tot = {"n": 0, "vis": 0, "blind": 0, "unreadable": 0}
    for d in per.values():
        for k in tot:
            tot[k] += d[k]

    print("=== recovering the value the AUTHOR read off the plot ===")
    print(f"  {'type':<16}{'n':>5}{'vision':>9}{'blind':>9}{'gap':>8}{'unreadable':>12}")
    for t, d in sorted(per.items(), key=lambda x: -x[1]["n"]):
        v, b = d["vis"] / d["n"], d["blind"] / d["n"]
        print(f"  {t:<16}{d['n']:>5}{v:>8.0%}{b:>9.0%}{v-b:>+8.0%}"
              f"{d['unreadable']/d['n']:>11.0%}")
    v, b = tot["vis"] / max(tot["n"], 1), tot["blind"] / max(tot["n"], 1)
    print(f"  {'ALL':<16}{tot['n']:>5}{v:>8.0%}{b:>9.0%}{v-b:>+8.0%}"
          f"{tot['unreadable']/max(tot['n'],1):>11.0%}")

    print(f"\n  vision {v:.0%} vs blind {b:.0%}. The GAP is the only number that means")
    print("  anything: it is what the model gets from looking rather than from knowing")
    print("  where oncology effect sizes usually sit.")
    if v - b < 0.10:
        print("\n  -> Below 10 points, this is not reading. Indexing these values would be")
        print("     indexing a prior, and §3.13 says a wrong number MISROUTES retrieval")
        print("     rather than failing quietly. Do not index VLM numerals.")
    out = local_data_dir() / "vlm_figure_eval.json"
    out.write_text(json.dumps({"per_type": {k: dict(v) for k, v in per.items()},
                               "total": tot, "model": MODEL,
                               "n_usable": len(usable)}, indent=1), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
