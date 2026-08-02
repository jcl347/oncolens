#!/usr/bin/env python
"""Cluster the corpus into research areas, and label them from NLM's own indexing.

**What makes a cluster meaningful rather than decorative.** A scatter of pretty dots is
worthless if a reader cannot say what a region *is*. So two things are separated here:

* **Position** comes from the document embeddings — the same vectors retrieval uses. Two
  papers sit near each other on the map for exactly the reason they would both be returned
  by the same query. The picture is therefore a picture of the retrieval space, not an
  illustration of one.
* **Labels** come from MeSH major-topic assignments, chosen by *distinctiveness* rather
  than frequency. "Humans" is the most common descriptor in almost every cluster and says
  nothing; the label a cluster deserves is the term that is common *inside* it and rare
  outside. That is a log-odds ratio, and it is what separates "Immunotherapy, Adoptive"
  from "Humans".

**Projection is 2-D PCA, not t-SNE or UMAP.** Those produce prettier separation and
*invent* it: distances in a t-SNE plot do not correspond to distances in the source space,
and cluster sizes are meaningless. Since the whole point is to show the space retrieval
actually operates in, a linear projection that preserves relative distance is the honest
choice even though it looks less dramatic. The explained-variance ratio is reported so a
reader knows how much of the structure the picture is showing.

    python scripts/build_clusters.py --k 14
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402

from oncolens.env import load_env, local_data_dir  # noqa: E402
from oncolens.eval.strata import GENERIC_DESCRIPTORS  # noqa: E402


def load_documents() -> tuple[list[dict], dict[str, list[str]]]:
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("POSTGRES_URL / DATABASE_URL not set")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT doc_id, title, year, descriptors, meta, "
            "       (SELECT count(*) FROM chunks c WHERE c.doc_id = d.doc_id) "
            "FROM documents d ORDER BY doc_id")
        docs, majors = [], {}
        for doc_id, title, year, descs, meta, n_chunks in cur.fetchall():
            maj = []
            if isinstance(meta, dict):
                for m in (meta.get("mesh") or []):
                    if isinstance(m, dict) and m.get("major") and m.get("descriptor"):
                        maj.append(m["descriptor"])
            docs.append({"doc_id": doc_id, "title": title or "", "year": year,
                         "pmid": (doc_id or "").replace("PAPER:PMID", ""),
                         "pmcid": (meta or {}).get("pmcid"),
                         "n_chunks": n_chunks})
            majors[doc_id] = maj or [d.replace("MESH:", "") for d in (descs or [])][:6]
    return docs, majors


def document_vectors(doc_ids: list[str]) -> np.ndarray:
    """Mean of a document's passage vectors, read from the store.

    Averaging passages is the same operation the retriever's document-level view implies,
    so a document's position on the map is where retrieval actually places it.
    """
    import psycopg

    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    idx = {d: i for i, d in enumerate(doc_ids)}
    dim = None
    acc: dict[int, list] = {}
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, embedding FROM chunks")
        for doc_id, emb in cur.fetchall():
            i = idx.get(doc_id)
            if i is None or emb is None:
                continue
            v = np.fromstring(str(emb).strip("[]"), sep=",")
            if dim is None:
                dim = len(v)
            acc.setdefault(i, []).append(v)
    out = np.zeros((len(doc_ids), dim or 192))
    for i, vs in acc.items():
        m = np.mean(vs, axis=0)
        n = np.linalg.norm(m)
        out[i] = m / n if n else m
    return out


def kmeans(X: np.ndarray, k: int, iters: int = 60, seed: int = 7):
    """Spherical k-means. Deterministic seeding so the map is stable between builds."""
    rng = np.random.default_rng(seed)
    # k-means++ style seeding: spread initial centres out rather than clumping.
    centres = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = 1.0 - (X @ np.array(centres).T).max(axis=1)
        d = np.clip(d, 0, None)
        probs = d / d.sum() if d.sum() > 0 else None
        centres.append(X[rng.choice(len(X), p=probs)])
    C = np.array(centres)
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        sims = X @ C.T
        new = sims.argmax(axis=1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                m = members.mean(axis=0)
                n = np.linalg.norm(m)
                C[j] = m / n if n else m
    return labels, C


def distinctive_terms(members: list[str], majors: dict[str, list[str]],
                      global_counts: Counter, total_docs: int, top: int = 4) -> list[str]:
    """Terms common INSIDE the cluster and rare outside it.

    Plain frequency labels every cluster "Humans". Log-odds against the corpus background
    is what makes a label informative.
    """
    local = Counter()
    for d in members:
        for t in majors.get(d, []):
            if t and t not in GENERIC_DESCRIPTORS:
                local[t] += 1
    scored = []
    for term, c in local.items():
        if c < 2:
            continue
        p_in = c / max(len(members), 1)
        p_out = max(global_counts.get(term, 0) - c, 0.5) / max(total_docs - len(members), 1)
        scored.append((math.log(p_in / p_out), c, term))
    scored.sort(reverse=True)
    return [t for _, _, t in scored[:top]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=14)
    ap.add_argument("--out", default=str(ROOT / "public" / "clusters.json"))
    args = ap.parse_args()
    load_env()

    docs, majors = load_documents()
    print(f"{len(docs)} documents")
    ids = [d["doc_id"] for d in docs]
    X = document_vectors(ids)
    keep = np.linalg.norm(X, axis=1) > 0
    print(f"{keep.sum()} have passage vectors")
    Xk = X[keep]
    docs_k = [d for d, k in zip(docs, keep) if k]

    labels, C = kmeans(Xk, args.k)

    # PCA to 2-D. Linear, so distances on the map mean something.
    Xc = Xk - Xk.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ Vt[:2].T
    explained = float((S[:2] ** 2).sum() / (S ** 2).sum())
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    norm = (coords - lo) / span * 2 - 1        # -> [-1, 1]

    global_counts = Counter()
    for d in ids:
        for t in majors.get(d, []):
            global_counts[t] += 1

    clusters = []
    for j in range(args.k):
        members = [d for d, lab in zip(docs_k, labels) if lab == j]
        if not members:
            continue
        terms = distinctive_terms([m["doc_id"] for m in members], majors,
                                  global_counts, len(docs))
        centroid = norm[labels == j].mean(axis=0)
        # Representative papers: closest to the centroid in the ORIGINAL space, not the
        # projection — the projection is for display, not for deciding what is typical.
        sims = Xk[labels == j] @ C[j]
        order = np.argsort(-sims)[:6]
        clusters.append({
            "id": j,
            "label": terms[0] if terms else f"Cluster {j}",
            "terms": terms,
            "size": len(members),
            "x": round(float(centroid[0]), 4),
            "y": round(float(centroid[1]), 4),
            "papers": [{
                "doc_id": members[i]["doc_id"], "title": members[i]["title"][:140],
                "year": members[i]["year"], "pmid": members[i]["pmid"],
                "pmcid": members[i]["pmcid"],
            } for i in order],
        })
    clusters.sort(key=lambda c: -c["size"])

    points = [{"x": round(float(norm[i][0]), 4), "y": round(float(norm[i][1]), 4),
               "c": int(labels[i])} for i in range(len(docs_k))]

    payload = {"k": args.k, "n_documents": len(docs_k),
               "explained_variance": round(explained, 4),
               "projection": "PCA (linear — distances are comparable, unlike t-SNE/UMAP)",
               "clusters": clusters, "points": points}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")

    print(f"\n2-D projection explains {explained:.1%} of variance")
    print(f"\n{'size':>6}  cluster label / distinctive MeSH terms")
    print("-" * 76)
    for c in clusters:
        print(f"{c['size']:>6}  {', '.join(c['terms']) or c['label']}")
    print(f"\nwrote {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
