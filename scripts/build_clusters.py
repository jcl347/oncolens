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
            "       (SELECT count(*) FROM chunks c WHERE c.doc_id = d.doc_id "
            "        AND c.kind = 'passage') "
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
        cur.execute("SELECT doc_id, embedding FROM chunks WHERE kind = 'passage'")
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


def variance_captured(Xc: np.ndarray, W: np.ndarray) -> float:
    """Fraction of total variance retained by projecting onto orthonormal rows of ``W``.

    Works for any orthonormal basis, not just the leading PCA axes, so the discriminant
    projection below can be reported on the same scale as the PCA one.
    """
    num = float(np.sum((Xc @ W.T) ** 2))
    den = float(np.sum(Xc ** 2))
    return num / den if den else 0.0


def between_cluster_ratio(Xc: np.ndarray, labels: np.ndarray) -> float:
    """BSS/TSS — how much of the total variance the CLUSTER ASSIGNMENT accounts for.

    This is the number people think "explained variance" means when they look at a cluster
    map, and it is not the projection's number. The projection can only ever show a shadow
    of the space; this says how much real structure the partition itself captures,
    independent of any picture.
    """
    tss = float(np.sum(Xc ** 2))
    if not tss:
        return 0.0
    bss = 0.0
    for j in np.unique(labels):
        members = Xc[labels == j]
        bss += len(members) * float(np.sum(members.mean(axis=0) ** 2))
    return bss / tss


def discriminant_axes(Xc: np.ndarray, labels: np.ndarray, n_axes: int = 3) -> np.ndarray:
    """Orthonormal axes maximising BETWEEN-cluster spread, not total spread.

    **Why this is offered alongside plain PCA, and why it is still honest.** PCA picks the
    directions of greatest total variance, which in a 192-dimensional embedding space is
    dominated by within-cluster spread — hence only ~14% in two dimensions, and a picture
    where real research areas overlap into a blob. Running PCA on the *cluster centroids*
    instead picks the plane in which the clusters are furthest apart.

    This is a **linear** map, exactly like plain PCA: a rotation and projection of the same
    space, so relative distances remain a true shadow rather than a warped one. That is the
    difference from t-SNE, which reshapes local neighbourhoods and makes distances and
    cluster sizes meaningless.

    It is not free of choice, though: these axes are selected *after* seeing the cluster
    labels, so they show the partition at its most flattering. That is a camera angle, not
    an invention — and it is why the total-variance figure is still reported for this
    projection, and why the UI labels which projection it is showing.
    """
    cents = []
    weights = []
    for j in np.unique(labels):
        members = Xc[labels == j]
        cents.append(members.mean(axis=0))
        weights.append(len(members))
    Cm = np.asarray(cents) * np.sqrt(np.asarray(weights))[:, None]
    # Centroids already live in the centred space, so no second centring.
    _, _, Vt = np.linalg.svd(Cm, full_matrices=False)
    return Vt[:n_axes]


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

    # ---- projections -------------------------------------------------------
    # Both are LINEAR maps of the same space, so distances stay comparable. Three
    # dimensions rather than two because the third axis is free structure: it costs one
    # float per point and recovers variance that a flat map simply discards.
    Xc = Xk - Xk.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)

    pca_axes3 = Vt[:3]
    disc_axes3 = discriminant_axes(Xc, labels, n_axes=3)

    def project(W: np.ndarray) -> np.ndarray:
        c = Xc @ W.T
        lo, hi = c.min(axis=0), c.max(axis=0)
        span = np.where(hi - lo == 0, 1, hi - lo)
        return (c - lo) / span * 2 - 1          # -> [-1, 1] per axis

    norm = project(pca_axes3)
    norm_disc = project(disc_axes3)

    var_pca2 = float((S[:2] ** 2).sum() / (S ** 2).sum())
    var_pca3 = float((S[:3] ** 2).sum() / (S ** 2).sum())
    var_disc3 = variance_captured(Xc, disc_axes3)
    bss = between_cluster_ratio(Xc, labels)
    # How much of the SEPARATION BETWEEN CLUSTERS each projection preserves — the
    # question the map is actually asked, and the one plain total-variance understates.
    cent = np.vstack([Xc[labels == j].mean(axis=0) for j in np.unique(labels)])
    w = np.array([(labels == j).sum() for j in np.unique(labels)], dtype=float)
    bss_total = float(np.sum(w[:, None] * cent ** 2))
    bss_pca = float(np.sum(w[:, None] * (cent @ pca_axes3.T) ** 2)) / (bss_total or 1)
    bss_disc = float(np.sum(w[:, None] * (cent @ disc_axes3.T) ** 2)) / (bss_total or 1)
    explained = var_pca3

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
        centroid_d = norm_disc[labels == j].mean(axis=0)
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
            "z": round(float(centroid[2]), 4),
            "dx": round(float(centroid_d[0]), 4),
            "dy": round(float(centroid_d[1]), 4),
            "dz": round(float(centroid_d[2]), 4),
            "papers": [{
                "doc_id": members[i]["doc_id"], "title": members[i]["title"][:140],
                "year": members[i]["year"], "pmid": members[i]["pmid"],
                "pmcid": members[i]["pmcid"],
            } for i in order],
        })
    clusters.sort(key=lambda c: -c["size"])

    points = [{"x": round(float(norm[i][0]), 4), "y": round(float(norm[i][1]), 4),
               "z": round(float(norm[i][2]), 4),
               "dx": round(float(norm_disc[i][0]), 4),
               "dy": round(float(norm_disc[i][1]), 4),
               "dz": round(float(norm_disc[i][2]), 4),
               "c": int(labels[i])} for i in range(len(docs_k))]

    payload = {
        "k": args.k, "n_documents": len(docs_k),
        # Kept for backwards compatibility with anything reading the old key.
        "explained_variance": round(explained, 4),
        "variance": {
            "pca_2d": round(var_pca2, 4),
            "pca_3d": round(var_pca3, 4),
            "discriminant_3d": round(var_disc3, 4),
            "cluster_bss": round(bss, 4),
            "between_cluster_kept_pca": round(bss_pca, 4),
            "between_cluster_kept_discriminant": round(bss_disc, 4),
            "source_dim": int(Xk.shape[1]),
        },
        "projection": "PCA (linear — distances are comparable, unlike t-SNE/UMAP)",
        "clusters": clusters, "points": points,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")

    print(f"\nVARIANCE (source space {Xk.shape[1]}-dim)")
    print(f"  PCA 2-D                          {var_pca2:>7.1%}")
    print(f"  PCA 3-D                          {var_pca3:>7.1%}   <- third axis is free structure")
    print(f"  discriminant 3-D                 {var_disc3:>7.1%}")
    print(f"  cluster assignment (BSS/TSS)     {bss:>7.1%}   <- what the PARTITION captures")
    print(f"  between-cluster kept, PCA        {bss_pca:>7.1%}")
    print(f"  between-cluster kept, discrim.   {bss_disc:>7.1%}   <- what the map is asked to show")
    print(f"\n{'size':>6}  cluster label / distinctive MeSH terms")
    print("-" * 76)
    for c in clusters:
        print(f"{c['size']:>6}  {', '.join(c['terms']) or c['label']}")
    print(f"\nwrote {out} ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
