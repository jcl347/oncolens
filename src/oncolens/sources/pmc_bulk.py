"""PMC Open Access bulk ingestion — real full text, at scale.

Per-article API calls do not scale to a real corpus: NCBI asks for <= 3 requests/second,
so a million articles is roughly four days of polite requests. The bulk service is the
supported path and moves the same content in hours.

**Licensing is a product decision, not a detail.** The OA Subset is split three ways:

| Subset | Licenses | Use in a hosted product |
|---|---|---|
| ``oa_comm`` | CC0, CC BY, CC BY-SA, CC BY-ND | **Commercial use allowed** — use this one |
| ``oa_noncomm`` | CC BY-NC, CC BY-NC-SA, CC BY-NC-ND | Non-commercial only |
| ``oa_other`` | no machine-readable licence / custom | Review individually before use |

``DEFAULT_SUBSET`` is ``oa_comm`` deliberately: a deployed product that indexes
``oa_noncomm`` full text is a licensing problem, and defaulting to the permissive subset
makes the safe choice the easy one. Abstracts from PubMed are separately fine to index
regardless of subset, which is why metadata ingestion and full-text ingestion are
different modules.

**BioC is worth preferring where available.** The BioC distribution is already segmented
into passages *with character offsets*, which is exactly the provenance this product needs
— it removes the need to re-derive offsets after parsing.

This module is written for a machine with ordinary network access (a laptop, a GitHub
Action, a Vercel build step). It is not reachable from a sandbox whose shell has no egress.
"""

from __future__ import annotations

import csv
import io
import tarfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

BASE = "https://ftp.ncbi.nlm.nih.gov/pub/pmc"
OA_FILE_LIST = f"{BASE}/oa_file_list.csv"
OA_BULK = f"{BASE}/oa_bulk"

#: Commercial-use-permitted subset. See the licensing table above before changing this.
DEFAULT_SUBSET = "oa_comm"


@dataclass(frozen=True)
class OAEntry:
    file_path: str      # e.g. "oa_package/08/e0/PMC13900.tar.gz"
    citation: str
    pmcid: str
    pmid: str | None
    license: str | None

    @property
    def url(self) -> str:
        return f"{BASE}/{self.file_path}"


def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "oncolens/0.1 (research retrieval benchmark)"})
    return s


def stream_oa_file_list(*, subset_prefix: str | None = DEFAULT_SUBSET) -> Iterator[OAEntry]:
    """Stream the OA index (~hundreds of MB) without holding it in memory.

    Columns: File, Article Citation, Accession ID (PMCID), Last Updated, PMID, License.
    Streaming matters — this file is large enough that a naive ``.text`` read is a
    memory problem on a small build container.
    """
    s = _session()
    with s.get(OA_FILE_LIST, stream=True, timeout=300) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        reader = csv.reader(io.TextIOWrapper(r.raw, encoding="utf-8", errors="replace"))
        header = next(reader, None)
        if header is None:
            return
        idx = {name.strip().lower(): i for i, name in enumerate(header)}

        def get(row: list[str], *names: str) -> str | None:
            for n in names:
                j = idx.get(n)
                if j is not None and j < len(row) and row[j].strip():
                    return row[j].strip()
            return None

        for row in reader:
            if not row:
                continue
            path = get(row, "file")
            pmcid = get(row, "accession id", "accession_id", "pmcid")
            if not path or not pmcid:
                continue
            lic = get(row, "license")
            if subset_prefix and lic and not _license_allows(lic, subset_prefix):
                continue
            yield OAEntry(
                file_path=path,
                citation=get(row, "article citation") or "",
                pmcid=pmcid,
                pmid=get(row, "pmid"),
                license=lic,
            )


_COMMERCIAL_OK = ("CC0", "CC BY", "CC-BY")
_NONCOMMERCIAL = ("NC",)


def _license_allows(license_str: str, subset: str) -> bool:
    up = license_str.upper().replace("_", " ")
    is_nc = any(tok in up for tok in _NONCOMMERCIAL)
    if subset == "oa_comm":
        return (not is_nc) and any(tok in up for tok in _COMMERCIAL_OK)
    if subset == "oa_noncomm":
        return is_nc
    return True


def fetch_article_package(entry: OAEntry, *, dest: Path | None = None) -> dict:
    """Download and parse one article package (.tar.gz containing NXML + media).

    Returns a corpus document with real section structure. Use this for targeted pulls;
    for whole-corpus ingestion prefer :func:`iter_bulk_packages`.
    """
    from .europepmc import parse_jats

    s = _session()
    r = s.get(entry.url, timeout=300)
    r.raise_for_status()
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)

    sections: list[dict[str, str]] = []
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.endswith(".nxml"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            sections = parse_jats(f.read().decode("utf-8", errors="replace"))
            break

    return {
        "doc_id": f"PAPER:PMID{entry.pmid}" if entry.pmid else f"PAPER:{entry.pmcid}",
        "doc_type": "paper",
        "title": entry.citation,
        "year": None,
        "sections": sections,
        "meta": {"pmcid": entry.pmcid, "pmid": entry.pmid, "license": entry.license,
                 "source": "pmc_oa_bulk"},
        "descriptors": [],
        "funded_by": [],
        "cites": [],
    }


def iter_bulk_packages(
    entries: Iterable[OAEntry], *, workers: int = 8, on_error: str = "skip"
) -> Iterator[dict]:
    """Parallel download+parse of many article packages.

    ``workers`` is capped low on purpose. NCBI asks for restraint; a build job that
    hammers the FTP endpoint gets throttled or blocked, which is slower overall than
    staying polite. Failures are skipped rather than fatal — a single corrupt package
    must not abort a multi-hour ingestion.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    entries = list(entries)
    with ThreadPoolExecutor(max_workers=min(workers, 12)) as pool:
        futures = {pool.submit(fetch_article_package, e): e for e in entries}
        for fut in as_completed(futures):
            try:
                doc = fut.result()
            except Exception:
                if on_error == "raise":
                    raise
                continue
            if doc.get("sections"):
                yield doc


def attach_mesh(docs: Iterable[dict], *, email: str | None = None,
                api_key: str | None = None, batch: int = 100) -> list[dict]:
    """Attach real NLM MeSH indexing to bulk-fetched full-text documents.

    The bulk packages carry full text but not MeSH headings, and MeSH is the label source
    the whole evaluation depends on. This joins the two by PMID so a corpus built from
    bulk full text still comes with human-assigned relevance labels.
    """
    from . import pubmed

    docs = list(docs)
    by_pmid = {
        (d.get("meta") or {}).get("pmid"): d
        for d in docs
        if (d.get("meta") or {}).get("pmid")
    }
    pmids = [p for p in by_pmid if p]
    if not pmids:
        return docs

    for rec in pubmed.efetch(pmids, batch=batch, email=email, api_key=api_key):
        d = by_pmid.get(rec.pmid)
        if not d:
            continue
        d["descriptors"] = [f"MESH:{m['descriptor']}" for m in rec.mesh]
        d["mesh_detail"] = rec.mesh
        d["title"] = d.get("title") or rec.title
        d["year"] = d.get("year") or rec.year
        d["meta"]["journal"] = rec.journal
        d["meta"]["grants"] = rec.grants
        d["funded_by"] = [f"GRANT:{g['grant_id']}" for g in rec.grants if g.get("grant_id")]
        if rec.abstract and not any(s["name"] == "Abstract" for s in d["sections"]):
            d["sections"] = [{"name": "Abstract", "text": rec.abstract}] + d["sections"]
    return docs
