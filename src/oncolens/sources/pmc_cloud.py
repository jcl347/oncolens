"""PMC Cloud Service — real full text at scale (replaces the retiring FTP path).

**Time-critical.** NCBI is retiring the legacy PMC Article Datasets distribution. Per the
PMC OA Web Service page: *"On or after August 24 the legacy PMC Article Datasets files —
including the PMC OA Web Service API — will no longer be available."* Legacy objects were
already moved under ``s3://pmc-oa-opendata/deprecated/``. Anything built on
``ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/...`` or ``oa.fcgi`` stops working.

The replacement is the **PMC Cloud Service**:

* bucket ``pmc-oa-opendata`` in ``us-east-1``
* **world-readable, no AWS account required** (``--no-sign-request``, or plain HTTPS)
* formats per article: JATS XML, **plain text extracted from the XML**, PDF, JSON metadata
* subsets: ``oa_comm`` (commercial use OK), ``oa_noncomm``, ``author_manuscript``
* per-article ``metadata/PMC<id>.<ver>.json`` is the authoritative entry point

**Layout was verified by probing, not assumed.** NCBI reorganised this bucket during the
migration and the documented ``oa_comm/txt/...`` paths now 404. The live layout (confirmed
2026-08-01) is per-article version directories at the bucket root plus a ``metadata/``
prefix. Ingestion is driven from the metadata JSON, which carries the exact object URLs,
the licence code and the retraction flag - so nothing depends on a path convention that
has already changed once.

**Prefer the ``txt`` rendition.** NCBI extracts plain text from the JATS themselves, so
using it removes an entire XML-parsing failure mode. Fall back to ``xml`` when a record
has no text rendition, or when you need section structure that the flat text loses.

Licensing is still a product decision — see ``DEFAULT_SUBSET`` and the table in
``docs/DEPLOYMENT.md``. ``oa_comm`` is the default because indexing non-commercial full
text in a hosted product is a licensing problem.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

BUCKET = "pmc-oa-opendata"
REGION = "us-east-1"
HTTPS_BASE = f"https://{BUCKET}.s3.{REGION}.amazonaws.com"

#: Commercial-use-permitted subset (CC0 / CC BY / CC BY-SA / CC BY-ND).
DEFAULT_SUBSET = "oa_comm"
SUBSETS = ("oa_comm", "oa_noncomm", "author_manuscript")

#: VERIFIED bucket layout (probed anonymously 2026-08-01, post-reorganisation):
#:
#:   metadata/PMC<id>.<ver>.json          per-article metadata, incl. text/xml/pdf URLs
#:   PMC<id>.<ver>/PMC<id>.<ver>.txt      plain text extracted from JATS by NCBI
#:   PMC<id>.<ver>/PMC<id>.<ver>.xml      JATS XML
#:   PMC<id>.<ver>/PMC<id>.<ver>.pdf      PDF
#:   oa_comm/ , oa_noncomm/               OLD organisation, now effectively empty
#:
#: The metadata JSON is the authoritative entry point: it carries `license_code`,
#: `is_pmc_openaccess`, `is_retracted` and the exact object URLs, so nothing has to be
#: guessed from a path convention that NCBI has already changed once.
METADATA_KEY = "metadata/{pmcid}.{version}.json"
TEXT_KEY = "{pmcid}.{version}/{pmcid}.{version}.txt"
XML_KEY = "{pmcid}.{version}/{pmcid}.{version}.xml"

#: license_code values that permit commercial use in a hosted product.
COMMERCIAL_LICENSES = frozenset({"CC0", "CCBY", "CC BY", "CCBYSA", "CC BY-SA", "CCBYND", "CC BY-ND"})


@dataclass(frozen=True)
class CloudArticle:
    key: str            # S3 object key
    pmcid: str
    pmid: str | None
    license: str | None
    last_updated: str | None

    @property
    def https_url(self) -> str:
        return f"{HTTPS_BASE}/{self.key}"

    @property
    def s3_uri(self) -> str:
        return f"s3://{BUCKET}/{self.key}"


def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "oncolens/0.1 (research retrieval benchmark)"})
    return s


def list_prefix(prefix: str, *, max_keys: int = 1000, delimiter: str = "") -> list[str]:
    """Anonymous ListObjectsV2 over HTTPS. No credentials, no boto3.

    Useful for discovering the current layout when the inventory path is unknown.
    """
    from xml.etree import ElementTree as ET

    s = _session()
    keys: list[str] = []
    token = None
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": str(min(max_keys, 1000))}
        if delimiter:
            params["delimiter"] = delimiter
        if token:
            params["continuation-token"] = token
        r = s.get(HTTPS_BASE, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", ns):
            k = c.findtext("s3:Key", namespaces=ns)
            if k:
                keys.append(k)
        for cp in root.findall("s3:CommonPrefixes", ns):
            p = cp.findtext("s3:Prefix", namespaces=ns)
            if p:
                keys.append(p)
        if root.findtext("s3:IsTruncated", namespaces=ns) != "true" or len(keys) >= max_keys:
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=ns)
        if not token:
            break
    return keys[:max_keys]


def discover_pmcids(prefix: str = "metadata/", limit: int = 1000) -> list[tuple[str, int]]:
    """Walk the metadata prefix to discover (pmcid, version) pairs.

    Only for exploration or a full-corpus crawl. For a topical corpus, drive ingestion
    from a PubMed MeSH query instead - it is far cheaper than walking ~6M objects.
    """
    out: list[tuple[str, int]] = []
    for key in list_prefix(prefix, max_keys=limit):
        m = re.match(r"metadata/(PMC\d+)\.(\d+)\.json$", key)
        if m:
            out.append((m.group(1), int(m.group(2))))
    return out


def fetch_metadata(pmcid: str, version: int = 1) -> dict | None:
    """Per-article metadata JSON. Returns None if absent (not every version exists)."""
    s = _session()
    key = METADATA_KEY.format(pmcid=pmcid, version=version)
    r = s.get(f"{HTTPS_BASE}/{key}", timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def fetch_full_text(pmcid: str, version: int = 1) -> str | None:
    """Real, verbatim, plain-text full article. Returns None if unavailable."""
    s = _session()
    key = TEXT_KEY.format(pmcid=pmcid, version=version)
    r = s.get(f"{HTTPS_BASE}/{key}", timeout=180)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    r.encoding = r.encoding or "utf-8"
    return r.text


def commercial_use_ok(meta: dict) -> bool:
    """Licence gate. Applied before anything is indexed into a hosted product."""
    code = (meta.get("license_code") or "").upper().replace("-", "").replace(" ", "")
    return code in {c.upper().replace("-", "").replace(" ", "") for c in COMMERCIAL_LICENSES}


def fetch_text(article: CloudArticle) -> str:
    """Fetch one article's plain text over anonymous HTTPS."""
    s = _session()
    r = s.get(article.https_url, timeout=120)
    r.raise_for_status()
    return r.text


def to_corpus_doc(article: CloudArticle, text: str) -> dict:
    """Map plain text into the corpus schema, splitting on blank-line paragraphs.

    NCBI's text rendition is flat, so real section headings are not recoverable. Rather
    than invent them, everything goes into one ``Body`` section with paragraph breaks
    preserved — the chunker aggregates whole paragraphs, so passage retrieval still works
    and offsets remain exact. Use the ``xml`` rendition when section structure matters.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]
    title = paragraphs[0][:300] if paragraphs else article.pmcid
    body = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else text
    return {
        "doc_id": f"PAPER:PMID{article.pmid}" if article.pmid else f"PAPER:{article.pmcid}",
        "doc_type": "paper",
        "title": title,
        "year": None,
        "sections": [{"name": "Body", "text": body}],
        "meta": {
            "pmcid": article.pmcid, "pmid": article.pmid, "license": article.license,
            "source": "pmc_cloud", "s3_uri": article.s3_uri,
        },
        "funded_by": [],
        "cites": [],
    }


def ingest(
    pmcids: Iterable[str] | None = None,
    *,
    subset: str = DEFAULT_SUBSET,
    rendition: str = "txt",
    limit: int | None = None,
    workers: int = 8,
) -> Iterator[dict]:
    """Stream corpus documents from the PMC Cloud Service.

    ``pmcids`` restricts ingestion to a set (e.g. the result of a PubMed MeSH query);
    omit it to walk the whole subset. ``workers`` is capped low deliberately — a build job
    that hammers the endpoint gets throttled, which is slower overall than staying polite.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    wanted = {p.upper() for p in pmcids} if pmcids else None
    selected: list[CloudArticle] = []
    for art in stream_inventory(subset, rendition):
        if wanted is not None and art.pmcid.upper() not in wanted:
            continue
        selected.append(art)
        if limit and len(selected) >= limit:
            break

    with ThreadPoolExecutor(max_workers=min(workers, 12)) as pool:
        futures = {pool.submit(fetch_text, a): a for a in selected}
        for fut in as_completed(futures):
            art = futures[fut]
            try:
                text = fut.result()
            except Exception:
                continue  # one bad object must not abort a multi-hour ingestion
            if text and len(text) > 500:
                yield to_corpus_doc(art, text)
