"""PubMed / NCBI E-utilities ingestion.

All E-utilities endpoints are **GET**, which matters: they are reachable from restricted
environments where a POST-only API (NIH RePORTER) is not.

The reason this module exists is not the abstracts — it is ``MeshHeading``. Every PubMed
record is indexed by NLM's *human* indexers with MeSH descriptors, qualifiers, and a
``MajorTopicYN`` flag. That is a large, pre-existing, human-assigned concept->document
labelling built by people who never saw this retrieval system. It is the single
highest-validity label source available for this task, and it maps directly onto the
product requirement ("search a concept, get the associated papers").

``MajorTopicYN`` is the part that makes graded relevance possible without inventing
anything: NLM already distinguishes a paper's *central* topics from its incidental ones.

Rate limits: 3 requests/second without an API key, 10/second with one
(``NCBI_API_KEY``). ``email`` and ``tool`` are sent because NCBI asks for them and will
throttle anonymous heavy use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

#: Grade mapping from NLM's own indexing. Not invented — read off MajorTopicYN.
GRADE_MAJOR_TOPIC = 3      # descriptor flagged as a major topic of the paper
GRADE_MINOR_TOPIC = 1      # descriptor present but not major


@dataclass
class PubMedRecord:
    pmid: str
    title: str
    abstract: str
    journal: str
    year: int | None
    mesh: list[dict[str, Any]] = field(default_factory=list)   # {descriptor, major, qualifiers}
    keywords: list[str] = field(default_factory=list)
    grants: list[dict[str, str]] = field(default_factory=list)  # {grant_id, agency}
    pmcid: str | None = None

    def to_corpus_doc(self) -> dict:
        """Convert to the OncoLens corpus schema (see docs/CORPUS_SCHEMA.md)."""
        sections = []
        if self.abstract:
            sections.append({"name": "Abstract", "text": self.abstract})
        return {
            "doc_id": f"PAPER:PMID{self.pmid}",
            "doc_type": "paper",
            "title": self.title,
            "year": self.year,
            "sections": sections,
            "meta": {
                "journal": self.journal, "pmid": self.pmid, "pmcid": self.pmcid,
                "grants": self.grants, "keywords": self.keywords,
            },
            # NOTE: real MeSH descriptors, not synthetic labels.
            "descriptors": [f"MESH:{m['descriptor']}" for m in self.mesh],
            "mesh_detail": self.mesh,
            "funded_by": [f"GRANT:{g['grant_id']}" for g in self.grants if g.get("grant_id")],
            "cites": [],
        }


def _session():
    """Shared retrying session — see oncolens.http for why.

    Two long ingests died on transient transport errors from
    different services; retrying only the module that failed
    first fixed a location, not the class of bug.
    """
    from ..http import session

    return session("pubmed")


def _params(extra: dict, email: str | None, api_key: str | None) -> dict:
    p = {"tool": "oncolens", **extra}
    if email:
        p["email"] = email
    if api_key:
        p["api_key"] = api_key
    return p


def esearch(
    term: str, *, retmax: int = 100, mindate: int | None = None, maxdate: int | None = None,
    email: str | None = None, api_key: str | None = None,
) -> list[str]:
    """Return PMIDs for a query. Supports full PubMed field syntax, e.g.
    ``"Osimertinib"[MeSH Terms] AND "Drug Resistance, Neoplasm"[MeSH Terms]``.
    """
    s = _session()
    p = _params({"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"}, email, api_key)
    if mindate:
        p |= {"mindate": mindate, "maxdate": maxdate or mindate, "datetype": "pdat"}
    r = s.get(f"{EUTILS}/esearch.fcgi", params=p, timeout=30)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def efetch(
    pmids: list[str], *, batch: int = 100, email: str | None = None,
    api_key: str | None = None, sleep: float = 0.34,
) -> list[PubMedRecord]:
    """Fetch full records as XML and parse title, abstract, MeSH, keywords and grants.

    XML (not JSON) because ``retmode=json`` on efetch does not return MeSH headings —
    the labels are only in the XML representation.
    """
    s = _session()
    out: list[PubMedRecord] = []
    for i in range(0, len(pmids), batch):
        chunk = pmids[i : i + batch]
        p = _params({"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"}, email, api_key)
        r = s.get(f"{EUTILS}/efetch.fcgi", params=p, timeout=60)
        r.raise_for_status()
        out.extend(parse_pubmed_xml(r.text))
        time.sleep(sleep)  # NCBI asks for <= 3 req/s without a key
    return out


def parse_pubmed_xml(xml_text: str) -> list[PubMedRecord]:
    """Parse an efetch PubmedArticleSet. Kept pure so it is testable offline."""
    root = ET.fromstring(xml_text)
    records: list[PubMedRecord] = []
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        if pmid_el is None or not pmid_el.text:
            continue

        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""

        # Structured abstracts have multiple labelled AbstractText nodes; join with labels
        # so section structure survives into chunking.
        parts: list[str] = []
        for node in art.findall(".//Abstract/AbstractText"):
            text = "".join(node.itertext()).strip()
            if not text:
                continue
            label = node.get("Label")
            parts.append(f"{label}: {text}" if label else text)
        abstract = "\n\n".join(parts)

        journal_el = art.find(".//Journal/Title")
        year_el = art.find(".//JournalIssue/PubDate/Year")
        year = int(year_el.text) if year_el is not None and (year_el.text or "").isdigit() else None

        mesh: list[dict[str, Any]] = []
        for mh in art.findall(".//MeshHeadingList/MeshHeading"):
            d = mh.find("DescriptorName")
            if d is None or not d.text:
                continue
            mesh.append({
                "descriptor": d.text,
                "ui": d.get("UI"),
                "major": d.get("MajorTopicYN") == "Y",
                "qualifiers": [
                    {"name": q.text, "major": q.get("MajorTopicYN") == "Y"}
                    for q in mh.findall("QualifierName") if q.text
                ],
            })

        keywords = [
            "".join(k.itertext()).strip()
            for k in art.findall(".//KeywordList/Keyword")
            if "".join(k.itertext()).strip()
        ]

        grants = []
        for g in art.findall(".//GrantList/Grant"):
            gid = g.findtext("GrantID")
            agency = g.findtext("Agency")
            if gid or agency:
                grants.append({"grant_id": (gid or "").strip(), "agency": (agency or "").strip()})

        pmcid = None
        for aid in art.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "pmc":
                pmcid = aid.text

        records.append(PubMedRecord(
            pmid=pmid_el.text, title=title, abstract=abstract,
            journal=journal_el.text if journal_el is not None else "",
            year=year, mesh=mesh, keywords=keywords, grants=grants, pmcid=pmcid,
        ))
    return records


def mesh_qrels(records: list[PubMedRecord], *, min_docs: int = 3, max_docs: int = 40) -> list[dict]:
    """Build graded relevance judgments from NLM's human MeSH indexing.

    This is the payoff of the whole module: a concept->document labelling that no LLM
    invented and no retriever influenced.

    * query    = the MeSH descriptor's preferred term
    * relevant = every record indexed with that descriptor
    * grade    = 3 when NLM flagged it a **major topic**, 1 otherwise

    ``min_docs`` avoids single-relevant queries (which systematically penalise better
    systems). ``max_docs`` drops descriptors so generic they carry no discriminative
    signal — "Humans" is on nearly every record and would make a query every system
    answers identically.
    """
    by_desc: dict[str, dict[str, int]] = {}
    for rec in records:
        doc_id = f"PAPER:PMID{rec.pmid}"
        for m in rec.mesh:
            grade = GRADE_MAJOR_TOPIC if m["major"] else GRADE_MINOR_TOPIC
            by_desc.setdefault(m["descriptor"], {})[doc_id] = grade

    out: list[dict] = []
    for desc, judgments in sorted(by_desc.items()):
        if not (min_docs <= len(judgments) <= max_docs):
            continue
        majors = sum(1 for g in judgments.values() if g == GRADE_MAJOR_TOPIC)
        out.append({
            "query_id": f"MESH{abs(hash(desc)) % 100000:05d}",
            "query": desc,
            "stratum": "conceptual",
            "source": "mesh_indexing",
            "judgments": judgments,
            "notes": (
                f"NLM human MeSH indexing: {len(judgments)} records carry '{desc}', "
                f"{majors} as a major topic (grade 3), rest minor (grade 1)."
            ),
        })
    return out
