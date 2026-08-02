"""Europe PMC ingestion: articles, full text, and — importantly — real grant records.

Europe PMC covers three things this project needs that PubMed alone does not:

1. **Open-access full text** (``/{source}/{id}/fullTextXML``). PubMed gives abstracts;
   passage-level retrieval wants body text with real section structure.
2. **Grist, the grants database** (``/grist``). NIH RePORTER is POST-only and therefore
   unreachable from restricted environments; Grist is GET and returns real awarded grants
   with titles, abstracts, PIs and institutions.
3. **Article<->grant links**, which reproduce the NIH RePORTER grant->publication signal:
   a funder-asserted claim that a paper came out of an award. That is "found data" — a
   relevance judgment made by people, for their own reasons, years before this benchmark.

All endpoints are GET and unauthenticated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree as ET

REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
GRIST = "https://www.ebi.ac.uk/europepmc/GristAPI/rest"

#: JATS sections worth indexing, mapped to the corpus schema's section names.
JATS_SECTIONS = {
    "intro": "Introduction", "introduction": "Introduction",
    "methods": "Methods", "materials|methods": "Methods",
    "results": "Results", "discussion": "Discussion", "conclusion": "Discussion",
}


@dataclass
class EuropePMCRecord:
    ext_id: str
    source: str
    title: str
    abstract: str
    journal: str
    year: int | None
    pmid: str | None = None
    pmcid: str | None = None
    is_open_access: bool = False
    mesh: list[dict[str, Any]] = field(default_factory=list)
    grants: list[dict[str, str]] = field(default_factory=list)
    full_text_sections: list[dict[str, str]] = field(default_factory=list)

    def to_corpus_doc(self) -> dict:
        sections = self.full_text_sections or (
            [{"name": "Abstract", "text": self.abstract}] if self.abstract else []
        )
        if self.full_text_sections and self.abstract:
            sections = [{"name": "Abstract", "text": self.abstract}] + sections
        return {
            "doc_id": f"PAPER:PMID{self.pmid}" if self.pmid else f"PAPER:{self.source}{self.ext_id}",
            "doc_type": "paper",
            "title": self.title,
            "year": self.year,
            "sections": sections,
            "meta": {
                "journal": self.journal, "pmid": self.pmid, "pmcid": self.pmcid,
                "is_open_access": self.is_open_access, "grants": self.grants,
            },
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

    return session("europepmc")


def search(
    query: str, *, page_size: int = 100, max_results: int = 500, result_type: str = "core",
    open_access_only: bool = False,
) -> list[EuropePMCRecord]:
    """Cursor-paginated search. ``resultType=core`` is required for MeSH and grants.

    Europe PMC expands queries with MeSH synonyms by default, which is useful for corpus
    *building* but must never be used to construct evaluation queries — it would make the
    labels agree with the retriever by construction.
    """
    s = _session()
    if open_access_only:
        query = f"({query}) AND OPEN_ACCESS:Y"
    out: list[EuropePMCRecord] = []
    cursor = "*"
    while len(out) < max_results:
        r = s.get(f"{REST}/search", timeout=60, params={
            "query": query, "format": "json", "pageSize": min(page_size, max_results - len(out)),
            "resultType": result_type, "cursorMark": cursor,
        })
        r.raise_for_status()
        data = r.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            break
        out.extend(_parse_result(x) for x in results)
        nxt = data.get("nextCursorMark")
        if not nxt or nxt == cursor:
            break
        cursor = nxt
        time.sleep(0.2)
    return out


def _parse_result(x: dict) -> EuropePMCRecord:
    mesh = []
    for mh in (x.get("meshHeadingList") or {}).get("meshHeading", []):
        mesh.append({
            "descriptor": mh.get("descriptorName"),
            "major": mh.get("majorTopic_YN") == "Y",
            "qualifiers": [
                {"name": q.get("qualifierName"), "major": q.get("majorTopic_YN") == "Y"}
                for q in (mh.get("meshQualifierList") or {}).get("meshQualifier", [])
            ],
        })
    grants = [
        {"grant_id": g.get("grantId", ""), "agency": g.get("agency", ""),
         "acronym": g.get("acronym", "")}
        for g in (x.get("grantsList") or {}).get("grant", [])
    ]
    year = x.get("pubYear")
    return EuropePMCRecord(
        ext_id=x.get("id", ""), source=x.get("source", "MED"),
        title=x.get("title", ""), abstract=x.get("abstractText", "") or "",
        journal=(x.get("journalInfo") or {}).get("journal", {}).get("title", ""),
        year=int(year) if year and str(year).isdigit() else None,
        pmid=x.get("pmid"), pmcid=x.get("pmcid"),
        is_open_access=x.get("isOpenAccess") == "Y",
        mesh=[m for m in mesh if m["descriptor"]], grants=grants,
    )


def full_text_sections(source: str, ext_id: str) -> list[dict[str, str]]:
    """Fetch JATS full text for an open-access record and flatten to schema sections.

    Returns ``[]`` on 404, which is the normal outcome for non-OA records — that is a
    licensing boundary, not an error.
    """
    s = _session()
    r = s.get(f"{REST}/{source}/{ext_id}/fullTextXML", timeout=60)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    return parse_jats(r.text)


def parse_jats(xml_text: str) -> list[dict[str, str]]:
    """Flatten JATS body into named sections, preserving paragraph breaks.

    Paragraph breaks are load-bearing: the chunker aggregates whole paragraphs and never
    splits one, so losing ``<p>`` boundaries would collapse a whole section into one chunk
    and destroy passage-level retrieval.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[dict[str, str]] = []
    for sec in root.findall(".//body//sec"):
        title_el = sec.find("title")
        raw = ("".join(title_el.itertext()).strip() if title_el is not None else "Body")
        name = JATS_SECTIONS.get(raw.lower(), raw[:60] or "Body")
        paras = [
            " ".join("".join(p.itertext()).split())
            for p in sec.findall("p")
        ]
        paras = [p for p in paras if len(p) > 40]
        if paras:
            out.append({"name": name, "text": "\n\n".join(paras)})
    return out


def grist_grants(query: str, *, page_size: int = 100, max_results: int = 300) -> list[dict]:
    """Real awarded grants from Europe PMC's Grist database.

    The GET alternative to NIH RePORTER, which is POST-only and unreachable from
    restricted environments. Returns grant title, abstract, PI, institution and funder.
    """
    s = _session()
    out: list[dict] = []
    page = 1
    while len(out) < max_results:
        r = s.get(f"{GRIST}/get/query={query}", timeout=60,
                  params={"format": "json", "resultType": "core", "page": page})
        r.raise_for_status()
        records = (r.json().get("RecordList") or {}).get("Record", [])
        if not records:
            break
        out.extend(records)
        page += 1
        time.sleep(0.2)
    return [_grant_to_doc(g) for g in out[:max_results]]


def _grant_to_doc(g: dict) -> dict:
    """Map a Grist record into the corpus schema as a grant document."""
    gid = str(g.get("GrantId") or g.get("grantId") or "")
    title = g.get("Title") or ""
    abstract = g.get("Abstract") or ""
    person = g.get("Person") or {}
    inst = g.get("Institution") or {}
    sections = []
    if abstract:
        sections.append({"name": "Abstract", "text": abstract})
    return {
        "doc_id": f"GRANT:{gid}",
        "doc_type": "grant",
        "title": title,
        "year": g.get("StartDate", "")[:4] or None,
        "sections": sections,
        "meta": {
            "pi": f"{person.get('GivenName','')} {person.get('FamilyName','')}".strip(),
            "org": inst.get("Name", ""),
            "funder": g.get("Funder", ""),
            "grant_id": gid,
        },
        "descriptors": [],
        "funded_by": [],
        "cites": [],
    }


def funding_link_qrels(records: list[EuropePMCRecord]) -> list[dict]:
    """Grant -> publication judgments, the Europe PMC analogue of NIH RePORTER links.

    A funder-or-author-asserted claim that a paper resulted from an award. Independent of
    any retrieval system, and it exercises the ``multi_hop`` stratum: answering requires
    traversing a link rather than matching text.
    """
    by_grant: dict[str, dict[str, int]] = {}
    for rec in records:
        if not rec.pmid:
            continue
        for g in rec.grants:
            gid = g.get("grant_id")
            if gid:
                by_grant.setdefault(gid, {})[f"PAPER:PMID{rec.pmid}"] = 3

    out: list[dict] = []
    for gid, judgments in sorted(by_grant.items()):
        if len(judgments) < 3:      # never emit single-relevant queries
            continue
        out.append({
            "query_id": f"FUND{abs(hash(gid)) % 100000:05d}",
            "query": f"publications resulting from grant {gid}",
            "stratum": "multi_hop",
            "source": "funding_link",
            "judgments": judgments,
            "notes": f"Funder-asserted grant->publication links for {gid} ({len(judgments)} papers).",
        })
    return out
