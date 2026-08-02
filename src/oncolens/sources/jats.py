"""JATS XML from PMC — structural ground truth that the plain-text rendition throws away.

PMC serves the same article two ways. The ``.txt`` rendition is what we index (it is what
the Cloud Service bulk-distributes), but it is *lossy*: section structure, the reference
list boundary, and every in-text citation link are flattened into undifferentiated prose.

The JATS XML keeps all of it, explicitly:

* ``<ref-list>`` marks exactly where the bibliography starts. That is **ground truth** for
  the reference stripper — not a heuristic to be tuned against a human's eyeballing, but a
  label produced by the publisher.
* ``<xref ref-type="bibr" rid="...">`` marks every in-text citation, tying a *sentence* to
  the *work it cites*. That is the raw material for citation-context labelling.
* ``<pub-id pub-id-type="pmid">`` inside each ``<ref>`` resolves a citation to a real
  article we may also hold.

We do not index XML — parsing it per query would be wasteful and the txt rendition is what
scales through the Cloud Service. XML is fetched for *supervision*: to measure the stripper
and to mine labels. That separation matters, because it means the expensive fetch happens
offline and never on the request path.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

#: NCBI allows 3 requests/second without an API key, 10 with one. Staying under the
#: unauthenticated limit keeps this runnable by anyone who clones the repo.
_MIN_INTERVAL = 0.34
_last_call = [0.0]


def _throttle() -> None:
    import time as _t
    gap = _t.monotonic() - _last_call[0]
    if gap < _MIN_INTERVAL:
        _t.sleep(_MIN_INTERVAL - gap)
    _last_call[0] = _t.monotonic()


def fetch_xml(pmcid: str, *, email: str, api_key: str | None = None, timeout: int = 90) -> str | None:
    """Fetch JATS XML for one PMC article. Returns None if PMC has no XML for it."""
    import requests

    numeric = pmcid.upper().replace("PMC", "")
    params = {"db": "pmc", "id": numeric, "rettype": "xml",
              "tool": "oncolens", "email": email}
    if api_key:
        params["api_key"] = api_key
    _throttle()
    r = requests.get(EFETCH, params=params, timeout=timeout)
    if r.status_code != 200:
        return None
    text = r.text
    # PMC returns a stub with this message for articles whose XML is not redistributable.
    if "<body" not in text and "<ref-list" not in text:
        return None
    return text


# --- reference list -------------------------------------------------------------

_REF_LIST = re.compile(r"<ref-list\b.*?</ref-list>", re.S)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def strip_tags(xml_fragment: str) -> str:
    return _WS.sub(" ", _TAG.sub(" ", xml_fragment)).strip()


def reference_list_text(xml: str) -> str | None:
    """Plain text of the article's ``<ref-list>``, or None if it has none.

    Some articles carry more than one ref-list (per-section bibliographies). The *first*
    one is what matters: it marks where the body stops.
    """
    m = _REF_LIST.search(xml)
    if not m:
        return None
    return strip_tags(m.group(0)) or None


# --- citation contexts ----------------------------------------------------------

_REF = re.compile(r"<ref\b[^>]*\bid=\"([^\"]+)\"[^>]*>(.*?)</ref>", re.S)
_PMID = re.compile(r"<pub-id[^>]*pub-id-type=\"pmid\"[^>]*>(\d+)</pub-id>", re.I)
_DOI = re.compile(r"<pub-id[^>]*pub-id-type=\"doi\"[^>]*>([^<]+)</pub-id>", re.I)
_ARTICLE_TITLE = re.compile(r"<article-title>(.*?)</article-title>", re.S)


@dataclass(frozen=True)
class Reference:
    rid: str
    pmid: str | None
    doi: str | None
    title: str | None


def parse_references(xml: str) -> dict[str, Reference]:
    """Map ``rid`` -> resolved identifiers for every entry in the bibliography."""
    out: dict[str, Reference] = {}
    block = _REF_LIST.search(xml)
    scope = block.group(0) if block else xml
    for rid, body in _REF.findall(scope):
        pmid = _PMID.search(body)
        doi = _DOI.search(body)
        title = _ARTICLE_TITLE.search(body)
        out[rid] = Reference(
            rid=rid,
            pmid=pmid.group(1) if pmid else None,
            doi=doi.group(1).strip() if doi else None,
            title=strip_tags(title.group(1)) if title else None,
        )
    return out


_BODY = re.compile(r"<body\b.*?</body>", re.S)
_XREF = re.compile(r"<xref\b[^>]*ref-type=\"bibr\"[^>]*>", re.I)
_RID_ATTR = re.compile(r"\brid=\"([^\"]+)\"")
#: Sentence boundary that tolerates the abbreviations biomedical prose is full of.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class CitationContext:
    """One sentence, and the works it cites.

    ``sentence`` is the citing author's own description of the cited work — which is
    precisely a natural-language query for which the cited work is a relevant answer.
    """

    sentence: str
    rids: list[str] = field(default_factory=list)
    section: str = ""


def _section_titles(body: str) -> list[tuple[int, str]]:
    out = []
    for m in re.finditer(r"<title>(.*?)</title>", body, re.S):
        out.append((m.start(), strip_tags(m.group(1))))
    return out


def extract_citation_contexts(xml: str, *, min_chars: int = 60,
                              max_chars: int = 400) -> list[CitationContext]:
    """Pull every sentence that carries an in-text citation, with the ids it cites.

    The XML is reduced to text *while remembering where the ``<xref>`` markers landed*,
    by substituting each marker with a unique sentinel before tags are stripped. Doing it
    the other way round — stripping tags then regexing for "[12]" — fails on the many
    journals that render citations as superscripts or author-year.
    """
    mb = _BODY.search(xml)
    body = mb.group(0) if mb else xml

    sentinels: dict[str, str] = {}
    counter = [0]

    def _sub(m: re.Match) -> str:
        rid_m = _RID_ATTR.search(m.group(0))
        if not rid_m:
            return " "
        # rid can list several ids separated by whitespace.
        token = f" \x00{counter[0]}\x00 "
        sentinels[str(counter[0])] = rid_m.group(1)
        counter[0] += 1
        return token

    marked = _XREF.sub(_sub, body)
    # Drop the ref-list itself so bibliography entries are not mined as "contexts".
    marked = _REF_LIST.sub(" ", marked)
    # Tables and figures are not prose; their captions cite but rarely describe.
    marked = re.sub(r"<table-wrap\b.*?</table-wrap>", " ", marked, flags=re.S)

    text = _WS.sub(" ", _TAG.sub(" ", marked))

    out: list[CitationContext] = []
    for sent in _SENT_SPLIT.split(text):
        ids = re.findall(r"\x00(\d+)\x00", sent)
        if not ids:
            continue
        clean = _WS.sub(" ", re.sub(r"\x00\d+\x00", "", sent)).strip()
        # Strip the residual bracket punctuation left where the marker was.
        clean = re.sub(r"\s*\[\s*[,\-–\s]*\]", "", clean)
        clean = re.sub(r"\(\s*[,;\-–\s]*\)", "", clean).strip()
        if not (min_chars <= len(clean) <= max_chars):
            continue
        rids: list[str] = []
        for i in ids:
            for rid in sentinels.get(i, "").split():
                if rid not in rids:
                    rids.append(rid)
        if rids:
            out.append(CitationContext(sentence=clean, rids=rids))
    return out
