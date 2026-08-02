"""Clinical vocabulary expansion — matching the concept, not the string.

**The problem this addresses.** Oncology writing is densely synonymous in a way general
prose is not. The same drug appears as ``osimertinib``, ``AZD9291``, ``AZD-9291`` and
``Tagrisso``; the same receptor as ``EGFR``, ``ErbB1``, ``HER1`` and ``Epidermal Growth
Factor Receptor``. BM25 scores these as unrelated tokens, and a dense model only relates
them if the training corpus happened to co-locate them. A user searching one form silently
misses papers that use another — which is a recall failure invisible from the results page.

**Why MeSH rather than UMLS.** UMLS is the richer resource, but it requires a UTS licence
and an API key, which makes it a barrier for anyone cloning this repo and a credential to
manage in production. NLM's **MeSH database is free, unauthenticated, and supplies entry
terms** covering exactly the drug-brand/code and gene-alias synonymy above. The corpus is
already MeSH-indexed, so the vocabulary is the one the labels come from. If a UTS key is
ever available, ``expand_query`` is the single place that would change.

**Expansion is dangerous and is therefore conservative.** Measured while building this:
``"immune checkpoint inhibitor"`` maps to the *specific compound* ``BMS-1``, so naive
expansion injects a narrow drug name into a broad query and drags in the wrong papers. The
guards below exist because of observed failures, not caution in the abstract:

* only phrases that look like **entities** are expanded — a whole natural-language query is
  never sent to MeSH, because it matches something spuriously specific;
* IUPAC names are dropped (``N-(5-((4-(4-((dimethylamino)methyl)...``): they are real
  synonyms that never occur in running text, so they add only noise;
* the number of synonyms per term is capped, so one prolific concept cannot dominate;
* expansions are emitted **separately** from the original query, so the caller can weight
  them below the user's own words rather than treating them as equally intended.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

#: Cap per term. Beyond a handful the tail is chemical registry noise.
MAX_SYNONYMS_PER_TERM = 6
#: Longer than this and it is a systematic chemical name, not a term anyone writes.
MAX_SYNONYM_CHARS = 40
#: Expand at most this many distinct concepts from one query.
MAX_TERMS_PER_QUERY = 3

_MIN_INTERVAL = 0.35
_last = [0.0]

#: Systematic chemistry: heavy bracketing, locants, or element-prefixed fragments.
_IUPAC = re.compile(r"[\(\)\[\]]{2,}|\b\d+[HR]-|,\s*\d+-|\b[A-Z]{1,2}-\(")
#: An "entity-shaped" phrase: a gene/drug code, or an INN-suffixed drug name.
#:
#: Hyphenated codes are matched **whole**. Splitting on the hyphen turned ``PD-L1`` into
#: ``PD`` + ``L1`` and sent both to MeSH separately — ``PD`` resolved to Parkinson Disease.
#:
#: Two-letter symbols are excluded entirely. ``ER`` is the case that forced this: MeSH
#: resolves it to *Triple Negative Breast Neoplasms*, and that record legitimately contains
#: "ER" (triple-negative is defined as ER-negative), so no record-level guard can catch it.
#: Expanding an **ER-positive** query with **triple-negative** vocabulary inverts the
#: user's meaning, which is the worst outcome available. Two characters simply do not carry
#: enough signal to disambiguate, so they are left alone.
_ENTITY = re.compile(
    r"\b("
    r"[A-Z][A-Za-z0-9]{2,}(?:-[A-Za-z0-9]+)+"    # CTLA-4, AZD-9291, BRCA-1
    r"|[A-Z]{2,5}-[A-Za-z]?\d+"                  # PD-L1, HER-2, IL-6 — a short symbol is
                                                 # unambiguous once it carries a hyphenated
                                                 # locant, unlike bare "PD" or "ER"
    r"|[A-Z][A-Za-z]{1,5}\d+[A-Za-z0-9]*"        # CDK4, EGFR2, TP53, AZD9291, BRCA1
    r"|[A-Z]{3,6}"                               # EGFR, ALK, PARP, MET, BRAF
    r"|[a-z]+(?:tinib|mab|ciclib|parib|zomib|lisib|degib|rafenib|ximab|umab)"
    r")\b"
)


@dataclass
class Expansion:
    """What a query became, kept separable from what the user typed."""

    query: str
    terms: dict[str, list[str]] = field(default_factory=dict)

    @property
    def synonyms(self) -> list[str]:
        out: list[str] = []
        for syns in self.terms.values():
            for s in syns:
                if s not in out:
                    out.append(s)
        return out

    def expanded_query(self, weight_marker: str = " ") -> str:
        """Original query followed by its synonyms.

        Returned as one string for a lexical index that has no field weighting. Callers
        that *can* weight (the SQL path) should use ``synonyms`` directly and score them
        below the user's own terms — a synonym is evidence, not intent.
        """
        syns = self.synonyms
        return self.query if not syns else f"{self.query}{weight_marker}{' '.join(syns)}"


def _throttle() -> None:
    gap = time.monotonic() - _last[0]
    if gap < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - gap)
    _last[0] = time.monotonic()


def candidate_terms(query: str) -> list[str]:
    """Entity-shaped phrases worth looking up.

    Deliberately narrow. Sending the whole query to MeSH is what produced the
    ``immune checkpoint inhibitor`` -> ``BMS-1`` failure: MeSH will always return its best
    match, and for a descriptive phrase that match is often a single narrow compound.
    """
    seen: set[str] = set()
    out: list[str] = []
    for m in _ENTITY.finditer(query):
        t = m.group(1)
        # Common English words that happen to be short and capitalised.
        if t.upper() in {"AND", "OR", "NOT", "THE", "FOR", "WITH", "VS"}:
            continue
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out[:MAX_TERMS_PER_QUERY]


def _record_is_about(term: str, record_terms: list[str]) -> bool:
    """Is this MeSH record actually about ``term``, or merely its best guess?

    MeSH always returns *something*, and for short ambiguous symbols that something can be
    the wrong sense — or the opposite one. Both failures were observed:

    ==========  ================================  ====================================
    query term  MeSH's best match                 why it is wrong
    ==========  ================================  ====================================
    ``MET``     Metabolic Equivalent              the proto-oncogene was meant
    ``ER``      Triple Negative Breast Neoplasms  **the opposite** of ER-positive
    ==========  ================================  ====================================

    Injecting "triple-negative" into an ER-positive query is worse than not expanding at
    all, so a match is accepted only when the queried term appears as a **whole token**
    among the record's own terms. A record genuinely about osimertinib lists
    ``osimertinib``; a record about triple-negative breast cancer never lists ``ER``.
    """
    t = term.lower()
    for rt in record_terms:
        if t in {tok.strip("-,()").lower() for tok in rt.replace("-", " ").split()}:
            return True
        if rt.lower() == t:
            return True
    return False


def _useful_synonym(s: str, original: str) -> bool:
    s = s.strip()
    if not s or len(s) > MAX_SYNONYM_CHARS:
        return False
    if s.lower() == original.lower():
        return False
    if _IUPAC.search(s):
        return False
    # A synonym that is mostly digits is a registry number, not a term in prose.
    letters = sum(c.isalpha() for c in s)
    return letters >= max(2, len(s) * 0.4)


class MeshExpander:
    """Looks up MeSH entry terms, with an on-disk cache.

    The cache is not an optimisation but a requirement: expansion sits on the request path,
    and two uncached E-utilities round trips per term would add seconds to every search and
    put the product's latency at the mercy of NCBI's rate limit.
    """

    def __init__(self, cache_dir: Path | None = None, email: str = "oncolens@example.com",
                 api_key: str | None = None) -> None:
        self.email = email
        self.api_key = api_key
        self.cache_path = (cache_dir / "mesh_synonyms.json") if cache_dir else None
        self._cache: dict[str, list[str]] = {}
        if self.cache_path and self.cache_path.exists():
            try:
                self._cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._cache = {}

    def _save(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, indent=0), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def synonyms(self, term: str) -> list[str]:
        key = term.lower()
        if key in self._cache:
            return self._cache[key]
        syns: list[str] = []
        try:
            import requests

            p = {"tool": "oncolens", "email": self.email, "retmode": "json"}
            if self.api_key:
                p["api_key"] = self.api_key
            _throttle()
            r = requests.get(f"{EUTILS}/esearch.fcgi",
                             params={**p, "db": "mesh", "term": term}, timeout=20)
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if ids:
                _throttle()
                r2 = requests.get(f"{EUTILS}/esummary.fcgi",
                                  params={**p, "db": "mesh", "id": ids[0]}, timeout=20)
                rec = r2.json().get("result", {}).get(ids[0], {})
                raw: list[str] = []
                for k in ("ds_meshterms", "ds_meshsynonyms"):
                    v = rec.get(k)
                    if isinstance(v, list):
                        raw.extend(v)
                # PRECISION GUARD — see _record_is_about().
                if _record_is_about(term, raw):
                    for s in raw:
                        if _useful_synonym(s, term) and s not in syns:
                            syns.append(s)
                        if len(syns) >= MAX_SYNONYMS_PER_TERM:
                            break
        except Exception:  # noqa: BLE001 — expansion must never break a search
            syns = []
        self._cache[key] = syns
        self._save()
        return syns

    def expand(self, query: str) -> Expansion:
        exp = Expansion(query=query)
        for term in candidate_terms(query):
            syns = self.synonyms(term)
            if syns:
                exp.terms[term] = syns
        return exp


_DEFAULT: MeshExpander | None = None


def expand_query(query: str, *, cache_dir: Path | None = None,
                 email: str = "oncolens@example.com") -> Expansion:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = MeshExpander(cache_dir=cache_dir, email=email)
    return _DEFAULT.expand(query)
