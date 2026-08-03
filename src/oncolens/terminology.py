"""Clinical vocabulary expansion — matching the concept, not the string.

**The problem this addresses.** Oncology writing is densely synonymous in a way general
prose is not. The same drug appears as ``osimertinib``, ``AZD9291``, ``AZD-9291`` and
``Tagrisso``; the same receptor as ``EGFR``, ``ErbB1``, ``HER1`` and ``Epidermal Growth
Factor Receptor``. BM25 scores these as unrelated tokens, and a dense model only relates
them if the training corpus happened to co-locate them. A user searching one form silently
misses papers that use another — which is a recall failure invisible from the results page.

**Where oncology terms actually live — measured, not assumed.** An earlier version of this
note said MeSH was chosen over UMLS because UMLS needs a UTS licence, and left it there.
That reasoning was right about the credential and wrong about the conclusion, because it
never asked what the queries contain. Coverage of the **335 identifier queries this project
serves**, every registry free and unauthenticated:

=====================  ========  ==========================================================
registry               coverage  what it holds
=====================  ========  ==========================================================
**NCI Thesaurus**      **55.8%** NCI's own oncology ontology: 212,475 concepts — genes,
                                 proteins, drugs, variants, antigens, HLA alleles
ClinicalTrials.gov     22.7%     trial acronyms (``PALOMA-3``), resolved to interventions
HGNC (*what shipped*)  13.7%     human gene symbols only
Cellosaurus            8.4%      cell lines (``MCF-7``, ``UACC-812``, ``OVCAR-8``)
dbSNP                  0.9%      rsIDs, resolved to their gene
**union**              **87.5%**
=====================  ========  ==========================================================

**No single source is sufficient.** The shipped expander reached an eighth of its own
stratum because HGNC was picked for its authority over gene symbols and the stratum is only
partly genes — it also contains cell lines, investigational drug codes, trial acronyms and
an EORTC questionnaire. The remaining 12.5% is mostly source-literature spelling variants
(``CLTA-4`` for ``CTLA-4``) and artifacts of citation-context extraction (``HER2-0``,
``CD19-4``), not missing vocabulary.

**UMLS is the true superset, and it is still worth having.** It unifies ~200 vocabularies
under one CUI, so ``COX-2``'s senses arrive already merged across MeSH, NCIt, SNOMED and
HGNC. Measured 2026-08-03: ``uts-ws.nlm.nih.gov`` returns **401 without an API key**;
registration is free at uts.nlm.nih.gov. It is not a blocker, because **NCIt is a source
vocabulary inside UMLS** — the cascade below is the oncology slice of UMLS without the
credential. A UTS key would mainly add non-oncology vocabularies and cross-source CUI
merging, so it is an upgrade rather than a prerequisite. ``OntologyExpander.resolve`` is
the single place that would change.

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
    #: What each term was resolved *to*, when a typed registry answered. Populated by
    #: ``OntologyExpander``; empty for the MeSH-only path, which returns strings with no
    #: concept behind them. Carrying the type is what makes an expansion inspectable —
    #: "PALOMA-3 -> palbociclib, fulvestrant (clinical trial, NCT01942135)" can be shown
    #: to a user and checked; a bare list of extra words cannot.
    resolutions: dict[str, "Resolution"] = field(default_factory=dict)

    @property
    def synonyms(self) -> list[str]:
        out: list[str] = []
        for syns in self.terms.values():
            for s in syns:
                if s not in out:
                    out.append(s)
        return out

    @property
    def identity_synonyms(self) -> list[str]:
        """Only synonyms that denote the *same* thing as the term.

        Excludes association links (a trial's interventions, an rsID's gene). See
        ``Resolution.relation`` for the measurement that forced the distinction.
        """
        out: list[str] = []
        for term, syns in self.terms.items():
            r = self.resolutions.get(term)
            if r is not None and r.relation != "identity":
                continue
            for s in syns:
                if s not in out:
                    out.append(s)
        return out

    def expanded_query(self, weight_marker: str = " ", *, repeat: int = 1,
                       identity_only: bool = False) -> str:
        """Original query, then its synonyms.

        ``repeat`` is how the caller weights intent above evidence on a lexical index with
        no field weighting: repeating the user's own words raises their term frequency
        relative to the injected ones. **This is not decoration.** The first identifier run
        expanded 212 of 236 queries at ``repeat=1`` and regressed ``success@1`` by 0.0339
        (p=0.037), because a one-token query like ``MCF-7`` became 1 of 7 equally-weighted
        tokens — the user's actual term diluted to 14% of the query it was meant to be.

        This method's docstring already said callers "should score them below the user's
        own terms". Nothing did. That is the §4.15 shape: a guard that exists in the design,
        is described in the notes, and is never invoked.
        """
        syns = self.identity_synonyms if identity_only else self.synonyms
        base = " ".join([self.query] * max(1, repeat))
        return base if not syns else f"{base}{weight_marker}{' '.join(syns)}"


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
    # SHORT ALL-CAPS SYNONYMS ARE REJECTED, for the same reason `_ENTITY` refuses to expand
    # two-letter query terms. HGNC's alias_symbol field mixes true gene aliases with
    # disease-locus names, so CTLA4 comes back with CD152 (useful) alongside CD, GSE,
    # CELIAC3 and IDDM12 (celiac and diabetes loci). Injecting "CD" into a CTLA-4 query
    # matches every CD-something paper in an immunology corpus, which is a far larger
    # error than the recall the synonym could ever recover. Four characters is the point
    # where a symbol carries enough signal to be worth the risk: CD152 and ERBB1 survive,
    # CD and GSE do not.
    if len(s) < 4 and s.isupper():
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


HGNC = "https://rest.genenames.org"

#: A gene-symbol-shaped token. HGNC symbols are upper-case, 2-10 characters, and may carry
#: digits and a hyphen. Anything else is not worth a round trip to a gene registry.
_GENE_SHAPED = re.compile(r"^[A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]{1,4})?$")


class HgncExpander:
    """Gene aliases from the HUGO Gene Nomenclature Committee.

    **Why add this to MeSH rather than rely on MeSH alone.** MeSH is a *subject* thesaurus.
    It covers drugs and diseases well and gene aliases erratically, and its lookup returns a
    best match for anything, which is why ``MeshExpander`` needs ``_record_is_about`` to
    reject ``MET`` -> Metabolic Equivalent. HGNC is the opposite kind of resource: the
    official registry of human gene symbols, where ``fetch/symbol/EGFR`` either matches
    exactly or returns nothing. That exactness is the point — the precision guard is the
    endpoint itself, not a heuristic applied afterwards.

    It is also the resource that covers the specific failure the identifier stratum is made
    of: a bare symbol whose paper uses a different one. ``HER1``, ``ERBB1`` and ``EGFR`` are
    the same gene; ``alias_symbol`` and ``prev_symbol`` say so, with an authority behind it.

    Three endpoints are tried because a user may type the current symbol, a retired one, or
    a colloquial alias, and only the first is a direct hit.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_path = (cache_dir / "hgnc_aliases.json") if cache_dir else None
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

    def _fetch(self, field: str, term: str) -> dict | None:
        import requests

        _throttle()
        r = requests.get(f"{HGNC}/fetch/{field}/{term}",
                         headers={"Accept": "application/json"}, timeout=20)
        if not r.ok:
            return None
        docs = (r.json().get("response") or {}).get("docs") or []
        return docs[0] if docs else None

    def synonyms(self, term: str) -> list[str]:
        key = term.lower()
        if key in self._cache:
            return self._cache[key]
        syns: list[str] = []
        # HGNC symbols carry no hyphen: the registry spells it LAG3, papers and users
        # write LAG-3. Without trying the de-hyphenated form, `LAG-3` — an actual query in
        # the identifier stratum — expanded to nothing. Both forms are tried, original
        # first, because a hyphen is occasionally significant (HLA-A is a real symbol).
        variants = [term]
        if "-" in term:
            variants.append(term.replace("-", ""))
        if _GENE_SHAPED.match(term) or any(_GENE_SHAPED.match(v) for v in variants):
            try:
                doc = None
                for v in variants:
                    for field in ("symbol", "alias_symbol", "prev_symbol"):
                        doc = self._fetch(field, v)
                        if doc:
                            break
                    if doc:
                        break
                if doc:
                    raw = [doc.get("symbol") or ""]
                    for k in ("alias_symbol", "prev_symbol"):
                        v = doc.get(k)
                        if isinstance(v, list):
                            raw.extend(v)
                        elif isinstance(v, str):
                            raw.append(v)
                    # The approved long name is useful when it is a real phrase people
                    # write ("epidermal growth factor receptor") and noise when it is a
                    # registry description, so it goes through the same filter.
                    name = doc.get("name")
                    if isinstance(name, str):
                        raw.append(name)
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


NCIT = "https://api-evsrest.nci.nih.gov/api/v1"
CELLOSAURUS = "https://api.cellosaurus.org"
CTGOV = "https://clinicaltrials.gov/api/v2"

#: Trial arms that name no specific agent. ``Placebo`` was being injected into a
#: ``PALOMA-3`` query, where it matches every randomised trial in the corpus — a synonym
#: with no discriminating power is pure noise added to the lexical arm.
_GENERIC_ARM = re.compile(
    r"^(placebo|best supportive care|standard of care|observation|no intervention|"
    r"control|vehicle|saline|questionnaire|survey|virtual reality task)\b", re.I)


def norm_key(s: str) -> str:
    """The form a term is *looked up* under.

    Case, hyphens, dots and spaces carry no identity in this vocabulary — ``PD-L1``,
    ``PD L1`` and ``PDL1`` are one concept, and ``STAT-3`` is how a paper spells ``STAT3``.
    Folding them away is what lets a dictionary match surface variants without a rule per
    variant. Measured on the identifier stratum, this alone reaches ``VEGF-R2``, ``STAT-3``,
    ``CCL-2`` and ``MUC-16``, none of which resolve as written.
    """
    return "".join(ch for ch in s.casefold() if ch.isalnum())


#: Below this many characters (after ``norm_key``) a term is not expanded at all.
#:
#: The old code refused two-character terms because **MeSH** resolved ``ER`` to *Triple
#: Negative Breast Neoplasms* — the opposite of ER-positive. That reason is obsolete: NCIt
#: resolves ``ER`` correctly, to the Estrogen Receptor Family.
#:
#: The guard stays because the real hazard was never MeSH's error. ``ER`` matches **14**
#: NCIt concepts — Endoplasmic Reticulum, Adverse Event Emergency Room Visit, Eritrea,
#: Estrogen Receptor Positive *and* Estrogen Receptor Negative — and ``_SEMANTIC_RANK``
#: quietly resolves all fourteen to one. In an oncology corpus "ER stress" (the organelle)
#: and "ER+" (the receptor) are both common, and two characters carry no signal that could
#: separate them. Choosing silently is the failure; the ranking is not evidence, it is a
#: default. Three characters is where a symbol starts to be self-identifying: ``MET``,
#: ``ALK`` and ``CD8`` survive, ``ER`` and ``PR`` do not.
MIN_TERM_CHARS = 3

#: Preference over NCIt semantic types, used **only** to choose between competing senses of
#: one surface form. A term with a single sense keeps it whatever its type — otherwise
#: ``QLQ-C30``, whose only sense is an ``Intellectual Product``, would resolve to nothing.
#:
#: Measured need: ``PECAM-1`` matches a CDISC *Laboratory Procedure* concept ("PECAM-1
#: Measurement") **before** the protein, because NCIt orders it first. Expanding a protein
#: query with assay vocabulary is the §4.4 failure mode — evidence about how something is
#: measured injected into a question about what it does.
_SEMANTIC_RANK: dict[str, int] = {
    "Gene or Genome": 0,
    "Amino Acid, Peptide, or Protein": 0,
    "Enzyme": 0,
    "Immunologic Factor": 0,
    "Receptor": 0,
    "Nucleotide Sequence": 0,
    "Neoplastic Process": 1,
    "Cell or Molecular Dysfunction": 1,
    "Disease or Syndrome": 1,
    "Pharmacologic Substance": 1,
    "Organic Chemical": 1,
    "Cell": 1,
    # Demoted: these describe how a thing is measured or recorded, not the thing.
    "Laboratory Procedure": 8,
    "Laboratory or Test Result": 8,
    "Quantitative Concept": 8,
    "Finding": 8,
    "Intellectual Product": 8,
}
_DEFAULT_RANK = 4


@dataclass
class Resolution:
    """A term resolved to a concept, carrying **what kind of thing** it is.

    This is the property the regex could not have at any level of refinement. ``CTLA-4``
    and ``BOLERO-2`` are the same string shape; they are a gene and a clinical trial, and
    only the vocabulary that contains them knows that.
    """

    term: str
    kind: str                      #: gene | concept | cell line | trial | variant | ...
    source: str                    #: which registry answered
    code: str = ""
    label: str = ""
    synonyms: list[str] = field(default_factory=list)
    #: ``identity`` — the synonyms denote the SAME thing (``MCF7`` is ``MCF-7``).
    #: ``association`` — they denote something RELATED (``PALOMA-3`` -> ``Palbociclib``).
    #:
    #: Measured need: the first cascade returned both as one undifferentiated list and
    #: regressed identifier ``success@1`` by 0.0339 (p=0.037). A trial is not synonymous
    #: with its interventions — expanding ``PALOMA-3`` to ``Palbociclib`` turns a question
    #: about one study into every palbociclib paper in the corpus. Both relations are
    #: useful, and they must not be injected the same way.
    relation: str = "identity"
    #: Competing senses at the *same* preference rank — peers the ranking could not
    #: separate, so the choice fell to NCIt's relevance order.
    ambiguous: int = 0
    #: **Total** exact-match concepts for this surface form, across all ranks. Reported
    #: separately because ``ambiguous`` alone is misleading: ``ER`` matches 14 NCIt
    #: concepts — endoplasmic reticulum, emergency room, Eritrea, estrogen receptor
    #: *positive* and *negative* — and every one of them loses to the protein sense on
    #: semantic rank, so ``ambiguous`` is 0 while 13 senses are silently discarded. A
    #: number that reads as "unambiguous" in that situation is worse than no number.
    senses: int = 0


class _CachedResolver:
    """Shared on-disk cache plumbing. Expansion sits on the request path (see
    ``MeshExpander``), so an uncached round trip per query is not acceptable."""

    filename = "resolver.json"

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_path = (cache_dir / self.filename) if cache_dir else None
        self._cache: dict = {}
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

    def _get(self, term: str):
        return self._cache.get(norm_key(term))

    def _put(self, term: str, value) -> None:
        self._cache[norm_key(term)] = value
        self._save()


class NcitExpander(_CachedResolver):
    """NCI Thesaurus — the oncology vocabulary, and the one that should have been first.

    **Measured against what shipped.** On the 335 identifier queries this project actually
    serves, HGNC resolved **13.7%** and NCIt resolved **55.8%**. HGNC was chosen because it
    is authoritative for *gene symbols*; the stratum turned out to be only partly genes. It
    also contains cell lines, investigational drug codes (``CALAA-01``, ``HKI-272``), HLA
    alleles, antigens, protein variants (``EGFR T790M``), trial acronyms and even an EORTC
    questionnaire (``QLQ-C30``). Picking a vocabulary by reputation rather than by what the
    queries contain is how a component ends up covering an eighth of its own stratum.

    NCIt is free and unauthenticated: 212,475 concepts, no UTS licence, no API key. It is
    also a *source vocabulary inside UMLS*, so this is the oncology slice of UMLS without
    the credential (see ``expand_query`` for what UMLS would add).

    **Sense selection, without a hand-written rule.** A surface form routinely matches
    several concepts. ``COX-2`` matches three: prostaglandin G/H synthase 2 (right), the
    ``PTGS2`` allele (right), and the ``PTGER2`` allele (**wrong** — a different gene).
    That ambiguity is real *in the authoritative source*, not a lookup bug, and it is why
    the shipped expander returned "prostaglandin E receptor 2" for a COX-2 query.

    Three signals order the senses, none of them a curated answer key:

    1. **semantic type** — demote measurement/questionnaire senses (``_SEMANTIC_RANK``);
    2. **term type** — ``CN`` marks a development *code name*. ``MAGE-A3`` is an antigen and
       also the code name of the vaccine Astuprotimut-R; ``CN`` is a weaker identity claim
       than ``SY`` and loses to it;
    3. **NCIt's own relevance order** as the final tiebreak.

    Only the single best-ranked concept is expanded. When senses genuinely tie — as COX-2's
    do — taking the union would inject the wrong gene's vocabulary, and §4.4's rule is that
    a wrong expansion costs more than a missing one.
    """

    filename = "ncit_concepts.json"

    def resolve(self, term: str) -> Resolution | None:
        cached = self._get(term)
        if cached is not None:
            return Resolution(**cached) if cached else None
        out = None
        try:
            import requests

            _throttle()
            r = requests.get(f"{NCIT}/concept/ncit/search",
                             params={"term": term, "type": "match", "pageSize": 8,
                                     "include": "synonyms,properties"}, timeout=20)
            concepts = r.json().get("concepts") or [] if r.ok else []
            want = norm_key(term)
            scored = []
            for order, c in enumerate(concepts):
                sem = [p["value"] for p in (c.get("properties") or [])
                       if p.get("type") == "Semantic_Type"]
                rank = min((_SEMANTIC_RANK.get(s, _DEFAULT_RANK) for s in sem),
                           default=_DEFAULT_RANK)
                mine = [y for y in (c.get("synonyms") or [])
                        if norm_key(y.get("name") or "") == want]
                if not mine:
                    continue
                # A code name is a weaker identity claim than a synonym.
                tt = 1 if all((y.get("termType") or "") == "CN" for y in mine) else 0
                scored.append(((rank, tt, order), c, sem))
            if scored:
                scored.sort(key=lambda x: x[0])
                best = scored[0][0][:2]
                ties = sum(1 for k, _, _ in scored if k[:2] == best)
                _, c, sem = scored[0]
                syns: list[str] = []
                for y in (c.get("synonyms") or []):
                    s = (y.get("name") or "").strip()
                    if _useful_synonym(s, term) and s not in syns:
                        syns.append(s)
                    if len(syns) >= MAX_SYNONYMS_PER_TERM:
                        break
                out = Resolution(term=term, kind=(sem[0] if sem else "NCIt concept"),
                                 source="NCIt", code=c.get("code") or "",
                                 label=c.get("name") or "", synonyms=syns,
                                 ambiguous=max(0, ties - 1), senses=len(scored))
        except Exception:  # noqa: BLE001 — expansion must never break a search
            out = None
        self._put(term, out.__dict__ if out else {})
        return out


class CellosaurusExpander(_CachedResolver):
    """Cell lines, which a gene registry structurally cannot resolve.

    ``MCF-7``, ``UACC-812``, ``OVCAR-8``, ``NALM-6``, ``B16-F10`` and ``P493-6`` are all
    real identifier queries, and all of them are cell lines. HGNC returns nothing for every
    one, and no amount of pattern tuning changes that — ``UACC-812`` is *shaped* exactly
    like ``CTLA-4``. Cellosaurus is the registry for these, free and unauthenticated, and it
    supplies precisely the synonymy that matters for retrieval: ``MCF-7`` is written
    ``MCF7``, ``MCF 7`` and ``Michigan Cancer Foundation-7`` across the literature.
    """

    filename = "cellosaurus.json"

    def resolve(self, term: str) -> Resolution | None:
        cached = self._get(term)
        if cached is not None:
            return Resolution(**cached) if cached else None
        out = None
        try:
            import requests

            _throttle()
            r = requests.get(f"{CELLOSAURUS}/search/cell-line",
                             params={"q": f'id:"{term}"', "format": "json", "rows": 3},
                             timeout=20)
            cls = (r.json().get("Cellosaurus") or {}).get("cell-line-list") or [] if r.ok else []
            want = norm_key(term)
            for cl in cls:
                names = [n.get("value") for n in (cl.get("name-list") or []) if n.get("value")]
                if not names or norm_key(names[0]) != want:
                    continue
                syns = [s for s in names[1:] if _useful_synonym(s, term)][:MAX_SYNONYMS_PER_TERM]
                out = Resolution(term=term, kind="cell line", source="Cellosaurus",
                                 code=cl.get("accession") or "", label=names[0],
                                 synonyms=syns)
                break
        except Exception:  # noqa: BLE001
            out = None
        self._put(term, out.__dict__ if out else {})
        return out


class TrialExpander(_CachedResolver):
    """Trial acronyms and rsIDs — resolved to what they *are about*.

    ``PALOMA-3`` is the single most frequent unresolved identifier query in the stratum
    (7 occurrences), and it is not in any terminology because a trial is not a concept. It
    is in ClinicalTrials.gov, where it resolves to NCT01942135 and, usefully, to its
    **interventions**: palbociclib and fulvestrant. That is the expansion a reader wants —
    passages discussing the drugs are the passages about the trial.

    dbSNP does the same job for ``rs4149056`` -> ``SLCO1B1``. Both are unauthenticated.
    """

    filename = "trials.json"

    def resolve(self, term: str) -> Resolution | None:
        cached = self._get(term)
        if cached is not None:
            return Resolution(**cached) if cached else None
        out = None
        try:
            import requests

            if term.lower().startswith("rs") and term[2:].isdigit():
                _throttle()
                r = requests.get(f"{EUTILS}/esummary.fcgi",
                                 params={"db": "snp", "id": term[2:], "retmode": "json"},
                                 timeout=20)
                d = r.json().get("result", {}).get(term[2:], {}) if r.ok else {}
                genes = [g.get("name") for g in (d.get("genes") or []) if g.get("name")]
                if genes:
                    # The gene an rsID sits in is RELATED to it, not another name for it.
                    out = Resolution(term=term, kind="sequence variant", source="dbSNP",
                                     code=term, label=", ".join(genes),
                                     synonyms=genes[:MAX_SYNONYMS_PER_TERM],
                                     relation="association")
            else:
                out = self._trial(term)
        except Exception:  # noqa: BLE001
            out = None
        self._put(term, out.__dict__ if out else {})
        return out

    def _trial(self, term: str) -> Resolution | None:
        import requests

        want = norm_key(term)
        fields = {"fields": "NCTId,BriefTitle,Acronym,InterventionName"}
        if term.upper().startswith("NCT") and term[3:].isdigit():
            _throttle()
            r = requests.get(f"{CTGOV}/studies/{term.upper()}", params=fields, timeout=20)
            if not r.ok:
                return None
            p, nct = r.json()["protocolSection"], term.upper()
        else:
            _throttle()
            r = requests.get(f"{CTGOV}/studies",
                             params={"query.term": term, "pageSize": 5, **fields}, timeout=20)
            if not r.ok:
                return None
            p = None
            for st in (r.json().get("studies") or []):
                ident = st["protocolSection"].get("identificationModule", {})
                # Match the acronym field, or the acronym as printed in the title —
                # PALOMA-3 and MONARCH-2 are real trials whose acronym field is not
                # populated the way the literature spells them.
                if (norm_key(ident.get("acronym") or "") == want
                        or want in norm_key(ident.get("briefTitle") or "")):
                    p, nct = st["protocolSection"], ident.get("nctId") or ""
                    break
            if p is None:
                return None
        iv = [i.get("name") for i in
              (p.get("armsInterventionsModule", {}).get("interventions") or []) if i.get("name")]
        syns = [s for s in dict.fromkeys(iv)
                if _useful_synonym(s, term) and not _GENERIC_ARM.search(s)
                ][:MAX_SYNONYMS_PER_TERM]
        if not syns:
            return None
        # A trial is not another name for its drugs. Marked `association` so the caller
        # can decide whether to inject it — `Placebo`, before it was filtered here, was
        # being added to a PALOMA-3 query and matches every RCT in the corpus.
        return Resolution(term=term, kind="clinical trial", source="ClinicalTrials.gov",
                          code=nct, label=(p.get("identificationModule", {})
                                           .get("briefTitle") or "")[:120], synonyms=syns,
                          relation="association")


class OntologyExpander:
    """A cascade of registries, looked up **whole query first**.

    **Why the regex is no longer the arbiter.** The previous design used a character-class
    pattern to decide which spans were entities, then looked those up. Two measurements
    killed it:

    * **89.6% of identifier queries are a single token.** There is nothing to segment, so
      the recognizer was solving a problem this stratum does not have;
    * **17 of 17 multiword identifiers were destroyed by it.** ``EGFR T790M`` became
      ``EGFR``. That is worse than losing the expansion: it *broadens* a question about one
      resistance mutation into every EGFR paper in the corpus, which is the opposite of what
      was asked. NCIt holds ``EGFR T790M`` whole, as concept C98503.

    A pattern can only ever guess at *shape*, and shape does not carry identity. ``CTLA-4``
    (gene), ``UACC-812`` (cell line), ``CALAA-01`` (a nanoparticle drug), ``BOLERO-2`` (a
    trial) and ``QLQ-C30`` (a questionnaire) are indistinguishable to any regex, and the
    old one returned all five unchanged. **The dictionary decides what is an entity, and it
    returns the type along with the match** — that is the property being bought here, not
    better precision.

    So the whole query string is offered to each registry in turn, and the regex survives
    only as a fallback for genuinely multi-entity natural-language queries, where
    segmentation is a real problem rather than an invented one.

    Registry order follows measured coverage of the 335 identifier queries — NCIt 55.8%,
    ClinicalTrials.gov 22.7%, HGNC 13.7%, Cellosaurus 8.4%, dbSNP 0.9%, any 87.5%, against
    13.7% for the shipped expander. Each is free and unauthenticated. **No single one is
    sufficient**, which is the substantive answer to "where do oncology terms live": they do
    not live in one place, and a design that assumes they do will cover an eighth of them.
    """

    def __init__(self, cache_dir: Path | None = None,
                 email: str = "oncolens@example.com") -> None:
        self.ncit = NcitExpander(cache_dir=cache_dir)
        self.cells = CellosaurusExpander(cache_dir=cache_dir)
        self.trials = TrialExpander(cache_dir=cache_dir)
        self.hgnc = HgncExpander(cache_dir=cache_dir)
        self.mesh = MeshExpander(cache_dir=cache_dir, email=email)

    def resolve(self, term: str) -> Resolution | None:
        """First registry that recognises the term wins."""
        if len(norm_key(term)) < MIN_TERM_CHARS:
            return None
        for r in (self.ncit.resolve(term), self.cells.resolve(term),
                  self.trials.resolve(term)):
            if r and r.synonyms:
                return r
        for name, syns in (("gene", self.hgnc.synonyms(term)),
                           ("MeSH concept", self.mesh.synonyms(term))):
            if syns:
                return Resolution(term=term, kind=name,
                                  source="HGNC" if name == "gene" else "MeSH",
                                  label=term, synonyms=syns[:MAX_SYNONYMS_PER_TERM])
        return None

    def expand(self, query: str) -> Expansion:
        exp = Expansion(query=query)
        whole = query.strip()
        if whole:
            r = self.resolve(whole)
            if r:
                exp.terms[whole] = r.synonyms
                exp.resolutions[whole] = r
                return exp
        # Fallback only: the query is not itself a term, so it may contain several.
        for term in candidate_terms(query):
            r = self.resolve(term)
            if r:
                exp.terms[term] = r.synonyms
                exp.resolutions[term] = r
        return exp


_DEFAULT: MeshExpander | None = None
_ONTOLOGY: OntologyExpander | None = None


def expand_query(query: str, *, cache_dir: Path | None = None,
                 email: str = "oncolens@example.com",
                 source: str = "mesh") -> Expansion:
    """Expand ``query`` against curated registries.

    ``source="mesh"`` keeps the original single-vocabulary behaviour.

    ``source="ontology"`` runs the cascade in ``OntologyExpander``: the **whole query** is
    offered to NCIt, Cellosaurus, ClinicalTrials.gov, HGNC and MeSH in that order, and the
    regex is consulted only if the query is not itself a term. This is what the identifier
    stratum needs — 89.6% of its queries are a single token, and the multiword remainder
    (``EGFR T790M``) are single concepts that the regex used to split.
    """
    global _DEFAULT, _ONTOLOGY
    if source == "ontology":
        if _ONTOLOGY is None:
            _ONTOLOGY = OntologyExpander(cache_dir=cache_dir, email=email)
        return _ONTOLOGY.expand(query)
    if _DEFAULT is None:
        _DEFAULT = MeshExpander(cache_dir=cache_dir, email=email)
    return _DEFAULT.expand(query)
