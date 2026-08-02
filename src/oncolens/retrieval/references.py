"""Detecting and removing bibliographies from full text.

**The problem, measured on real data.** NCBI's PMC plain-text rendition includes the
reference list, and it is not a rounding error: across 60 articles with publisher-labelled
boundaries, the bibliography is a **median 19% of all characters** (range 5.6%–42.4%).
Citation strings match queries lexically while containing no findings — two of the top
three hits for "osimertinib resistance mechanism" were reference entries, not claims. That
is a pure precision leak: retrievable, plausible-looking, useless.

**Why the obvious approach fails.** There is usually no ``REFERENCES`` heading to split on.
The txt rendition emits only ``JOURNAL INFORMATION`` and ``ARTICLE INFORMATION`` as
structural markers; the bibliography simply begins. Detection has to be content-based.

**Why the first version of this module failed on ~1 in 6 articles.** It scored paragraphs
and looked for a trailing *run* of at least three. But PMC frequently emits the entire
bibliography as **one paragraph** — one article had 14,123 characters and 55 entries in a
single block — so the run length was 1 and the rule never fired. It also normalised every
signal *per word*, which makes a 14k-character block look like a 200-character one, and it
required ``[ ,;]`` after author initials, so the ACS/Nature style ``Zhou J. Xu Y.`` matched
nothing at all.

**How this version was built.** ``<ref-list>`` in PMC's JATS XML is the publisher's own
statement of where the bibliography starts (see ``sources/jats.py``). Aligning it onto the
plain text gives labelled data, and every candidate signal was scored on it
(``scripts/analyze_ref_signals.py``). Measured pairwise AUC over 60 articles:

===========================  =========  =========  =====
signal (per 1000 chars)      body med   refs med    AUC
===========================  =========  =========  =====
years                            0.254      5.952  1.000
dois                             0.034      4.090  1.000
author initials (tolerant)       0.730     24.503  1.000
function-word fraction           0.230      0.086  0.000
page ranges                      0.255      2.431  0.983
author initials (strict, old)    0.588     13.303  0.900
numbered entries                 0.000      0.000  0.692  <- noise
volume:page                      0.000      0.000  0.533  <- noise
===========================  =========  =========  =====

So the signals were never the problem — separation is essentially perfect at block level.
The problems were per-word normalisation, the run requirement, and two hand-weighted
signals that turned out to be pure noise. This version therefore:

* normalises **per 1000 characters**, so block size does not distort the evidence;
* drops ``numbered`` and ``volume:page`` entirely rather than diluting with them;
* finds the boundary by **suffix search** — the earliest point from which everything to
  the end is reference-dense — which is agnostic to whether the bibliography arrives as
  one paragraph or fifty.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- signals, each independently strong, jointly decisive -------------------
_DOI = re.compile(r"\b10\.\d{4,9}/\S+")
_PMCID = re.compile(r"\bPMC\d{5,}\b")
_PMID_BARE = re.compile(r"(?<!\d)\d{8}(?!\d)")
#: Surname + initials. The trailing character class **includes ``.``** so that the ACS /
#: Nature style ``Zhou J. Xu Y. Liu J.`` matches. The previous version required ``[ ,;]``
#: and scored 0.900 AUC instead of 1.000 purely because of that omission.
_AUTHOR_INITIALS = re.compile(r"\b[A-Z][a-z]{1,20}\s+[A-Z][A-Za-z.\-]{0,3}(?=[\s,;.])")
#: Page ranges: "709-20", "1123–1130". Pervasive in citations, rare in prose.
_PAGE_RANGE = re.compile(r"\b\d{1,5}\s*[-–]\s*\d{1,5}\b")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

#: Function words: prose has them, citation strings barely do. Measured as the single most
#: reliable signal (AUC 0.000 — i.e. perfectly *inverted*), because a bibliography is a
#: list of names, not sentences.
_FUNCTION_WORDS = frozenset("""
the of and to in that we was for is are with as by this on were be have has had not but
which from at an it their our they these those than then when where while can could may
""".split())


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 below ``lo``, 1 above ``hi``, linear between.

    ``lo``/``hi`` are set from the measured body/bibliography medians in the table above,
    so each signal saturates where real text actually saturates rather than at a round
    number someone liked.
    """
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


#: Weights sum to 1.0. Function words and author initials carry the most because they are
#: the least style-dependent: a journal may omit DOIs or use author-year instead of
#: numbered entries, but every bibliography is names rather than sentences.
_WEIGHTS = {"function_words": 0.30, "authors": 0.25, "years": 0.20,
            "ids": 0.15, "page_ranges": 0.10}


@dataclass(frozen=True)
class ParagraphScore:
    index: int
    score: float
    reasons: tuple[str, ...]


def reference_density(text: str) -> float:
    """How bibliography-like a block of text is, in [0, 1].

    Normalised **per 1000 characters**, not per word. That distinction is what makes the
    score comparable between a 200-character citation and a 14,000-character block — the
    exact case that defeated the previous implementation.
    """
    if not text.strip():
        return 0.0
    per_k = 1000.0 / max(len(text), 1)
    words = text.split()
    n_words = max(len(words), 1)

    ids = (len(_DOI.findall(text)) + len(_PMCID.findall(text))
           + len(_PMID_BARE.findall(text))) * per_k
    authors = len(_AUTHOR_INITIALS.findall(text)) * per_k
    years = len(_YEAR.findall(text)) * per_k
    pages = len(_PAGE_RANGE.findall(text)) * per_k
    fw = sum(1 for w in words if w.lower().strip(".,;:()") in _FUNCTION_WORDS) / n_words

    parts = {
        # Inverted: prose sits near 0.23, bibliography near 0.09.
        "function_words": _ramp(0.23 - fw, 0.04, 0.13),
        "authors": _ramp(authors, 2.0, 15.0),
        "years": _ramp(years, 0.6, 4.0),
        "ids": _ramp(ids, 0.15, 2.5),
        "page_ranges": _ramp(pages, 0.4, 2.0),
    }
    return sum(_WEIGHTS[k] * v for k, v in parts.items())


def score_paragraph(text: str) -> ParagraphScore:
    """Back-compatible wrapper around :func:`reference_density`."""
    d = reference_density(text)
    return ParagraphScore(index=-1, score=d, reasons=(f"density={d:.3f}",))


#: A block at or above this is bibliography. Set between the measured body and
#: bibliography distributions, which do not overlap at block scale.
BLOCK_THRESHOLD = 0.55
#: A paragraph may be absorbed into the bibliography suffix at this lower bar *provided*
#: the suffix as a whole stays above ``BLOCK_THRESHOLD``. Bibliographies contain the
#: occasional short or atypical entry; requiring every entry to clear the full bar would
#: leave the head of the reference list behind.
ABSORB_THRESHOLD = 0.35
#: Short paragraphs carry too little text to score reliably ("Supplementary Material",
#: "Conflicts of interest"), so they are absorbed on the suffix's evidence, not their own.
SHORT_PARA_CHARS = 320
#: MEASURED: the largest true bibliography in the labelled set is 42.4% of characters.
#: Refusing to cut more than 60% is a safety rail against a pathological document, not a
#: tuned parameter — it should never bind on real input.
MAX_SHARE = 0.60

#: Legacy names kept so existing imports and tests continue to resolve.
PARA_THRESHOLD = BLOCK_THRESHOLD
RUN_THRESHOLD = ABSORB_THRESHOLD

#: A standalone heading paragraph. Not an ALL-CAPS line — PMC emits it as its own short
#: paragraph, which is why scanning for capitalised headings found nothing.
_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited|references and notes)"
    r"\s*:?\s*$",
    re.I,
)


def find_reference_start(paragraphs: list[str]) -> int | None:
    """Index of the first bibliography paragraph, or None if there is none.

    **Suffix search.** Walk backwards from the end, extending a candidate bibliography one
    paragraph at a time, and keep the *earliest* start for which the whole suffix is
    reference-dense. This replaces the previous "trailing run of >= 3 reference-shaped
    paragraphs" rule, which could not fire when PMC emitted the bibliography as a single
    paragraph — the measured cause of the ~1-in-6 miss rate.

    Because the test is applied to the accumulated suffix rather than to paragraphs
    individually, it is insensitive to how the rendition happens to break lines, which is
    precisely the thing that varies between publishers.
    """
    if len(paragraphs) < 2:
        return None

    total_chars = sum(len(p) for p in paragraphs) or 1
    limit = max(1, int(len(paragraphs) * 0.30))  # never start before 30% of the way in

    # 1. An explicit heading is the publisher telling us directly. Trust it only in the
    #    back half, so an inline mention of "References" cannot truncate the body.
    for i in range(len(paragraphs) - 1, max(0, len(paragraphs) // 2) - 1, -1):
        if _HEADING.match(paragraphs[i].strip()):
            return i

    # 2. Suffix search.
    best: int | None = None
    suffix_chars = 0
    misses = 0
    for i in range(len(paragraphs) - 1, limit - 1, -1):
        para = paragraphs[i]
        suffix_chars += len(para)
        if suffix_chars / total_chars > MAX_SHARE:
            break

        d_para = reference_density(para)
        short = len(para.strip()) < SHORT_PARA_CHARS
        if d_para < ABSORB_THRESHOLD and not short:
            misses += 1
            # Tolerate at most one non-reference block inside the tail (acknowledgements
            # and data-availability statements routinely sit between body and references).
            if misses > 1:
                break
        else:
            misses = 0

        suffix = "\n\n".join(paragraphs[i:])
        if reference_density(suffix) >= BLOCK_THRESHOLD and d_para >= ABSORB_THRESHOLD:
            best = i

    if best is None:
        return None
    # Guard: the surviving body must still be the majority of the article.
    dropped = sum(len(p) for p in paragraphs[best:])
    if dropped / total_chars > MAX_SHARE:
        return None
    return best


@dataclass
class StripResult:
    kept: list[str]
    dropped: list[str]
    start_index: int | None

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def strip_references(text: str) -> StripResult:
    """Remove the bibliography from a full-text body, preserving paragraph structure."""
    paragraphs = list(text.split("\n\n"))
    idx = find_reference_start(paragraphs)
    if idx is None:
        return StripResult(kept=paragraphs, dropped=[], start_index=None)
    return StripResult(kept=paragraphs[:idx], dropped=paragraphs[idx:], start_index=idx)


#: Standalone per-passage filtering needs a STRICTER bar than the positional stripper.
#: MEASURED false positive: a genuine prose paragraph citing "Chen et al. (2019)" with a
#: DOI scores well above the block threshold. The positional stripper is unaffected because
#: it can only ever remove a *suffix*; a standalone check has no such protection, so it
#: demands stronger evidence. Retaining a few reference passages is far cheaper than
#: deleting real findings.
STANDALONE_THRESHOLD = 0.72


def is_reference_like(text: str, threshold: float = STANDALONE_THRESHOLD) -> bool:
    """Standalone check for passages that survived section stripping."""
    return reference_density(text) >= threshold
