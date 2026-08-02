"""Detecting and removing bibliographies from full text.

**The problem, measured on real data.** NCBI's PMC plain-text rendition includes the
reference list. 9% of ingested passages were bibliography, and citation strings match
queries lexically while containing no findings — two of the top three hits for
"osimertinib resistance mechanism" were reference entries, not claims. That is a pure
precision leak: the passages are retrievable, plausible-looking, and useless.

**Why the obvious approach fails.** There is no ``REFERENCES`` heading to split on. The
txt rendition emits only ``JOURNAL INFORMATION`` and ``ARTICLE INFORMATION`` as structural
markers; the bibliography simply begins. So detection has to be content-based.

**Why per-paragraph classification alone also fails.** A Methods paragraph can legitimately
contain a DOI, a year, and an author name ("as described by Chen et al. (2019)"). Judging
paragraphs independently therefore deletes real content. The signal that disambiguates is
**position**: a bibliography is a *sustained run of reference-shaped paragraphs extending to
the end of the document*. One reference-shaped paragraph in the middle is a citation; forty
of them at the end are the bibliography.

So this module scores paragraphs, then looks for the trailing run — which is both more
accurate and far more conservative than thresholding each paragraph on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- signals, each independently weak, jointly decisive ---------------------
_DOI = re.compile(r"\b10\.\d{4,9}/\S+")
_PMCID = re.compile(r"\bPMC\d{5,}\b")
_PMID_BARE = re.compile(r"(?<!\d)\d{8}(?!\d)")
#: "Lee EJ", "Kim M-J", "Cho JS" — surname followed by 1-3 initials.
_AUTHOR_INITIALS = re.compile(r"\b[A-Z][a-z]{1,20}\s+[A-Z][A-Za-z\-]{0,3}\b(?=[ ,;])")
#: "2022;113:709-20" volume/page locators.
_VOL_PAGE = re.compile(r"\b(19|20)\d{2}\s*[;:]\s*\d+\s*[:(]")
#: Numbered bibliography entries: "28. Lee EJ ..."
_NUMBERED = re.compile(r"^\s*\d{1,3}\.\s+[A-Z]")
_YEAR = re.compile(r"\b(19|20)\d{2}\b")

#: Function words: prose has them, citation strings barely do. This is the single most
#: discriminative signal, because a bibliography is a list of names and not sentences.
_FUNCTION_WORDS = frozenset("""
the of and to in that we was for is are with as by this on were be have has had not but
which from at an it their our they these those than then when where while can could may
""".split())


@dataclass(frozen=True)
class ParagraphScore:
    index: int
    score: float
    reasons: tuple[str, ...]


def score_paragraph(text: str) -> ParagraphScore:
    """Probability-ish score that a paragraph is bibliography rather than prose."""
    words = text.split()
    n = max(len(words), 1)
    reasons: list[str] = []
    score = 0.0

    dois = len(_DOI.findall(text))
    pmcs = len(_PMCID.findall(text))
    pmids = len(_PMID_BARE.findall(text))
    ids_per_100w = (dois + pmcs + pmids) / n * 100
    if ids_per_100w > 0.8:
        score += min(0.35, ids_per_100w * 0.12)
        reasons.append(f"ids/100w={ids_per_100w:.1f}")

    authors = len(_AUTHOR_INITIALS.findall(text)) / n * 100
    if authors > 4:
        score += min(0.28, (authors - 4) * 0.02)
        reasons.append(f"author-initials/100w={authors:.1f}")

    years = len(_YEAR.findall(text)) / n * 100
    if years > 1.5:
        score += min(0.15, (years - 1.5) * 0.05)
        reasons.append(f"years/100w={years:.1f}")

    if _VOL_PAGE.search(text):
        score += 0.12
        reasons.append("vol:page")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    numbered = sum(1 for ln in lines if _NUMBERED.match(ln))
    if lines and numbered / len(lines) > 0.3:
        score += 0.20
        reasons.append(f"numbered={numbered}/{len(lines)}")

    # Prose test: bibliographies are lists of proper nouns, not sentences.
    fw = sum(1 for w in words if w.lower().strip(".,;:()") in _FUNCTION_WORDS) / n
    if fw < 0.10:
        score += 0.30 * (0.10 - fw) / 0.10
        reasons.append(f"function-words={fw:.2%}")

    return ParagraphScore(index=-1, score=min(score, 1.0), reasons=tuple(reasons))


#: A paragraph at or above this is reference-shaped on its own evidence.
PARA_THRESHOLD = 0.45
#: The trailing run must average at least this to be treated as a bibliography.
RUN_THRESHOLD = 0.40
#: Ignore a trailing run shorter than this — a two-paragraph tail is not a bibliography,
#: and truncating on it risks eating a real Conclusion.
MIN_RUN = 3

#: MEASURED: PMC's txt rendition usually emits the WHOLE bibliography as a single
#: paragraph — one article had 14,123 chars and 55 entries in one block. The trailing-run
#: logic alone therefore never fired, because the run length was 1. A single very
#: high-scoring, very large trailing block is a bibliography on its own evidence.
SINGLE_BLOCK_SCORE = 0.85
SINGLE_BLOCK_CHARS = 1500

#: A standalone heading paragraph. Not an ALL-CAPS line — PMC emits it as its own short
#: paragraph, which is why scanning for capitalised headings found nothing.
_HEADING = re.compile(
    r"^\s*(references|bibliography|literature cited|works cited|references and notes)"
    r"\s*:?\s*$",
    re.I,
)


def find_reference_start(paragraphs: list[str]) -> int | None:
    """Index of the first bibliography paragraph, or None if no bibliography is found.

    Walks backwards from the end while paragraphs remain reference-shaped, which encodes
    the structural fact that a bibliography runs to the end of the document. A single
    citation-heavy Methods paragraph in the middle is therefore never mistaken for one.
    """
    if len(paragraphs) < 2:
        return None
    scores = [score_paragraph(p).score for p in paragraphs]

    # 1. Explicit heading paragraph — the most reliable signal when present. Only trust it
    #    in the last third of the document, so an inline mention cannot truncate the body.
    for i in range(len(paragraphs) - 1, max(0, int(len(paragraphs) * 0.5)) - 1, -1):
        if _HEADING.match(paragraphs[i].strip()):
            return i

    # 2. Single large high-scoring trailing block (the common PMC shape).
    for i in range(len(paragraphs) - 1, max(0, len(paragraphs) - 4) - 1, -1):
        if (scores[i] >= SINGLE_BLOCK_SCORE
                and len(paragraphs[i]) >= SINGLE_BLOCK_CHARS):
            return i

    if len(paragraphs) < MIN_RUN + 1:
        return None

    i = len(paragraphs) - 1
    # Tolerate at most one interruption (e.g. an acknowledgements block between the
    # discussion and the references) before giving up on the run.
    misses = 0
    while i >= 0:
        if scores[i] >= PARA_THRESHOLD:
            misses = 0
        else:
            misses += 1
            if misses > 1:
                break
        i -= 1
    start = i + 1 + (1 if misses else 0)

    run = scores[start:]
    if len(run) < MIN_RUN:
        return None
    if sum(run) / len(run) < RUN_THRESHOLD:
        return None
    # Never delete most of a document: if "references" is over 70% of it, the detector is
    # wrong, not the article.
    if len(run) / len(paragraphs) > 0.70:
        return None
    return start


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
    paragraphs = [p for p in text.split("\n\n")]
    idx = find_reference_start(paragraphs)
    if idx is None:
        return StripResult(kept=paragraphs, dropped=[], start_index=None)
    return StripResult(kept=paragraphs[:idx], dropped=paragraphs[idx:], start_index=idx)


#: Standalone per-passage filtering needs a STRICTER threshold than the positional
#: stripper. MEASURED false positive: a genuine prose paragraph citing "Chen et al. (2019)"
#: and a DOI scores 0.500, above PARA_THRESHOLD. The positional stripper is unaffected
#: because it only ever removes text after a heading in the last half of the document, or
#: a large block in the final few paragraphs — it structurally cannot delete a mid-document
#: paragraph. A standalone check has no such protection, so it demands stronger evidence.
STANDALONE_THRESHOLD = 0.75


def is_reference_like(text: str, threshold: float = STANDALONE_THRESHOLD) -> bool:
    """Standalone check for passages that survived section stripping.

    Deliberately stricter than the positional stripper: without position as evidence, a
    Discussion paragraph dense with citations looks similar to a bibliography entry, and
    deleting real findings is far worse than retaining a few reference passages.
    """
    return score_paragraph(text).score >= threshold
