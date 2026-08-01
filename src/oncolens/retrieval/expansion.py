"""Ontology / synonym query expansion, with contamination accounting.

Expansion is the oncology-specific lever: a user typing "EGFR inhibitor" should reach
documents that only ever say "osimertinib". But it is also the single easiest way to
*fake* an improvement on a synthetic benchmark, so this module carries its own audit.

**The contamination hazard.** If the synonym resource used at retrieval time were derived
from the same artifact as the gold labels, expansion would win by construction and the
measured gain would be meaningless. ``contamination_report`` quantifies the overlap
between the retrieval lexicon and the gold concept space so every experiment can state how
much of an expansion gain is potentially artifactual. A gain that survives only because of
overlap is not a gain.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .text import tokenize


class Lexicon:
    """Surface-term -> variants. Deliberately imperfect, like a real ontology dump."""

    def __init__(self, mapping: Mapping[str, Sequence[str]] | None = None) -> None:
        self.mapping: dict[str, list[str]] = {
            k.lower(): [v for v in vs] for k, vs in (mapping or {}).items()
        }
        # Longest-first so multi-word entries ("egfr tyrosine kinase inhibitor") are
        # matched before their single-word constituents.
        self._keys = sorted(self.mapping, key=len, reverse=True)

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        p = Path(path)
        if not p.exists():
            return cls({})
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def expand_query(
        self, query: str, *, max_variants: int = 4, max_total_added: int = 24
    ) -> tuple[str, list[str]]:
        """Return (expanded_query_text, added_terms).

        Caps exist because unbounded expansion is a known precision killer: every added
        synonym dilutes the query's IDF mass and drags in off-topic documents. The caps are
        exposed as knobs so the loop can measure the precision/recall trade rather than
        assume it.
        """
        low = query.lower()
        added: list[str] = []
        seen = set(tokenize(query))
        for key in self._keys:
            if len(added) >= max_total_added:
                break
            if key in low:
                for variant in self.mapping[key][:max_variants]:
                    for tok in tokenize(variant):
                        if tok not in seen:
                            seen.add(tok)
                            added.append(tok)
        return (query + " " + " ".join(added)).strip(), added


def contamination_report(
    lexicon_path: str | Path, concepts_path: str | Path
) -> dict:
    """Quantify overlap between the retrieval lexicon and the gold concept space.

    Reported in every experiment. High overlap means expansion gains must be discounted;
    it does not by itself invalidate the run, but an unreported overlap would.
    """
    lex = Lexicon.load(lexicon_path)
    cp = Path(concepts_path)
    concepts = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}

    concept_tokens: set[str] = set()
    for meta in concepts.values():
        concept_tokens |= set(tokenize(meta.get("preferred", "")))

    lex_tokens: set[str] = set()
    for key, variants in lex.mapping.items():
        lex_tokens |= set(tokenize(key))
        for v in variants:
            lex_tokens |= set(tokenize(v))

    overlap = concept_tokens & lex_tokens
    return {
        "concept_preferred_tokens": len(concept_tokens),
        "lexicon_tokens": len(lex_tokens),
        "overlapping_tokens": len(overlap),
        "jaccard": (len(overlap) / len(concept_tokens | lex_tokens)) if (concept_tokens | lex_tokens) else 0.0,
        "concept_coverage_by_lexicon": (len(overlap) / len(concept_tokens)) if concept_tokens else 0.0,
        "interpretation": (
            "Fraction of gold-concept vocabulary the retrieval lexicon can reach. High values "
            "mean ontology-expansion gains are partly artifactual and must be discounted."
        ),
    }
