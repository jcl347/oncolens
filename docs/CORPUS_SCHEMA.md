# OncoLens corpus & label schema (v1, frozen before retriever implementation)

This schema is authored **before** any retrieval code exists. That ordering is deliberate: it
prevents corpus/label content from being shaped to favor a retrieval technique.

## `data/corpus/documents.jsonl`

One JSON object per line.

```jsonc
{
  "doc_id": "GRANT:R01CA210111",        // GRANT:<activity+serial> or PAPER:PMID<n>
  "doc_type": "grant",                   // "grant" | "paper"
  "title": "…",
  "year": 2021,
  "sections": [                          // ordered; grants and papers use different section names
    {"name": "Abstract",       "text": "…"},
    {"name": "Specific Aims",  "text": "…"}
  ],
  "meta": {
    "pi": "…", "org": "…",
    "activity_code": "R01", "ic": "NCI",     // grants only
    "journal": "…", "pmid": "34xxxxxx"       // papers only
  },
  "descriptors": ["D:EGFR_TKI_RESISTANCE"],  // GOLD labels — see rules below
  "funded_by": ["GRANT:R01CA210111"],        // papers -> grants (the "found data" link)
  "cites": ["PAPER:PMID33000001"]            // papers -> papers
}
```

### Section names
- **Grants:** `Abstract`, `Specific Aims`, `Significance`, `Innovation`, `Approach`
- **Papers:** `Abstract`, `Introduction`, `Methods`, `Results`, `Discussion`

### Descriptor rules (this is what makes the eval honest)
`descriptors` emulate **NLM human MeSH indexing**: they are assigned on the document's
*semantics*, not its surface strings.

1. A descriptor MUST be assignable even when the document never contains the descriptor's
   preferred term or any of its synonyms. Aim for **≥30% of assignments to be "lexically silent"**
   — the concept is clearly present, the words are not.
2. Assign 3–8 descriptors per document, mixing broad and specific.
3. Do **not** consult `data/vocab/lexicon.json` when assigning descriptors. Gold labels and the
   retrieval-time synonym lexicon are separate artifacts on purpose (see below).

## `data/vocab/concepts.json` — gold concept space (label space)

```jsonc
{
  "D:EGFR_TKI_RESISTANCE": {
    "preferred": "EGFR tyrosine kinase inhibitor resistance",
    "broader": ["D:TARGETED_THERAPY_RESISTANCE"],
    "narrower": ["D:EGFR_T790M", "D:MET_AMPLIFICATION"]
  }
}
```

Used **only** to build queries and qrels. Never loaded by retrieval code.

## `data/vocab/lexicon.json` — retrieval-time synonym resource

A deliberately **imperfect** stand-in for UMLS/NCIt, as a real ontology would be:
missing synonyms, some over-broad entries, some entries for concepts absent from the corpus.

```jsonc
{ "osimertinib": ["AZD9291", "Tagrisso", "third-generation EGFR TKI"] }
```

Keyed by surface term, **not** by `D:` id. Contamination overlap with `concepts.json` is
computed and reported in every experiment, so ontology-expansion gains can be discounted.

## `data/qrels/<split>.jsonl` — graded, multi-relevant judgments

```jsonc
{
  "query_id": "Q0142",
  "query": "resistance to third-generation EGFR inhibitors",
  "stratum": "conceptual",     // see strata below
  "source": "descriptor",      // provenance of the labels
  "judgments": {"PAPER:PMID34000123": 3, "GRANT:R01CA210111": 2},
  "notes": "…"
}
```

**Graded scale:** `3` = directly on-topic / definitive, `2` = substantially relevant,
`1` = marginal or passing mention, `0` = explicitly judged non-relevant.
Judging a doc `0` is valuable — it distinguishes "known irrelevant" from "unjudged".

### Query strata (every stratum is reported separately; a regression in any one blocks promotion)
| Stratum | Tests | Example |
|---|---|---|
| `lexical` | exact rare strings | `NCT04185883`, `KRAS G12C` |
| `conceptual` | semantics w/o shared vocabulary | `resistance to third-generation EGFR inhibitors` |
| `paraphrase` | same idea, different words | `why do lung tumors stop responding to targeted drugs` |
| `multi_hop` | needs 2+ docs / a link | `papers from grants studying CDK4/6 escape` |
| `boolean_scope` | conjunction/negation | `CAR-T in solid tumors, not hematologic` |
| `no_answer` | correct action is to return nothing | `CRISPR base editing for sickle cell` |

### Label provenance (`source`) — mirrors real-world found data
| `source` | Real-world analogue | How built |
|---|---|---|
| `descriptor` | NLM MeSH human indexing | all docs carrying descriptor D |
| `funding_link` | NIH RePORTER grant→publication | grant aims text → its funded pubs |
| `citation_ctx` | citation contexts (SPECTER/SciNCL) | citing sentence → cited paper |
| `pooled` | TREC-style pooling | union of variant top-k, judged after the fact |

## Splits
- `dev` — tuning is allowed. Every evaluation against it is counted as a multiple-comparisons draw.
- `test` — **locked**. Read only at promotion time.
Split assignment is by hash of `query_id`, and *documents are not disjoint* across splits (correctly
so — the corpus is shared; only queries are split).
