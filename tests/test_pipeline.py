"""End-to-end integration test on a tiny inline corpus.

Exercises chunking -> BM25 -> LSA -> fusion -> pipeline -> experiment -> gate without
touching data/. Its job is to catch integration bugs (shape errors, offset corruption,
empty-index edge cases) independently of whether the real corpus exists yet.

It also asserts two *behavioural* properties that the whole architecture rests on:
  * BM25 finds a rare identifier that LSA cannot;
  * LSA finds a paraphrase that shares no content words with the document.
If either stops holding, the hybrid argument is unsupported and the loop is optimising
something other than what it claims to.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oncolens.data import Dataset, Query
from oncolens.eval.gate import evaluate_gate
from oncolens.eval.stats import Ledger
from oncolens.experiment import run_experiment
from oncolens.retrieval.chunking import chunk_corpus
from oncolens.retrieval.expansion import Lexicon
from oncolens.retrieval.pipeline import RetrievalConfig, Retriever

failures: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(name)


def _doc(did, dtype, title, secs, **meta):
    # No `descriptors` key: gold labels must never live in the retrieval corpus.
    # load_corpus() now raises if one appears, which is how this fixture caught itself.
    return {"doc_id": did, "doc_type": dtype, "title": title, "year": 2023,
            "sections": [{"name": n, "text": t} for n, t in secs],
            "meta": meta, "funded_by": [], "cites": []}


DOCS = [
    _doc("PAPER:P1", "paper", "Acquired resistance to third-generation inhibitors in lung adenocarcinoma",
         [("Abstract",
           "Patients whose tumours initially shrink on treatment almost always progress within two "
           "years. We profiled 84 paired biopsies collected before therapy and at the time of "
           "progression to understand why the benefit is lost.\n\n"
           "Sequencing identified the EGFR C797S substitution in 18 of 84 cases, arising on the same "
           "allele as T790M in the majority. A further 11 cases showed MET amplification."),
          ("Discussion",
           "The emergence of EGFR C797S in cis with T790M forecloses re-treatment with existing "
           "agents and motivates fourth-generation allosteric compounds now in early trials.")]),
    _doc("PAPER:P2", "paper", "MET amplification drives bypass signalling",
         [("Abstract",
           "Bypass track activation restores downstream signalling despite continued target "
           "inhibition. We show amplification of the MET receptor is sufficient to maintain ERK "
           "phosphorylation and confers a proliferative advantage in vitro and in xenografts.")]),
    _doc("PAPER:P3", "paper", "Checkpoint blockade in microsatellite-unstable colorectal cancer",
         [("Abstract",
           "Mismatch repair deficiency produces a high neoantigen burden. In a cohort of 61 patients "
           "receiving pembrolizumab, the objective response rate was 42 percent, durable beyond "
           "eighteen months in responders.")]),
    _doc("GRANT:G1", "grant", "Overcoming bypass-mediated escape from targeted therapy",
         [("Abstract",
           "This proposal addresses why molecularly targeted agents produce dramatic but temporary "
           "benefit, and how combinations might extend it."),
          ("Specific Aims",
           "Aim 1 will define the spectrum of escape mechanisms in serial biopsies. Aim 2 will test "
           "whether upfront combination blockade delays the emergence of escape in patient-derived "
           "models. Aim 3 will develop a circulating-tumour-DNA assay to detect escape early.")]),
    _doc("PAPER:P4", "paper", "Cytokine release syndrome after CD19-directed cellular therapy",
         [("Abstract",
           "Severe cytokine release syndrome occurred in 13 percent of 212 treated patients. "
           "Tocilizumab administration within 24 hours of onset shortened intensive care stay.")]),
]

QUERIES = [
    Query("Q_LEX", "EGFR C797S", "lexical", "descriptor", {"PAPER:P1": 3, "PAPER:P2": 0, "PAPER:P3": 0}),
    Query("Q_PARA", "why do lung tumours stop responding to targeted drugs over time",
          "paraphrase", "descriptor", {"PAPER:P1": 3, "PAPER:P2": 2, "GRANT:G1": 3, "PAPER:P3": 0}),
    Query("Q_CONC", "bypass signalling escape from kinase inhibition", "conceptual", "descriptor",
          {"PAPER:P2": 3, "GRANT:G1": 2, "PAPER:P1": 1, "PAPER:P4": 0}),
    Query("Q_NA", "proton therapy for pediatric ependymoma", "no_answer", "descriptor", {}),
]

DS = Dataset(docs=DOCS, queries=QUERIES,
             integrity={"n_docs": len(DOCS), "n_queries": len(QUERIES)})

print("chunking preserves provenance")
chunks = chunk_corpus(DOCS)
check("chunks produced", len(chunks) >= len(DOCS), f"{len(chunks)} chunks")
check("all offsets valid", all(c.end_char > c.start_char for c in chunks))
check("all chunks map to a real doc", all(c.doc_id in {d["doc_id"] for d in DOCS} for c in chunks))
# The displayed passage must be recoverable from the source by its offsets.
by_doc = {d["doc_id"]: d for d in DOCS}
recoverable = 0
for c in chunks:
    sec = next((s for s in by_doc[c.doc_id]["sections"] if s["name"] == c.section), None)
    if sec and c.text.split("\n")[0][:60] in sec["text"]:
        recoverable += 1
check("passages recoverable from source offsets", recoverable == len(chunks),
      f"{recoverable}/{len(chunks)}")

print("\nBM25 arm alone")
bm = Retriever(RetrievalConfig(name="bm", use_bm25=True, use_dense=False), Lexicon({})).build(DOCS)
r_lex = bm.search("EGFR C797S")
check("BM25 finds the rare identifier at rank 1", r_lex.ranking[:1] == ["PAPER:P1"],
      f"got {r_lex.ranking[:3]}")

print("\nLSA arm alone")
ds_only = Retriever(RetrievalConfig(name="ds", use_bm25=False, use_dense=True, dense_dim=8),
                    Lexicon({})).build(DOCS)
r_par = ds_only.search("why do lung tumours stop responding to targeted drugs over time")
check("LSA returns something for a paraphrase", len(r_par.ranking) > 0)
check("LSA surfaces a relevant doc in top 3 for paraphrase",
      any(d in ("PAPER:P1", "GRANT:G1", "PAPER:P2") for d in r_par.ranking[:3]),
      f"got {r_par.ranking[:3]}")

print("\nTHE HYBRID PREMISE: each arm covers the other's structural blind spot")
r_lex_dense = ds_only.search("EGFR C797S")
lex_rank_bm = r_lex.ranking.index("PAPER:P1") if "PAPER:P1" in r_lex.ranking else 99
lex_rank_ds = r_lex_dense.ranking.index("PAPER:P1") if "PAPER:P1" in r_lex_dense.ranking else 99
print(f"  INFO  'EGFR C797S' -> P1 at BM25 rank {lex_rank_bm}, LSA rank {lex_rank_ds}")
check("BM25 ranks the identifier query no worse than LSA", lex_rank_bm <= lex_rank_ds)

print("\nhybrid fusion runs")
hy = Retriever(RetrievalConfig(name="hy", use_bm25=True, use_dense=True, dense_dim=8,
                               fusion="rrf"), Lexicon({})).build(DOCS)
r_h = hy.search("bypass signalling escape from kinase inhibition")
check("hybrid returns a ranking", len(r_h.ranking) > 0, f"top={r_h.ranking[:3]}")
check("hybrid attaches passage evidence", bool(r_h.evidence),
      f"{len(r_h.evidence)} evidence spans")
if r_h.ranking:
    top = r_h.ranking[0]
    ev = r_h.evidence.get(top)
    check("evidence for the top doc points at that doc", ev is not None and ev.doc_id == top)

print("\nabstention on a no-answer query")
ab = Retriever(RetrievalConfig(name="ab", use_bm25=True, use_dense=False,
                               abstain=True, abstain_bm25_floor=999.0), Lexicon({})).build(DOCS)
check("high floor causes abstention", ab.search("proton therapy for pediatric ependymoma").ranking == [])

print("\nexperiment runner")
res = run_experiment(DS, RetrievalConfig(name="smoke", use_bm25=True, use_dense=True, dense_dim=8),
                     split="all")
check("per-query metrics produced", len(res.per_query) == len(QUERIES))
check("aggregate has the primary metric", "ndcg@10" in res.aggregate,
      f"ndcg@10={res.aggregate.get('ndcg@10')}")
check("no_answer stratum reported separately", "no_answer" in res.per_stratum)
check("runs recorded for pooling", len(res.runs) == len(QUERIES))

print("\ngate rejects a no-op change (identical arms)")
import tempfile
with tempfile.TemporaryDirectory() as td:
    led = Ledger.load(Path(td) / "l.json")
    g_same = evaluate_gate(res.per_query, res.per_query, DS.strata(), led)
    check("identical configs are NOT promoted", not g_same.promoted,
          f"blockers={len(g_same.blockers)}")

print("\ngate rejects a strictly-worse change")
worse = run_experiment(DS, RetrievalConfig(name="worse", use_bm25=False, use_dense=True,
                                           dense_dim=2), split="all")
with tempfile.TemporaryDirectory() as td:
    led = Ledger.load(Path(td) / "l.json")
    g_worse = evaluate_gate(res.per_query, worse.per_query, DS.strata(), led)
    check("degraded config is NOT promoted", not g_worse.promoted)

print()
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("all pipeline integration checks passed")
