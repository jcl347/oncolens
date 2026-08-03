# Getting a UMLS licence and API key

Everything OncoLens does today works **without** this. NCI Thesaurus, Cellosaurus,
ClinicalTrials.gov, dbSNP and HGNC are all unauthenticated, and together they resolve
**87.2%** of the identifier stratum (§4.14). Read this as the upgrade path, not a blocker.

---

## Why bother, and what it actually buys

UMLS is the Metathesaurus: roughly 200 source vocabularies mapped onto one another so that
every surface form for a concept — MeSH's, SNOMED's, NCIt's, HGNC's, RxNorm's — hangs off a
single **CUI** (Concept Unique Identifier).

**What it adds over the current cascade:**

| | today (5 free registries) | with UMLS |
|---|---|---|
| oncology terms | NCIt, 212,475 concepts | same NCIt, **plus** its mappings into SNOMED/MeSH/RxNorm |
| synonym merging | per-registry, unioned by us | pre-merged by NLM under one CUI |
| non-oncology terms | thin | comorbidities, procedures, devices, findings |
| sense disambiguation | our `_SEMANTIC_RANK` heuristic | same UMLS semantic types, but consistently applied across all sources |
| cost | free, no account | free, requires an account |

**What it does not add.** NCIt *is a source vocabulary inside UMLS*. The oncology-specific
coverage that matters most here is already reachable. A UTS key mainly buys breadth outside
oncology and cross-source CUI merging — real, but incremental. Do not expect a coverage
jump like 58.5% → 87.2%; expect something much smaller on this stratum.

---

## Step 1 — request the licence

1. Go to <https://uts.nlm.nih.gov/uts/signup-login>
2. Sign in with a supported identity provider (Google, Login.gov, eRA Commons, or an
   existing NIH account). There is no separate UTS password to create.
3. Complete the **UMLS Metathesaurus License Request**. You will be asked for:
   - name, email, country, and organisation
   - **the purpose of use** — for this project, "research: information retrieval over
     biomedical literature" is the accurate description
   - agreement to the [UMLS Metathesaurus License Agreement](https://uts.nlm.nih.gov/uts/assets/LicenseAgreement.pdf)

**Cost: free.** **Approval: usually automatic and immediate**; NLM states up to 3 business
days if a request is flagged for review.

⚠️ **The licence carries real obligations.** Some source vocabularies inside UMLS have
their own restrictions — SNOMED CT requires your country to have an IHTSDO member licence,
and a few sources are excluded from commercial use. This is the same category of issue as
§3.1's `LICENSE_POLICIES`: if OncoLens is ever commercialised, the source-level restrictions
need reading, not assuming. NLM publishes the per-source terms in the
[License Agreement Appendix](https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement_appendix.html).

You must also file a **brief annual usage report** — a short form, but it is a condition of
keeping the licence.

## Step 2 — get the API key

1. Sign in at <https://uts.nlm.nih.gov/uts/>
2. Open **My Profile**
3. Copy the **API Key** shown there

The key is a UUID. It is a credential: it belongs in `.env`, which this repo already
`.gitignore`s.

## Step 3 — wire it in

Add to `.env`:

```
UMLS_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

Verify it works (this is the exact call that returned **401** before the key existed):

```bash
python - <<'PY'
import os, requests
from pathlib import Path
from oncolens.env import load_env
load_env(Path("."))
r = requests.get("https://uts-ws.nlm.nih.gov/rest/search/current",
                 params={"string": "COX-2", "apiKey": os.environ["UMLS_API_KEY"]},
                 timeout=30)
print(r.status_code)
for x in r.json()["result"]["results"][:5]:
    print(" ", x["ui"], x["rootSource"], x["name"])
PY
```

Expect `200` and a list of CUIs. `401` means the key is wrong or the licence is not yet
approved; **`403` means the licence is approved but the specific source is restricted** in
your country, which is the SNOMED case above.

### Where the code changes

**One place.** `OntologyExpander.resolve` in [`src/oncolens/terminology.py`](../src/oncolens/terminology.py)
is a cascade of registries tried in order; UMLS becomes another entry. The existing
`_CachedResolver` base class supplies the on-disk cache, and the same
`Resolution(term, kind, source, code, label, synonyms, ambiguous, senses)` shape applies —
`kind` maps to the UMLS **semantic type**, `code` to the **CUI**.

Sketch:

```python
class UmlsExpander(_CachedResolver):
    filename = "umls_concepts.json"

    def resolve(self, term):
        # /search/current  -> CUI
        # /content/current/CUI/{cui}/atoms  -> every surface form, all sources
        # Rank senses with _SEMANTIC_RANK exactly as NcitExpander does; the semantic
        # types are the same UMLS network in both cases.
        ...
```

⚠️ **Do not put it first in the cascade without measuring.** NCIt is oncology-curated and
its relevance ordering is tuned for cancer terminology; UMLS spans all of medicine, so for
a bare oncology symbol it has *more* wrong senses to choose between, not fewer. §4.14's
`ER` case (14 NCIt concepts, of which 13 discarded) gets worse, not better, with 200
vocabularies in play. Register it as a candidate and let the identifier stratum decide.

---

## The alternative: download it instead

The REST API is rate-limited and adds a network hop per term. The full Metathesaurus is
downloadable from <https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html>
once the licence is approved.

| file | holds | size |
|---|---|---|
| `MRCONSO.RRF` | every atom: CUI, source, term type, string | ~1.5 GB |
| `MRSTY.RRF` | CUI → semantic type | ~120 MB |
| Full release | everything, all sources | ~30 GB |

For this project, `MRCONSO.RRF` + `MRSTY.RRF` is enough to build the **local gazetteer**
§4.14 recommends: a normalised-key trie giving recognition *and* typing in one O(query)
lookup, with no rate limit and no request-path latency. `MetamorphoSys` (ships with the
release) subsets it — restricting to oncology-relevant sources would cut it to a fraction
of that.

**This is the better long-term shape.** It removes the credential from the *request path*
entirely: the key is used once, offline, to build a dictionary that then ships with the
index.
