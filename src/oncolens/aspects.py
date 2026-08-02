"""The technical dimensions along which oncology papers get compared.

Split out of ``compare.py`` so the **serving path can import the definitions without
importing the offline retriever**. ``compare.py`` pulls in ``retrieval.pipeline``, which
reaches scipy and expects a fitted in-process index — neither of which exists in a
serverless function, and requiring them is what made /api/compare return
``No module named 'scipy'`` in production while working locally.

The data here is the contract; how a passage is found for a cell is the caller's business
(in-process for the offline harness, SQL for the live store).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Aspect:
    key: str
    label: str
    question: str            # used to steer retrieval toward this dimension
    cues: tuple[str, ...]    # surface markers that a passage reports this aspect
    numeric: bool = False    # does a well-formed answer contain a number?


ASPECTS: tuple[Aspect, ...] = (
    Aspect(
        "cohort", "Cohort / sample",
        "how many patients or samples were studied, and what population",
        ("patients", "participants", "cohort", "enrolled", "n =", "n=", "samples",
         "subjects", "cases", "consecutive", "median age", "eligible"),
        numeric=True,
    ),
    Aspect(
        "assay", "Assay / platform",
        "what assay, sequencing platform, or measurement technology was used",
        ("sequencing", "ngs", "ddpcr", "digital pcr", "qpcr", "rna-seq", "wes", "wgs",
         "immunohistochemistry", "ihc", "flow cytometry", "mass spectrometry", "elisa",
         "panel", "assay", "platform", "illumina", "single-cell", "spatial"),
    ),
    Aspect(
        "endpoint", "Endpoint",
        "what clinical or biological endpoint was measured",
        ("progression-free survival", "overall survival", "pfs", "os", "response rate",
         "orr", "objective response", "duration of response", "endpoint", "primary outcome",
         "disease-free", "event-free", "time to", "clearance"),
    ),
    Aspect(
        "effect", "Effect size / statistics",
        "what effect size, hazard ratio, or statistical result was reported",
        ("hazard ratio", "hr ", "odds ratio", "95% ci", "confidence interval", "p =",
         "p<", "p =", "median pfs", "median os", "ic50", "fold change", "significant",
         "log-rank", "auc", "sensitivity", "specificity"),
        numeric=True,
    ),
    Aspect(
        "model", "Model system",
        "what experimental model was used",
        ("cell line", "xenograft", "pdx", "patient-derived", "organoid", "mouse", "mice",
         "in vitro", "in vivo", "knockout", "crispr", "transgenic", "syngeneic"),
    ),
    Aspect(
        "intervention", "Intervention / agent",
        "what drug, dose, or intervention was administered",
        ("mg", "dose", "administered", "treated with", "monotherapy", "combination",
         "regimen", "cycle", "intravenous", "oral", "twice daily", "once daily"),
    ),
    Aspect(
        "resistance", "Resistance mechanism",
        "what mechanism of resistance or escape was identified",
        ("resistance", "resistant", "escape", "bypass", "acquired", "refractory",
         "relapse", "progression on", "mutation", "amplification", "loss of"),
    ),
    Aspect(
        "limitation", "Limitations",
        "what limitations or caveats did the authors state",
        ("limitation", "caveat", "retrospective", "single-center", "small sample",
         "not powered", "confounding", "selection bias", "should be interpreted"),
    ),
)

ASPECTS_BY_KEY = {a.key: a for a in ASPECTS}

#: Default columns. Four is what fits a readable table; the rest are opt-in.
DEFAULT_ASPECT_KEYS = ("cohort", "assay", "endpoint", "effect")
