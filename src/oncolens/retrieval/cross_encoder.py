"""Local cross-encoder reranking, on the GPU, with no API cost.

**Why a cross-encoder is the largest untested lever here.** Every arm in the first stage is
a *bi-encoder*: the query and the passage are embedded independently and compared by a dot
product. That architecture structurally cannot represent an interaction between them. It
can tell that a passage is *about* osimertinib resistance; it cannot tell whether the
passage **reports** a resistance mechanism or merely mentions that one exists. A
cross-encoder concatenates query and passage into one sequence and attends across both, so
that distinction is available to it. It is also far too expensive to run over 180,850
passages, which is exactly why it belongs in a second stage over the fused top-k.

**Why this replaces the LLM reranker rather than joining it.** ``llm_rerank`` calls
gpt-4o-mini at roughly $0.0017 per query, which put a hard economic cap on the rerank depth
(24 passages) and made a full evaluation sweep cost real money. A local model on an idle
RTX 4070 costs nothing per query, so depth becomes a free parameter and the whole eval set
can be run repeatedly. That matters more than the last point of accuracy: a reranker too
expensive to measure is a reranker that never gets measured.

**Two models, because one would confound domain with architecture.** This project has
already been burned by attributing a gain to the wrong cause (§4.13, §4.14): MedCPT looked
like a domain win until a capacity control showed otherwise, and a three-arm fusion looked
like a model win until a weight control showed half of it was a config line. So:

* ``ncbi/MedCPT-Cross-Encoder`` — trained on PubMed click logs, biomedical.
* ``cross-encoder/ms-marco-MiniLM-L-6-v2`` — trained on MS MARCO web search, general.

If both help equally, the gain is *cross-attention* and any reranker will do. If only the
biomedical one helps, the gain is domain training. Those imply different next steps and the
difference is not guessable from either number alone.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

#: Domain: NCBI's reranker, trained on the same PubMed click logs as MedCPT's bi-encoder.
MEDCPT_CROSS = "ncbi/MedCPT-Cross-Encoder"
#: Control: a general web-search reranker of similar size. Cheap and strong on MS MARCO.
MINILM_CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Query + passage share one sequence, so this budget covers both. 512 is what both models
#: were trained at; truncating the passage is preferable to truncating the query, so the
#: tokenizer is given the pair and told to trim the longer member.
MAX_LENGTH = 512


class CrossEncoderReranker:
    """Score (query, passage) pairs with cross-attention.

    Deliberately mirrors ``MedCPTBackend``'s operational choices, which were measured on
    this hardware: a modest GPU batch because the card also drives the display, an explicit
    synchronise-and-yield so the compositor can draw, and float32 accumulation on the host.
    """

    def __init__(self, model_name: str = MEDCPT_CROSS, *, gpu_batch: int = 32,
                 yield_ms: float = 2.0, max_length: int = MAX_LENGTH) -> None:
        self.model_name = model_name
        self.gpu_batch = gpu_batch
        self.yield_ms = yield_ms
        self.max_length = max_length
        self._tok = None
        self._model = None

    @property
    def name(self) -> str:
        if self.model_name == MEDCPT_CROSS:
            return "medcpt-cross"
        if self.model_name == MINILM_CROSS:
            return "minilm-cross"
        # A local checkpoint. Named distinctly so a run cannot report a fine-tuned
        # score under a base-model label, or the reverse.
        return "finetuned-cross"

    def _device(self) -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        dev = self._device()
        if dev == "cuda":
            # fp16 halves the weights and roughly doubles throughput. These are relevance
            # SCORES used only to sort, so the reduced mantissa cannot change an ordering
            # that fp32 would have gotten right by more than a hair.
            model = model.half()
        model.to(dev)
        if dev == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
        self._model = model

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Relevance score per passage, higher is better. Order matches the input."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        import time

        import torch

        self._load()
        dev = self._device()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(passages), self.gpu_batch):
                batch = [p if p.strip() else " " for p in passages[i : i + self.gpu_batch]]
                enc = self._tok(
                    [query] * len(batch), batch,
                    truncation="only_second",   # keep the whole query, trim the passage
                    padding=True, max_length=self.max_length, return_tensors="pt",
                )
                enc = {k: v.to(dev) for k, v in enc.items()}
                logits = self._model(**enc).logits
                # Both models are single-logit relevance regressors. Guard anyway: a
                # 2-logit checkpoint would otherwise be silently misread as a score.
                s = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
                out.append(s.float().cpu().numpy())
                if dev == "cuda" and self.yield_ms:
                    torch.cuda.synchronize()
                    time.sleep(self.yield_ms / 1000.0)
        return np.concatenate(out).astype(np.float32)

    def rerank(self, query: str, passages: Sequence[str]) -> list[tuple[int, float]]:
        """``(original_index, score)`` sorted best first."""
        s = self.score(query, passages)
        order = np.argsort(-s)
        return [(int(i), float(s[i])) for i in order]

    def unload(self) -> None:
        """Free the GPU. Measured free RAM on this machine has run as low as 0.6 GB."""
        import torch

        self._model = None
        self._tok = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_CACHE: dict[str, CrossEncoderReranker] = {}


def finetuned_path() -> Path:
    """Where `scripts/finetune_reranker.py` writes its checkpoint.

    Under ``local_data_dir()`` for the §2 reason: the repo lives in OneDrive and a
    ~400 MB checkpoint churning there causes sync storms and file locks.
    """
    from oncolens.env import local_data_dir

    return local_data_dir() / "finetune_reranker"


def get_reranker(name: str = "medcpt-cross") -> CrossEncoderReranker:
    """Resolve by short name, loading each model at most once per process."""
    key = (name or "medcpt-cross").lower()
    if key not in _CACHE:
        if key == "finetuned-cross":
            p = finetuned_path()
            if not (p / "config.json").exists():
                raise FileNotFoundError(
                    f"no fine-tuned reranker at {p}; run scripts/finetune_reranker.py "
                    f"first. Refusing to silently fall back to the base model, which "
                    f"would report the BASE model's score under the fine-tuned name.")
            _CACHE[key] = CrossEncoderReranker(str(p))
            return _CACHE[key]
        model = {"medcpt-cross": MEDCPT_CROSS, "minilm-cross": MINILM_CROSS}.get(key)
        if model is None:
            raise ValueError(f"unknown cross-encoder {name!r}")
        _CACHE[key] = CrossEncoderReranker(model)
    return _CACHE[key]
