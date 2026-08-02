"""Evaluation endpoint — retrieval quality, shown in the UI rather than hidden in a report.

Most RAG demos show results and hide their error rate. This exposes the harness numbers
directly to the user, including the unflattering ones, because a retrieval system that
cannot state how often it is wrong should not be trusted with a literature review.

What it returns, and why each number is there:

* ``floor`` — a raw term-frequency scorer with no IDF and no length normalisation. If the
  real system barely beats it, none of the machinery is earning its place. Random and
  popularity floors are reported too, so a headline score can be read in context.
* ``ceiling`` — a perfect ranking of the judged documents. Says how much headroom remains
  and doubles as a self-check: it must be 1.0, or the metric is broken.
* ``metrics`` — the consensus panel, not one number. A change that moves one metric while
  degrading three is a trade, not an improvement.
* ``per_stratum`` — separated by query type, because an aggregate mean routinely rises
  while an entire query class (exact identifier lookup, say) collapses.
* ``pool`` — ``unjudged@10`` and judgment coverage. When most of what the system returns
  has never been judged, the score is an underestimate of unknown size and the honest
  answer is that the number is not yet interpretable.
* ``caveats`` — plain-language statements of what these numbers do NOT establish.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

#: Precomputed offline by scripts/run.py and shipped with the deploy. Recomputing an
#: evaluation inside a serverless function would exceed the execution limit and would
#: also be wrong: the numbers must come from the same frozen run the team reviewed.
REPORT_PATH = Path(os.environ.get("ONCOLENS_EVAL_REPORT", _ROOT / "public" / "eval_report.json"))


def live_document_count() -> int | None:
    """How many documents the SERVED index actually holds. None if unreachable."""
    dsn = os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        return None
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            return int(cur.fetchone()[0])
    except Exception:  # noqa: BLE001 — a stale-check must never break the endpoint
        return None


def annotate_applicability(report: dict) -> dict:
    """Mark whether this report describes the index that is actually being served.

    **The failure this prevents, which was live.** ``public/eval_report.json`` is generated
    from ``fixtures/synthetic`` — 140 machine-generated documents, 116 queries, strata
    named ``boolean_scope / conceptual / lexical / multi_hop / paraphrase``. The served
    index is thousands of real papers with strata ``synthesis / concept / identifier /
    claim``, and production scores ``ts_rank_cd`` where that harness scored BM25. The
    search page rendered this report's ``ndcg@10`` in a strip directly above every result
    list, so the single most prominent number in the product was a synthetic-fixture score
    presented as this system's measured quality.

    ``fixtures/README.md`` and CLAUDE.md both say a number from the synthetic corpus must
    never be quoted as evidence about real retrieval. Nothing enforced it, so this does:
    the endpoint compares the report's corpus against the live store and says plainly when
    they are not the same thing. The client refuses to show the headline on a mismatch.
    """
    corpus = report.get("corpus") or {}
    reported = corpus.get("documents")
    live = live_document_count()
    report["applies_to_served_index"] = None
    if live is None or reported is None:
        report["applicability_note"] = (
            "Could not confirm this report describes the index being served."
        )
        return report
    # Any material difference means a different corpus, not a slightly stale count.
    same = abs(int(live) - int(reported)) <= max(2, int(reported) * 0.02)
    report["applies_to_served_index"] = bool(same)
    report["live_documents"] = live
    if not same:
        report["applicability_note"] = (
            f"This report was generated on a {reported}-document corpus; the index being "
            f"served holds {live}. The numbers below describe a DIFFERENT corpus and must "
            f"not be read as this system's retrieval quality."
        )
    return report


def load_report() -> dict:
    if REPORT_PATH.exists():
        return annotate_applicability(json.loads(REPORT_PATH.read_text(encoding="utf-8")))
    return {
        "status": "unavailable",
        "message": (
            "No evaluation report has been generated. Run "
            "`python scripts/build_eval_report.py` and redeploy. Until then, treat every "
            "result from this system as unmeasured."
        ),
    }


class handler(BaseHTTPRequestHandler):  # noqa: N801 — Vercel requires this name
    def do_GET(self):  # noqa: N802
        params = parse_qs(urlparse(self.path).query)
        report = load_report()
        if params.get("section"):
            key = params["section"][0]
            report = {key: report.get(key)} if key in report else {"error": f"no section {key!r}"}
        raw = json.dumps(report, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # Short cache, not the old hour. This response now depends on a LIVE document
        # count, so a long s-maxage would keep serving "this report describes a different
        # corpus" for an hour after a re-measure fixed it, or worse, keep asserting the
        # report applies after an ingest made it stale.
        self.send_header("Cache-Control", "public, max-age=60, s-maxage=120")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        return
