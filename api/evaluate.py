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


def load_report() -> dict:
    if REPORT_PATH.exists():
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
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
        self.send_header("Cache-Control", "public, max-age=300, s-maxage=3600")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        return
