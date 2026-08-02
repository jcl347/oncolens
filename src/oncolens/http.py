"""One retrying HTTP session for every outbound call.

**Why this module exists.** A 1,700-article ingest makes thousands of requests over ~45
minutes to three different services. Two separate runs died on transient transport errors —
``SSLEOFError: EOF occurred in violation of protocol`` from S3, then
``ChunkedEncodingError: Response ended prematurely`` from NCBI E-utilities — each time
after tens of minutes, having written nothing, because the store write happens at the end.

The second failure is the instructive one: retries had already been added, but only to the
module where the *first* failure happened. That fixed a location, not a class. At this
request volume a transport blip is the expected case, and **every** outbound session needs
the same treatment, so there is now one place that defines it.

Retrying is not the whole answer — a caller must still tolerate an item that fails all
attempts rather than aborting the batch — but it is the part that belongs in one place.
"""

from __future__ import annotations

import threading

#: Retry budget. 5 attempts with a 1.0 backoff factor spans roughly 1+2+4+8+16 = 31s of
#: waiting, which comfortably covers the transient TLS and chunked-transfer failures
#: observed, without turning a genuinely dead endpoint into a five-minute hang.
TOTAL_RETRIES = 5
BACKOFF_FACTOR = 1.0
RETRY_STATUS = (429, 500, 502, 503, 504)

_lock = threading.Lock()
_sessions: dict[str, object] = {}


def session(name: str = "default", *, user_agent: str = "oncolens/0.1 (research retrieval benchmark)"):
    """A retrying ``requests.Session``, created once per ``name`` and reused.

    Reuse matters as much as retrying: a fresh TLS handshake per article is a large part
    of the wall clock on a multi-thousand-request job.
    """
    with _lock:
        s = _sessions.get(name)
        if s is not None:
            return s

        import requests
        from requests.adapters import HTTPAdapter

        try:
            from urllib3.util.retry import Retry

            retry = Retry(
                total=TOTAL_RETRIES,
                connect=TOTAL_RETRIES,
                read=TOTAL_RETRIES,
                status=TOTAL_RETRIES,
                backoff_factor=BACKOFF_FACTOR,
                status_forcelist=RETRY_STATUS,
                allowed_methods=frozenset({"GET", "HEAD", "POST"}),
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
        except Exception:  # noqa: BLE001 — urllib3 API drift must not break ingestion
            adapter = HTTPAdapter(pool_maxsize=16)

        s = requests.Session()
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": user_agent})
        _sessions[name] = s
        return s
