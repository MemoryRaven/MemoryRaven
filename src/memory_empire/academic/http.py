from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


def default_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=60.0, write=20.0, pool=10.0)


def default_limits() -> httpx.Limits:
    return httpx.Limits(max_connections=100, max_keepalive_connections=20)


class HttpClientFactory:
    """Creates shared httpx clients with sane defaults.

    Keep one client per service process; do not create per-request.
    """

    @staticmethod
    def client(base_url: str | None = None, headers: dict | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=base_url or "",
            headers=headers,
            timeout=default_timeout(),
            limits=default_limits(),
            follow_redirects=True,
        )


TransientHttpError = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


def transient_retry():
    return retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=0.5, max=10.0),
        retry=retry_if_exception_type(TransientHttpError),
    )
