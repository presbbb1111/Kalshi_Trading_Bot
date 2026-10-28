"""Kalshi Trade API client.

Market-data reads need no authentication, so this client is usable immediately.
Signed requests (orders, portfolio) are added in a later step — see
`docs/kalshi/_llms-full.txt` for the RSA signing scheme.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from src.kalshi_auth import KalshiCreds, auth_headers, load_creds
from src.utils import get_logger, retry_with_backoff

logger = get_logger(__name__)

# Production vs demo. Note the TLD differs: production is .com, demo is .co.
PROD_REST = "https://external-api.kalshi.com/trade-api/v2"
PROD_WS = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
DEMO_REST = "https://external-api.demo.kalshi.co/trade-api/v2"
DEMO_WS = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"


class KalshiClient:
    """Thin REST wrapper. Read-only for now."""

    API_ROOT = "/trade-api/v2"

    def __init__(self, *, demo: bool | None = None, timeout: float = 15.0) -> None:
        if demo is None:
            demo = os.getenv("KALSHI_ENV", "demo").lower() == "demo"
        self.demo = demo
        self.rest_url = DEMO_REST if demo else PROD_REST
        self.ws_url = DEMO_WS if demo else PROD_WS
        self.timeout = timeout
        self._session = requests.Session()
        logger.info(f"KalshiClient initialized ({'demo' if demo else 'production'}) {self.rest_url}")

    @property
    def creds(self) -> KalshiCreds:
        """Credentials matching this client's environment.

        Resolved lazily so unauthenticated market-data reads work with no keys
        configured at all.
        """
        return load_creds(prod=not self.demo)

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        authed: bool = False,
    ) -> dict[str, Any]:
        """GET `path` (relative to the API root) and return the decoded JSON body.

        Most market-data endpoints need no auth; pass `authed=True` for the ones
        that do (portfolio, CF Benchmarks passthrough).
        """
        clean = path.lstrip("/")
        url = f"{self.rest_url}/{clean}"
        # The signature covers the full path from the API root, without the query.
        headers = auth_headers("GET", f"{self.API_ROOT}/{clean}", self.creds) if authed else None

        def _do() -> dict[str, Any]:
            resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
            # Surface 429s to retry_with_backoff rather than returning a bad body.
            resp.raise_for_status()
            return resp.json()

        return retry_with_backoff(_do, logger=logger)

    def post(
        self,
        path: str,
        json: dict[str, Any] | None = None,
        *,
        authed: bool = True,
    ) -> dict[str, Any]:
        """POST `path`. Not retried — see `_write`."""
        return self._write("POST", path, json=json, authed=authed)

    def delete(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        authed: bool = True,
    ) -> dict[str, Any]:
        """DELETE `path`. Not retried — see `_write`."""
        return self._write("DELETE", path, params=params, authed=authed)

    def _write(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        authed: bool = True,
    ) -> dict[str, Any]:
        """Send a state-changing request.

        Deliberately **not** wrapped in `retry_with_backoff`. A GET is safe to
        repeat; a POST that creates an order is not. A timeout doesn't tell you
        whether the exchange got the request, so retrying risks placing the same
        order twice. Callers should reconcile against the exchange instead.
        """
        clean = path.lstrip("/")
        url = f"{self.rest_url}/{clean}"
        headers = (
            auth_headers(method, f"{self.API_ROOT}/{clean}", self.creds) if authed else None
        )
        resp = self._session.request(
            method, url, json=json, params=params, headers=headers, timeout=self.timeout
        )
        if resp.status_code >= 400:
            # Attach the response body to the error. `raise_for_status()` alone
            # reports only the status code, and a bare "404 Not Found" on a POST
            # is indistinguishable from a dozen different causes — which is how
            # a three-day order outage (2026-08-28 to 08-31) went undiagnosed.
            raise requests.HTTPError(
                f"{resp.status_code} {resp.reason} for {method} {clean}: "
                f"{resp.text[:400]}",
                response=resp,
            )
        return resp.json() if resp.content else {}
