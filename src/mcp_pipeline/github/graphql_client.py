from __future__ import annotations

import logging
import time
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mcp_pipeline.config import GITHUB_GRAPHQL_URL

logger = logging.getLogger("mcp_pipeline.graphql")


class GraphQLError(RuntimeError):
    """Raised when the GitHub GraphQL API returns an `errors` payload."""

    def __init__(self, errors: list[dict], query: str, variables: dict):
        self.errors = errors
        self.query = query
        self.variables = variables
        super().__init__(f"GitHub GraphQL API returned errors: {errors}")


class RetryableHTTPError(RuntimeError):
    """Transient error (5xx, secondary rate limit, network) — safe to retry."""


class GraphQLClient:
    """Thin wrapper around the GitHub GraphQL API (POST /graphql).

    Tracks the `rateLimit` block returned alongside every query (queries in
    this project always request `rateLimit { cost remaining resetAt }` at the
    top level) and sleeps proactively when the budget runs low, instead of
    waiting to be told off with a 403.
    """

    def __init__(self, token: str, min_remaining_before_pause: int = 50):
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "tcc-mcp-pipeline/0.1",
            }
        )
        self.min_remaining_before_pause = min_remaining_before_pause
        self.last_rate_limit: dict[str, Any] | None = None

    @retry(
        retry=retry_if_exception_type(RetryableHTTPError),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def execute(self, query: str, variables: dict | None = None) -> dict:
        variables = variables or {}
        response = self._session.post(
            GITHUB_GRAPHQL_URL, json={"query": query, "variables": variables}, timeout=30
        )

        if response.status_code in (502, 503, 504):
            raise RetryableHTTPError(f"HTTP {response.status_code} from GitHub GraphQL API")

        if response.status_code == 403 and "retry-after" in response.headers:
            retry_after = int(response.headers["retry-after"])
            logger.warning("Secondary rate limit hit, sleeping %ss", retry_after)
            time.sleep(retry_after)
            raise RetryableHTTPError("Secondary rate limit (retry-after)")

        response.raise_for_status()
        payload = response.json()

        if "errors" in payload:
            raise GraphQLError(payload["errors"], query, variables)

        data = payload["data"]
        rate_limit = data.get("rateLimit")
        if rate_limit:
            self.last_rate_limit = rate_limit
            self._pause_if_budget_low(rate_limit)

        return data

    def _pause_if_budget_low(self, rate_limit: dict) -> None:
        remaining = rate_limit["remaining"]
        if remaining > self.min_remaining_before_pause:
            return
        reset_at = rate_limit["resetAt"]
        sleep_seconds = _seconds_until(reset_at) + 5
        logger.warning(
            "GraphQL rate limit low (remaining=%s), sleeping %ss until reset (%s)",
            remaining,
            sleep_seconds,
            reset_at,
        )
        time.sleep(max(sleep_seconds, 0))


def _seconds_until(iso_timestamp: str) -> float:
    import datetime

    reset_dt = datetime.datetime.fromisoformat(iso_timestamp)
    now = datetime.datetime.now(datetime.UTC)
    return (reset_dt - now).total_seconds()
