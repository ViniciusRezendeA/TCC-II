from __future__ import annotations

"""Secondary, OPT-IN code-search fallback.

GitHub's GraphQL API has no `type: CODE` search — real code search only
exists via the REST endpoint `/search/code`, which is capped at 10 requests/
minute even when authenticated. Per the pipeline design, the primary
dependency-detection signal is the GraphQL `in:readme,description` text
search (see search_text.py), which stays consistent with the thesis's stated
"GraphQL API" methodology and should already catch most real MCP servers
(READMEs conventionally document the install command / package name).

This module is NOT called by run_step1.py by default. Only wire it in if a
manual review of the Phase-1 candidate pool shows a real recall gap that the
README/description text search is missing — e.g. repos that use the SDK but
never mention it in their README.
"""

import logging
import time
from collections.abc import Iterator

import requests

logger = logging.getLogger("mcp_pipeline.rest_code_search")

GITHUB_CODE_SEARCH_URL = "https://api.github.com/search/code"
_MIN_SECONDS_BETWEEN_REQUESTS = 6.5  # keeps well under the 10 req/min cap


def search_code_for_signal(token: str, signal: str, per_page: int = 100) -> Iterator[dict]:
    """Yields raw repository dicts (from the `repository` field of each code
    search hit) for files that reference `signal` in the default branch.
    Caller is responsible for deduping and mapping into RepoCandidate.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "tcc-mcp-pipeline/0.1",
    }
    page = 1
    while True:
        resp = requests.get(
            GITHUB_CODE_SEARCH_URL,
            headers=headers,
            params={"q": signal, "per_page": per_page, "page": page},
            timeout=30,
        )
        if resp.status_code == 403:
            retry_after = int(resp.headers.get("retry-after", 60))
            logger.warning("Code search rate limited, sleeping %ss", retry_after)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        payload = resp.json()

        for item in payload.get("items", []):
            repo = item.get("repository")
            if repo:
                yield repo

        if len(payload.get("items", [])) < per_page:
            break
        page += 1
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS)
