from __future__ import annotations

"""Secondary code-search fallback, used by github/search_manifest.py.

GitHub's GraphQL API has no `type: CODE` search — real code search only
exists via the REST endpoint `/search/code`, which is capped at 10 requests/
minute even when authenticated. Per the pipeline design, the primary
dependency-detection signal is the GraphQL `in:readme,description` text
search (see search_text.py), which stays consistent with the thesis's stated
"GraphQL API" methodology and should already catch most real MCP servers
(READMEs conventionally document the install command / package name).

This module is also called by run_step1.py via search_manifest.py, to catch
repos that declare a signal as a library dependency (e.g. in package.json)
but never mention it in their README — see manifest_signals in
config/mcp_signals.yaml for which signal is checked against which file.
"""

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path

import requests

from mcp_pipeline.collection.checkpoint import Checkpoint

logger = logging.getLogger("mcp_pipeline.rest_code_search")

GITHUB_CODE_SEARCH_URL = "https://api.github.com/search/code"
_MIN_SECONDS_BETWEEN_REQUESTS = 6.5  # keeps well under the 10 req/min cap
_MAX_SEARCH_RESULTS = 1000  # GitHub Search API hard cap; page beyond this 422s


def search_code_for_signal(
    token: str,
    signal_label: str,
    query_string: str,
    checkpoint: Checkpoint,
    raw_pages_dir: Path,
    per_page: int = 100,
) -> Iterator[dict]:
    """Runs one REST Code Search query to exhaustion (or GitHub's 1000-result
    cap), yielding raw repository dicts (from the `repository` field of each
    code search hit). Caller is responsible for deduping and mapping into
    RepoCandidate.

    Mirrors github/search_runner.py's run_paginated_repo_search for the
    GraphQL path: every raw page is written to disk under `raw_pages_dir`
    before being yielded, so a crash — or hitting the 10 req/min cap's 403 —
    never loses an already-paid-for API call, and `signal_label`'s progress
    is checkpointed page by page so an interrupted run resumes from the last
    completed page instead of restarting the whole sub-query from page 1.

    If `signal_label` is already marked done AND `query_string` matches what
    produced that saved state, replays the previously saved raw pages from
    disk instead of re-querying (see run_paginated_repo_search's docstring
    for why a "done" signal still needs its repos replayed on every run).
    """
    state_key = f"code_search::{signal_label}"
    saved = checkpoint.get(state_key, {})
    done = saved.get("done", False)
    saved_query_string = saved.get("query_string")

    if done and saved_query_string == query_string:
        logger.info("Code sub-query %r already completed, replaying cached pages", signal_label)
        yield from _replay_cached_pages(signal_label, raw_pages_dir)
        return

    if done and saved_query_string != query_string:
        logger.warning(
            "Code sub-query %r was marked done for a different query string (%r vs current %r) "
            "— treating checkpoint as stale and re-running from scratch",
            signal_label,
            saved_query_string,
            query_string,
        )
        _delete_cached_pages(signal_label, raw_pages_dir)
        page = 1
    else:
        page = saved.get("next_page_num", 1)

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "tcc-mcp-pipeline/0.1",
    }

    while True:
        resp = requests.get(
            GITHUB_CODE_SEARCH_URL,
            headers=headers,
            params={"q": query_string, "per_page": per_page, "page": page},
            timeout=30,
        )
        if resp.status_code == 403:
            retry_after = int(resp.headers.get("retry-after", 60))
            logger.warning("Code search rate limited, sleeping %ss", retry_after)
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        payload = resp.json()

        raw_pages_dir.mkdir(parents=True, exist_ok=True)
        page_file = raw_pages_dir / f"{_safe_filename(signal_label)}__page{page:04d}.json"
        page_file.write_text(
            json.dumps(_trim_payload_for_storage(payload), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        items = payload.get("items", [])
        for item in items:
            repo = item.get("repository")
            if repo:
                yield repo

        reached_end = len(items) < per_page
        hit_cap = page * per_page >= _MAX_SEARCH_RESULTS
        if hit_cap and not reached_end:
            logger.warning(
                "Code search for %r hit GitHub's %s-result cap (total_count=%s) — "
                "results beyond that are not retrievable; signal may be too broad",
                signal_label,
                _MAX_SEARCH_RESULTS,
                payload.get("total_count"),
            )
        is_last_page = reached_end or hit_cap

        checkpoint.set(
            state_key,
            {
                "next_page_num": page + 1,
                "done": is_last_page,
                "query_string": query_string,
            },
        )

        if is_last_page:
            logger.info("Code sub-query %r complete: %s page(s) fetched", signal_label, page)
            break

        page += 1
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS)


def _trim_payload_for_storage(payload: dict) -> dict:
    """Strips each hit down to the fields search_manifest.py and a future
    manual reviewer actually need (full_name to hydrate via GraphQL; id/
    node_id/owner/html_url/description/fork for a quick sanity check without
    re-fetching). The raw GitHub response carries dozens of *_url boilerplate
    fields per repository (and per-file fields we never read at all, e.g.
    the matched file's own path/sha) that only bloat raw/search_pages/ for no
    benefit here.
    """
    return {
        "total_count": payload.get("total_count"),
        "incomplete_results": payload.get("incomplete_results"),
        "items": [
            {"repository": _trim_repo(item["repository"]), "score": item.get("score")}
            for item in payload.get("items", [])
            if item.get("repository")
        ],
    }


def _trim_repo(repo: dict) -> dict:
    owner = repo.get("owner") or {}
    return {
        "id": repo.get("id"),
        "node_id": repo.get("node_id"),
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "owner": {
            "id": owner.get("id"),
            "node_id": owner.get("node_id"),
            "login": owner.get("login"),
        },
        "private": repo.get("private"),
        "html_url": repo.get("html_url"),
        "description": repo.get("description"),
        "fork": repo.get("fork"),
    }


def _replay_cached_pages(signal_label: str, raw_pages_dir: Path) -> Iterator[dict]:
    for page_file in cached_page_files(signal_label, raw_pages_dir):
        payload = json.loads(page_file.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            repo = item.get("repository")
            if repo:
                yield repo


def _delete_cached_pages(signal_label: str, raw_pages_dir: Path) -> None:
    """Removes stale page files left over from a query_string that no
    longer matches, so a future replay never mixes pages from two different
    queries.
    """
    for page_file in cached_page_files(signal_label, raw_pages_dir):
        page_file.unlink()


def cached_page_files(signal_label: str, raw_pages_dir: Path) -> list[Path]:
    prefix = f"{_safe_filename(signal_label)}__page"
    return sorted(raw_pages_dir.glob(f"{prefix}*.json"))


def _safe_filename(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
