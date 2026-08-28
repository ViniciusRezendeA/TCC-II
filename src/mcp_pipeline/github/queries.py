from __future__ import annotations

SMOKE_TEST_QUERY = """
query SmokeTest {
  viewer { login }
  rateLimit { limit cost remaining resetAt }
}
"""

# NOTE: `type: CODE` does not exist in the GitHub GraphQL API — only
# `type: REPOSITORY` (used here), ISSUE, USER, DISCUSSION are supported.
# Code search is REST-only (`/search/code`, 10 req/min) and is deliberately
# NOT used as the primary mechanism — see github/rest_code_search.py.
SEARCH_REPOS_QUERY = """
query SearchRepos($queryString: String!, $after: String) {
  rateLimit { cost remaining resetAt }
  search(query: $queryString, type: REPOSITORY, first: 50, after: $after) {
    repositoryCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on Repository {
        id
        nameWithOwner
        url
        description
        stargazerCount
        forkCount
        isFork
        isArchived
        pushedAt
        createdAt
        primaryLanguage { name }
        languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
          nodes { name }
        }
        repositoryTopics(first: 20) {
          nodes { topic { name } }
        }
        defaultBranchRef {
          name
          target {
            ... on Commit { oid }
          }
        }
        licenseInfo { spdxId }
      }
    }
  }
}
"""


def build_topic_query_string(topic: str, min_stars: int) -> str:
    """e.g. topic:mcp-server fork:false stars:>=100"""
    return f"topic:{topic} fork:false stars:>={min_stars}"


def build_text_query_string(signal: str, min_stars: int) -> str:
    """e.g. "@modelcontextprotocol/sdk" in:readme,description fork:false stars:>=100

    Each signal is run as an independent query (not combined with `OR (...)`)
    because that combined-query syntax was not validated against a live token
    during design — see plan risk #2. Revisit as an optimization only after
    smoke-testing it for real.
    """
    return f'"{signal}" in:readme,description fork:false stars:>={min_stars}'


def build_manifest_query_string(
    signal: str,
    file_qualifier: str,
    *,
    language: str,
) -> str:
    """Build a REST Code Search query for a manifest dependency signal.

    Example:

        "@modelcontextprotocol/sdk" filename:package.json language:TypeScript path:/

    The query is intended for the REST Code Search API (see
    github/rest_code_search.py and github/search_manifest.py), not the
    GraphQL repository search above.

    `file_qualifier` is a full GitHub code-search qualifier such as
    "filename:package.json" or "extension:csproj", not a bare filename.

    `language` is added directly to the Code Search query so that each
    manifest signal is partitioned by its configured programming language.
    This is important because GitHub Code Search has a 1,000-result
    retrieval cap for a single query. Filtering by language only after
    retrieving the results would not recover repositories that were already
    excluded by that cap.

    `path:/` restricts matches to the repo root, which is where a project's
    own manifest normally lives. This cuts out the dominant source of false
    positives observed when this query was first run without it: a manifest
    matching the signal deep inside e.g. node_modules/ is almost always the
    dependency's own committed copy (its package.json's "name" field
    trivially equals the signal), not the analyzed repo declaring the signal
    as one of its own dependencies.

    The trade-off is that a legitimate manifest nested in a monorepo
    subdirectory (e.g. packages/foo/package.json) is no longer matched.

    Two qualifiers that would otherwise help do NOT work on this endpoint:

    - `fork:false` errors with 422 ERROR_TYPE_QUERY_PARSING_FATAL (unlike
      repository search, which accepts it). It is omitted because forks are
      excluded from code search by default without an explicit qualifier,
      and dedupe_rank.filter_and_rank re-checks is_fork after hydration.

    - `-path:DIR` negation and `stars:>=N` are not usable on this endpoint.
      The former is a no-op and the latter does not behave as a supported
      star filter. Popularity filtering therefore remains a client-side
      concern and is performed after hydration by
      dedupe_rank.filter_and_rank.

    Actual lockfiles (package-lock.json, yarn.lock, poetry.lock, ...) are
    excluded by construction when `filename:` is used because it is an
    exact-match qualifier.

    `requirements.txt` may be configured as a manifest qualifier for some
    signals. Keep in mind that requirements.txt can contain transitive
    dependencies and therefore may produce more false positives than
    hand-authored manifests such as pyproject.toml or setup.py.

    Even with `path:/` and language partitioning, some individual
    signal/language combinations may still exceed GitHub's 1,000-result
    retrieval cap. Those cases require additional query partitioning.
    """

    return (
        f'"{signal}" '
        f"{file_qualifier} "
    )


# Used to hydrate a repo found via REST code search (github/search_manifest.py)
# with the same fields SEARCH_REPOS_QUERY returns, since the REST code-search
# response's nested `repository` object omits stargazerCount/language/fork —
# the fields dedupe_rank.py's filters need.
REPO_BY_NAME_QUERY = """
query RepoByName($owner: String!, $name: String!) {
  rateLimit { cost remaining resetAt }
  repository(owner: $owner, name: $name) {
    id
    nameWithOwner
    url
    description
    stargazerCount
    forkCount
    isFork
    isArchived
    pushedAt
    createdAt
    primaryLanguage { name }
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      nodes { name }
    }
    repositoryTopics(first: 20) {
      nodes { topic { name } }
    }
    defaultBranchRef {
      name
      target {
        ... on Commit { oid }
      }
    }
    licenseInfo { spdxId }
  }
}
"""
