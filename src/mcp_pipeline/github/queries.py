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


def build_manifest_query_string(signal: str, file_qualifier: str) -> str:
    """e.g. "@modelcontextprotocol/sdk" filename:package.json path:/

    For the REST Code Search API (see github/rest_code_search.py and
    github/search_manifest.py), not the GraphQL repository search above —
    GraphQL repository search has no `in:file`/`filename:`/`extension:`
    qualifier, so it can only ever match signals mentioned in a README or
    description, never a signal declared as a library dependency (e.g. in
    package.json, pyproject.toml, pom.xml). `file_qualifier` is a full
    GitHub code-search qualifier such as "filename:package.json" or
    "extension:csproj", not a bare filename.

    `path:/` restricts matches to the repo root, which is where a project's
    own manifest normally lives — this cuts out the dominant source of false
    positives observed when this query was first run without it: a manifest
    matching the signal deep inside e.g. node_modules/ is almost always the
    *dependency's own* committed copy (its package.json's "name" field
    trivially equals the signal), not the analyzed repo declaring the signal
    as one of its own dependencies (verified against a live token:
    "@modelcontextprotocol/sdk" filename:package.json dropped from
    total_count=103424 to 52352 once path:/ was added). The trade-off is
    that a legitimate manifest nested in a monorepo subdirectory (e.g.
    packages/foo/package.json) is no longer matched.

    Two qualifiers that would otherwise help do NOT work on this endpoint —
    verified against a live token, not just documentation:
    - `fork:false` errors with 422 ERROR_TYPE_QUERY_PARSING_FATAL (unlike
      repository search, which accepts it). Omitted; forks are excluded from
      code search by default without an explicit qualifier, and
      dedupe_rank.filter_and_rank re-checks is_fork after hydration anyway.
    - `-path:DIR` negation and `stars:>=N` are both silently no-ops (the
      former leaves total_count unchanged, the latter returns
      total_count=0) — this endpoint's index doesn't support path exclusion
      or star filtering, so popularity filtering only happens after
      hydration, in dedupe_rank.filter_and_rank.

    Actual lockfiles (package-lock.json, yarn.lock, poetry.lock, ...) are
    excluded by construction — `filename:` is an exact-match qualifier, so a
    query scoped to one manifest filename can never also match a
    differently-named lockfile. `requirements.txt` is deliberately absent
    from manifest_signals' file_qualifiers in config/mcp_signals.yaml for
    the Python signals: pip-freeze-style requirements.txt lists the full
    transitive dependency closure, so it behaves like a lockfile (matches
    any repo that merely depends on something that depends on FastMCP, not
    only repos that use it directly) — pyproject.toml/setup.py are
    hand-authored and stay direct-dependency signals.

    Even with path:/, some signals stay far above GitHub's 1000-result cap
    (e.g. "@modelcontextprotocol/sdk" filename:package.json: 52352;
    "FastMCP" filename:pyproject.toml: 12928) — see the manifest_signals
    comment in config/mcp_signals.yaml for how that's handled.
    """
    return f'"{signal}" {file_qualifier} path:/'


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
