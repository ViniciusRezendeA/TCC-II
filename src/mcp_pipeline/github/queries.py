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
