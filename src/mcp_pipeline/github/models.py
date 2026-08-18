from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RepoCandidate:
    """One repository as returned by a GitHub GraphQL repository search."""

    id: str  # stable GraphQL node id — dedup key (nameWithOwner can change)
    name_with_owner: str
    url: str
    description: str | None
    stargazer_count: int
    fork_count: int
    is_fork: bool
    is_archived: bool
    pushed_at: str
    created_at: str
    primary_language: str | None
    languages: list[str]
    topics: list[str]
    default_branch: str | None
    default_branch_oid: str | None
    license_spdx_id: str | None
    matched_signals: list[str] = field(default_factory=list)  # which query/queries surfaced it

    @classmethod
    def from_graphql_node(cls, node: dict, matched_signal: str) -> RepoCandidate:
        default_branch_ref = node.get("defaultBranchRef") or {}
        target = default_branch_ref.get("target") or {}
        license_info = node.get("licenseInfo") or {}
        primary_language = node.get("primaryLanguage") or {}
        return cls(
            id=node["id"],
            name_with_owner=node["nameWithOwner"],
            url=node["url"],
            description=node.get("description"),
            stargazer_count=node["stargazerCount"],
            fork_count=node["forkCount"],
            is_fork=node["isFork"],
            is_archived=node["isArchived"],
            pushed_at=node["pushedAt"],
            created_at=node["createdAt"],
            primary_language=primary_language.get("name"),
            languages=[n["name"] for n in (node.get("languages") or {}).get("nodes", [])],
            topics=[
                n["topic"]["name"] for n in (node.get("repositoryTopics") or {}).get("nodes", [])
            ],
            default_branch=default_branch_ref.get("name"),
            default_branch_oid=target.get("oid"),
            license_spdx_id=license_info.get("spdxId"),
            matched_signals=[matched_signal],
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name_with_owner": self.name_with_owner,
            "url": self.url,
            "description": self.description,
            "stargazer_count": self.stargazer_count,
            "fork_count": self.fork_count,
            "is_fork": self.is_fork,
            "is_archived": self.is_archived,
            "pushed_at": self.pushed_at,
            "created_at": self.created_at,
            "primary_language": self.primary_language,
            "languages": self.languages,
            "topics": self.topics,
            "default_branch": self.default_branch,
            "default_branch_oid": self.default_branch_oid,
            "license_spdx_id": self.license_spdx_id,
            "matched_signals": self.matched_signals,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RepoCandidate:
        return cls(**d)
