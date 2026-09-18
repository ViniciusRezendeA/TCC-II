from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceLocation:
    file: str  # path relative to the repo's cloned src/ root
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive

    def to_dict(self) -> dict:
        return {"file": self.file, "start_line": self.start_line, "end_line": self.end_line}

    @classmethod
    def from_dict(cls, d: dict) -> SourceLocation:
        return cls(file=d["file"], start_line=d["start_line"], end_line=d["end_line"])


@dataclass
class ToolRecord:
    name: str
    description: str
    description_is_literal: bool
    sdk_pattern: str  # e.g. "python.fastmcp_decorator" — see patterns/*.py
    source_location: SourceLocation
    qualified_name: str  # links this tool to its Level-1 node in the definition index
    loc: int = 0  # total lines of code summed over the tool's own implementing
    # function AND every function it transitively calls within the repo (see
    # collect_reachable_definitions() in call_graph_builder.py) -- "the
    # complete tool", not just its outer handler. Deliberately NOT bounded by
    # call_graph_depth's 3-level cap below: that cap only limits the
    # *serialized* call graph tree, not what counts as part of the tool's own
    # implementation. 0 is a "not computed yet" sentinel; real values are
    # always >= 1.
    call_graph_depth: int = 0  # max `level` reached in this tool's call graph tree.
    # Bounded to {1, 2, 3} by MAX_LEVEL in call_graph_builder.py. 0 is a "not
    # computed yet" sentinel; real values are always >= 1.
    cyclomatic_complexity: int = 0  # McCabe cyclomatic complexity (classic
    # definition: decision points + 1, NOT counting short-circuit boolean
    # operators -- see extraction/complexity.py), summed over the same
    # reachable-function set as `loc` above, for the same "complete tool"
    # reason. 0 is a "not computed yet" sentinel; real values are always >= 1.

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "description_is_literal": self.description_is_literal,
            "sdk_pattern": self.sdk_pattern,
            "source_location": self.source_location.to_dict(),
            "qualified_name": self.qualified_name,
            "loc": self.loc,
            "call_graph_depth": self.call_graph_depth,
            "cyclomatic_complexity": self.cyclomatic_complexity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ToolRecord:
        return cls(
            name=d["name"],
            description=d["description"],
            description_is_literal=d["description_is_literal"],
            sdk_pattern=d["sdk_pattern"],
            source_location=SourceLocation.from_dict(d["source_location"]),
            qualified_name=d["qualified_name"],
            loc=d["loc"],
            call_graph_depth=d["call_graph_depth"],
            cyclomatic_complexity=d["cyclomatic_complexity"],
        )


@dataclass
class CallGraphNode:
    level: int
    resolved: bool
    external: bool
    ambiguous: bool
    qualified_name: str | None = None  # None when unresolved (external/dynamic call)
    raw_call_text: str | None = None  # always set — the literal call expression source text
    source_location: SourceLocation | None = None  # None when unresolved
    calls: list[CallGraphNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "resolved": self.resolved,
            "external": self.external,
            "ambiguous": self.ambiguous,
            "qualified_name": self.qualified_name,
            "raw_call_text": self.raw_call_text,
            "source_location": self.source_location.to_dict() if self.source_location else None,
            "calls": [c.to_dict() for c in self.calls],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CallGraphNode:
        return cls(
            level=d["level"],
            resolved=d["resolved"],
            external=d["external"],
            ambiguous=d["ambiguous"],
            qualified_name=d.get("qualified_name"),
            raw_call_text=d.get("raw_call_text"),
            source_location=SourceLocation.from_dict(d["source_location"]) if d.get("source_location") else None,
            calls=[cls.from_dict(c) for c in d.get("calls", [])],
        )
