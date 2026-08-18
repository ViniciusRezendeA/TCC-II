from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

from mcp_pipeline.extraction.language_registry import LanguageSpec
from mcp_pipeline.extraction.parser_utils import (
    iter_source_files,
    parse_file,
    relative_path,
)


@dataclass
class FunctionDef:
    qualified_name: str  # "ClassName.method_name" or bare "function_name"
    bare_name: str
    file: str  # relative to repo_src_root
    start_line: int
    end_line: int
    body_node: Node  # in-memory only — never serialized, only used during this repo's processing
    class_name: str | None = None


@dataclass
class DefinitionIndex:
    by_qualified_name: dict[str, FunctionDef]
    by_bare_name: dict[str, list[FunctionDef]]

    @classmethod
    def build(cls, defs: list[FunctionDef]) -> DefinitionIndex:
        by_qualified_name: dict[str, FunctionDef] = {}
        by_bare_name: dict[str, list[FunctionDef]] = {}
        for d in defs:
            by_qualified_name[d.qualified_name] = d
            by_bare_name.setdefault(d.bare_name, []).append(d)
        return cls(by_qualified_name=by_qualified_name, by_bare_name=by_bare_name)


# A language-specific extractor: given a parsed file's root node, its source
# bytes, and its path relative to the repo's src root, returns every
# function/method definition found in that file. Python's implementation is
# in patterns/python_patterns.py; each new language plugs in its own here
# without touching the aggregation logic below (architecture principle #6).
DefinitionExtractor = Callable[[Node, bytes, str], list[FunctionDef]]


def build_definition_index(
    repo_src_root: Path, spec: LanguageSpec, extract_definitions: DefinitionExtractor
) -> DefinitionIndex:
    all_defs: list[FunctionDef] = []
    for file_path in iter_source_files(repo_src_root, spec):
        tree, source_bytes = parse_file(file_path, spec)
        rel_path = relative_path(file_path, repo_src_root)
        all_defs.extend(extract_definitions(tree.root_node, source_bytes, rel_path))
    return DefinitionIndex.build(all_defs)
