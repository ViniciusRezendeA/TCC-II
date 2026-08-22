from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import nearest_by_directory
from mcp_pipeline.extraction.import_index import ImportIndex
from mcp_pipeline.extraction.language_registry import LanguageSpec
from mcp_pipeline.extraction.parser_utils import (
    iter_source_files,
    parse_file,
    relative_path,
)

# Validated against the deepest real case found (chrisryugj/korean-law-mcp:
# exposedTools -> allTools.filter(...) -> allTools -> literal array, plus a
# .map() unwrap at the use site = 4 structural steps) — not an arbitrary
# round number. Genuinely dynamic cases (a runtime dict/Map lookup, a live
# network call) don't fail because they're a few hops too deep; they bottom
# out at an unresolvable call/lookup regardless of budget, so raising this
# further would not risk fabricating data for those, only wasted resolution
# attempts.
MAX_VALUE_RESOLUTION_HOPS = 4


@dataclass
class ValueDef:
    bare_name: str
    file: str  # relative to repo_src_root
    value_node: Node  # in-memory only — never serialized, mirrors FunctionDef.body_node


@dataclass
class ValueIndex:
    by_file_and_name: dict[str, dict[str, ValueDef]]
    by_bare_name: dict[str, list[ValueDef]]

    @classmethod
    def build(cls, defs: list[ValueDef]) -> ValueIndex:
        by_file_and_name: dict[str, dict[str, ValueDef]] = {}
        by_bare_name: dict[str, list[ValueDef]] = {}
        for d in defs:
            by_file_and_name.setdefault(d.file, {})[d.bare_name] = d
            by_bare_name.setdefault(d.bare_name, []).append(d)
        return cls(by_file_and_name=by_file_and_name, by_bare_name=by_bare_name)


# A language-specific extractor: given a parsed file's root node, its source
# bytes, and its path relative to the repo's src root, returns every
# module/program-top-level value binding found in that file (e.g. `search_tool
# = types.Tool(...)` in Python, `const allTools = [...]` in TS/JS). Mirrors
# DefinitionExtractor's shape exactly (definition_index.py) — same
# "per-language plug-in, shared aggregation" split.
ValueExtractor = Callable[[Node, bytes, str], list[ValueDef]]


def build_value_index(repo_src_root: Path, spec: LanguageSpec, extract_values: ValueExtractor) -> ValueIndex:
    all_values: list[ValueDef] = []
    for file_path in iter_source_files(repo_src_root, spec):
        tree, source_bytes = parse_file(file_path, spec)
        rel_path = relative_path(file_path, repo_src_root)
        all_values.extend(extract_values(tree.root_node, source_bytes, rel_path))
    return ValueIndex.build(all_values)


def resolve_value(
    name: str,
    current_file: str,
    values: ValueIndex,
    imports_by_file: dict[str, ImportIndex],
) -> tuple[ValueDef | None, bool]:
    """Mirrors call_graph_builder.resolve_call()'s cascade (cheapest/most
    local match first, explicit bounded fallback, give up cleanly rather
    than guess) applied to bare value references instead of function calls.
    There is no self/this step (values aren't accessed via a receiver) and
    no receiver-qualified step (a value reference is always a bare name at
    the point this is called — TS/JS's `SomeModule.TOOLS` qualified form is
    not a confirmed real case and isn't handled). Returns (resolved_or_None,
    ambiguous).
    """
    same_file = values.by_file_and_name.get(current_file, {}).get(name)
    if same_file is not None:
        return same_file, False

    imported = imports_by_file.get(current_file, {}).get(name)
    if imported is not None:
        candidate_stems = {PurePosixPath(imported.module.lstrip(".")).name, name} - {""}
        module_candidates = [
            v for v in values.by_bare_name.get(name, []) if PurePosixPath(v.file).stem in candidate_stems
        ]
        if len(module_candidates) == 1:
            return module_candidates[0], False
        if len(module_candidates) > 1:
            return nearest_by_directory(module_candidates, current_file), True

    all_candidates = values.by_bare_name.get(name, [])
    if len(all_candidates) == 1:
        return all_candidates[0], False
    if len(all_candidates) > 1:
        return nearest_by_directory(all_candidates, current_file), True

    return None, False
