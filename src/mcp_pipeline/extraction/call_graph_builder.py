from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath

from tree_sitter import Node

from mcp_pipeline.extraction.definition_index import DefinitionIndex, FunctionDef
from mcp_pipeline.extraction.import_index import ImportIndex
from mcp_pipeline.extraction.models import CallGraphNode, SourceLocation

MAX_LEVEL = 3


@dataclass
class CallSite:
    callee_name: str  # bare name being called, e.g. "get" in "requests.get(...)"
    receiver: str | None  # "self"/"this", a bare identifier, or None for an unqualified call
    raw_text: str  # the literal call expression source text, always captured


# A language-specific call-site extractor: given a definition's body node and
# its file's source bytes, returns every call expression found inside it.
# Provided per language in patterns/*.py; the resolution/BFS logic below is
# shared (architecture principle: shared logic parametrized by a per-language
# plug-in, not copy-pasted per language).
CallExtractor = Callable[[Node, bytes], list[CallSite]]


def build_call_graph(
    start_def: FunctionDef,
    definitions: DefinitionIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
    extract_calls: CallExtractor,
) -> CallGraphNode:
    """Builds the 3-level call graph tree rooted at `start_def` (the tool's
    own implementation = level 1). Terminates by construction at MAX_LEVEL —
    level-3 nodes are never call-site-scanned — which also means direct or
    mutual recursion can't cause an infinite loop without separate cycle
    detection.
    """
    return _build_node(start_def, level=1, definitions=definitions, imports_by_file=imports_by_file,
                        source_bytes_by_file=source_bytes_by_file, extract_calls=extract_calls)


def _build_node(
    fn_def: FunctionDef,
    level: int,
    definitions: DefinitionIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
    extract_calls: CallExtractor,
) -> CallGraphNode:
    node = CallGraphNode(
        level=level,
        resolved=True,
        external=False,
        ambiguous=False,
        qualified_name=fn_def.qualified_name,
        raw_call_text=None,
        source_location=SourceLocation(file=fn_def.file, start_line=fn_def.start_line, end_line=fn_def.end_line),
    )

    if level >= MAX_LEVEL:
        return node

    source_bytes = source_bytes_by_file[fn_def.file]
    seen_raw_texts: set[str] = set()
    for call_site in extract_calls(fn_def.body_node, source_bytes):
        if call_site.raw_text in seen_raw_texts:
            continue  # dedupe repeated identical calls within one body to a single edge
        seen_raw_texts.add(call_site.raw_text)

        resolved_def, ambiguous = resolve_call(
            call_site, current_file=fn_def.file, current_class=fn_def.class_name,
            definitions=definitions, imports_by_file=imports_by_file,
        )

        if resolved_def is None:
            node.calls.append(
                CallGraphNode(
                    level=level + 1, resolved=False, external=True, ambiguous=False,
                    qualified_name=None, raw_call_text=call_site.raw_text, source_location=None,
                )
            )
            continue

        child = _build_node(
            resolved_def, level + 1, definitions, imports_by_file, source_bytes_by_file, extract_calls
        )
        child.ambiguous = ambiguous
        child.raw_call_text = call_site.raw_text
        node.calls.append(child)

    return node


def resolve_call(
    call_site: CallSite,
    current_file: str,
    current_class: str | None,
    definitions: DefinitionIndex,
    imports_by_file: dict[str, ImportIndex],
) -> tuple[FunctionDef | None, bool]:
    """Implements the plan's 5-step resolution heuristic, first match wins.
    Returns (resolved_def_or_None, ambiguous). This is explicitly a
    best-effort heuristic (name-based, not type-resolved) — see the plan's
    accepted trade-offs.
    """
    name = call_site.callee_name

    # 1. self/this receiver -> same-class method.
    if call_site.receiver in ("self", "this") and current_class is not None:
        candidate = definitions.by_qualified_name.get(f"{current_class}.{name}")
        if candidate is not None:
            return candidate, False

    # 2. same bare name in the same file -> module-level/sibling helper.
    same_file_candidates = [d for d in definitions.by_bare_name.get(name, []) if d.file == current_file]
    if len(same_file_candidates) == 1:
        return same_file_candidates[0], False
    if len(same_file_candidates) > 1:
        return _nearest_by_directory(same_file_candidates, current_file), True

    # 3. receiver matches a known import alias -> prefer definitions whose
    #    file stem matches the imported module's last path segment, OR the
    #    alias itself (Python's `from . import cache_utils` / `import
    #    cache_utils` binds a name that IS the submodule's filename — the
    #    import's `module` field alone is uninformative for those forms,
    #    e.g. "." for a bare relative package import).
    if call_site.receiver is not None:
        import_index = imports_by_file.get(current_file, {})
        imported = import_index.get(call_site.receiver)
        if imported is not None:
            candidate_stems = {PurePosixPath(imported.module.lstrip(".")).name, call_site.receiver} - {""}
            module_candidates = [
                d for d in definitions.by_bare_name.get(name, [])
                if PurePosixPath(d.file).stem in candidate_stems
            ]
            if len(module_candidates) == 1:
                return module_candidates[0], False
            if len(module_candidates) > 1:
                return _nearest_by_directory(module_candidates, current_file), True

    # 4. repo-wide bare-name lookup.
    all_candidates = definitions.by_bare_name.get(name, [])
    if len(all_candidates) == 1:
        return all_candidates[0], False
    if len(all_candidates) > 1:
        return _nearest_by_directory(all_candidates, current_file), True

    # 5. not found anywhere in the repo -> external/dynamic, caller treats as a leaf.
    return None, False


def _nearest_by_directory(candidates: list[FunctionDef], current_file: str) -> FunctionDef:
    """A real guess, not a resolution — the caller marks the result
    `ambiguous: true` so it stays visible and discountable downstream."""
    current_parts = PurePosixPath(current_file).parent.parts

    def distance(d: FunctionDef) -> int:
        candidate_parts = PurePosixPath(d.file).parent.parts
        common = 0
        for a, b in zip(current_parts, candidate_parts):
            if a != b:
                break
            common += 1
        return len(current_parts) + len(candidate_parts) - 2 * common

    return min(candidates, key=distance)
