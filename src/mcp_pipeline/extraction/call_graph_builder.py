from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

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


def call_graph_depth(node: CallGraphNode) -> int:
    """Depth of the call graph tree rooted at `node`: the highest `level`
    reached by any node in the tree. The tool's own level-1 node alone (no
    calls) has depth 1. Bounded to MAX_LEVEL by construction -- build_call_graph()
    never produces a node past level MAX_LEVEL, so this is a deliberately
    coarse {1, 2, 3}-valued metric, not an unlimited depth.
    """
    if not node.calls:
        return node.level
    return max(call_graph_depth(child) for child in node.calls)


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


def collect_reachable_definitions(
    start_def: FunctionDef,
    definitions: DefinitionIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
    extract_calls: CallExtractor,
) -> list[FunctionDef]:
    """Every FunctionDef transitively reachable from `start_def` via resolved
    internal calls, deduped by qualified_name and including `start_def`
    itself. Used to measure "the complete tool" for `loc`/cyclomatic
    complexity (summed across the tool's own body AND every helper it calls,
    however deep) -- deliberately NOT bounded by MAX_LEVEL like
    `build_call_graph` above. That 3-level cap exists only to keep the
    *serialized* call graph tree small/readable; it isn't a claim that code
    past level 3 stops being part of the tool's own implementation.

    A visited-by-qualified_name set (rather than the level cap) is what
    keeps direct/mutual recursion from looping forever here, and also
    naturally dedupes diamond-shaped call graphs (two callees sharing one
    helper) so shared helpers aren't double-counted into loc/complexity.

    Only resolutions resolve_call marks unambiguous, AND that don't rely on
    its step-4 repo-wide-by-bare-name fallback (see resolve_call's
    `allow_repo_wide_fallback` docstring), are followed/counted here. Without
    this, a single wrong "only one function in the whole repo happens to be
    named this" guess snowballs: every further call made from that unrelated
    function is itself expanded with no depth limit to catch it, unlike
    build_call_graph's capped tree where the same wrong guess is a bounded,
    visible, discardable curiosity in one JSON node.
    """
    visited: dict[str, FunctionDef] = {start_def.qualified_name: start_def}
    queue: list[FunctionDef] = [start_def]
    while queue:
        current = queue.pop()
        source_bytes = source_bytes_by_file[current.file]
        seen_raw_texts: set[str] = set()
        for call_site in extract_calls(current.body_node, source_bytes):
            if call_site.raw_text in seen_raw_texts:
                continue
            seen_raw_texts.add(call_site.raw_text)

            resolved_def, ambiguous = resolve_call(
                call_site, current_file=current.file, current_class=current.class_name,
                definitions=definitions, imports_by_file=imports_by_file,
                allow_repo_wide_fallback=call_site.receiver is None,
            )
            if resolved_def is None or ambiguous or resolved_def.qualified_name in visited:
                continue
            visited[resolved_def.qualified_name] = resolved_def
            queue.append(resolved_def)

    return list(visited.values())


def resolve_call(
    call_site: CallSite,
    current_file: str,
    current_class: str | None,
    definitions: DefinitionIndex,
    imports_by_file: dict[str, ImportIndex],
    allow_repo_wide_fallback: bool = True,
) -> tuple[FunctionDef | None, bool]:
    """Implements the plan's 5-step resolution heuristic, first match wins.
    Returns (resolved_def_or_None, ambiguous). This is explicitly a
    best-effort heuristic (name-based, not type-resolved) — see the plan's
    accepted trade-offs.

    `allow_repo_wide_fallback=False` disables step 4 (below) for calls that
    have a receiver (`obj.method(...)`). build_call_graph (bounded to 3
    levels) always leaves this True, its original behavior.
    collect_reachable_definitions passes False for receiver-qualified calls
    only -- a receiver that didn't already match self/this or a tracked
    import (steps 1/3) is weak evidence the callee is a local function at
    all, and real case Accenture/mcp-bench (a repo bundling dozens of
    unrelated demo MCP servers under one src/) showed step 4 actively wrong
    here, not just imprecise: `client.p.submission.fetch(post_id)` (a PRAW
    library call) "resolved" to an unrelated local `fetch` tool defined in a
    completely different bundled server, purely because it's the only
    repo-wide function literally named `fetch`. Under the 3-level cap this
    is a bounded, visible curiosity in one JSON node; in an unbounded
    transitive closure the same mistake pulls that unrelated server's entire
    call graph in, and repeats at every further hop -- one tool's aggregate
    exploded to loc=11295/cc=1591 before this was disabled.

    Receiver-less calls (`helper(x)`) still get step 4 in the unbounded
    closure, unlike qualified calls: it's the documented, intentional way
    JS/TS's CommonJS `const { getCached } = require(...); getCached()` gets
    resolved at all (see extract_imports's docstring in
    patterns/ecmascript_common.py -- it doesn't track require() bindings),
    and an unqualified bare name colliding repo-wide is a much rarer failure
    mode than a receiver being silently misread as a local module.
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
        return nearest_by_directory(same_file_candidates, current_file), True

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
                return nearest_by_directory(module_candidates, current_file), True

    # 4. repo-wide bare-name lookup.
    if not allow_repo_wide_fallback:
        return None, False
    all_candidates = definitions.by_bare_name.get(name, [])
    if len(all_candidates) == 1:
        return all_candidates[0], False
    if len(all_candidates) > 1:
        return nearest_by_directory(all_candidates, current_file), True

    # 5. not found anywhere in the repo -> external/dynamic, caller treats as a leaf.
    return None, False


class _HasFile(Protocol):
    file: str


def nearest_by_directory[T: _HasFile](candidates: list[T], current_file: str) -> T:
    """A real guess, not a resolution — the caller marks the result
    `ambiguous: true` so it stays visible and discountable downstream.
    Generic over anything with a `.file` field (not just FunctionDef) so
    value_index.py's resolve_value() can reuse this exact tiebreak for
    value-reference ambiguity, not just call resolution.
    """
    current_parts = PurePosixPath(current_file).parent.parts

    def distance(d: T) -> int:
        candidate_parts = PurePosixPath(d.file).parent.parts
        common = 0
        for a, b in zip(current_parts, candidate_parts):
            if a != b:
                break
            common += 1
        return len(current_parts) + len(candidate_parts) - 2 * common

    return min(candidates, key=distance)
