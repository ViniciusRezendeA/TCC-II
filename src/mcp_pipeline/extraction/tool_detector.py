from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import CallExtractor, build_call_graph, call_graph_depth
from mcp_pipeline.extraction.definition_index import (
    DefinitionExtractor,
    DefinitionIndex,
    build_definition_index,
)
from mcp_pipeline.extraction.import_index import ImportIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.models import CallGraphNode, ToolRecord
from mcp_pipeline.extraction.parser_utils import (
    iter_source_files,
    parse_file,
    relative_path,
)
from mcp_pipeline.extraction.patterns.csharp_patterns import (
    detect_csharp_tools,
)
from mcp_pipeline.extraction.patterns.csharp_patterns import (
    extract_calls as csharp_extract_calls,
)
from mcp_pipeline.extraction.patterns.csharp_patterns import (
    extract_definitions as csharp_extract_definitions,
)
from mcp_pipeline.extraction.patterns.csharp_patterns import (
    extract_imports as csharp_extract_imports,
)
from mcp_pipeline.extraction.patterns.java_patterns import (
    detect_spring_ai_tools,
)
from mcp_pipeline.extraction.patterns.java_patterns import (
    extract_calls as java_extract_calls,
)
from mcp_pipeline.extraction.patterns.java_patterns import (
    extract_definitions as java_extract_definitions,
)
from mcp_pipeline.extraction.patterns.java_patterns import (
    extract_imports as java_extract_imports,
)
from mcp_pipeline.extraction.patterns.javascript_patterns import (
    detect_fastmcp_npm_addtool as detect_javascript_fastmcp_npm_addtool,
)
from mcp_pipeline.extraction.patterns.javascript_patterns import (
    detect_lowlevel_set_request_handler as detect_javascript_lowlevel_set_request_handler,
)
from mcp_pipeline.extraction.patterns.javascript_patterns import (
    detect_mcp_tools as detect_javascript_tools,
)
from mcp_pipeline.extraction.patterns.javascript_patterns import (
    extract_calls as javascript_extract_calls,
)
from mcp_pipeline.extraction.patterns.javascript_patterns import (
    extract_definitions as javascript_extract_definitions,
)
from mcp_pipeline.extraction.patterns.javascript_patterns import (
    extract_imports as javascript_extract_imports,
)
from mcp_pipeline.extraction.patterns.javascript_patterns import (
    extract_values as javascript_extract_values,
)
from mcp_pipeline.extraction.patterns.python_patterns import (
    detect_fastmcp_tools,
)
from mcp_pipeline.extraction.patterns.python_patterns import (
    detect_lowlevel_list_tools as detect_python_lowlevel_list_tools,
)
from mcp_pipeline.extraction.patterns.python_patterns import (
    extract_calls as python_extract_calls,
)
from mcp_pipeline.extraction.patterns.python_patterns import (
    extract_definitions as python_extract_definitions,
)
from mcp_pipeline.extraction.patterns.python_patterns import (
    extract_imports as python_extract_imports,
)
from mcp_pipeline.extraction.patterns.python_patterns import (
    extract_values as python_extract_values,
)
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    detect_fastmcp_npm_addtool as detect_typescript_fastmcp_npm_addtool,
)
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    detect_lowlevel_set_request_handler as detect_typescript_lowlevel_set_request_handler,
)
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    detect_mcp_tools as detect_typescript_tools,
)
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    extract_calls as typescript_extract_calls,
)
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    extract_definitions as typescript_extract_definitions,
)
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    extract_imports as typescript_extract_imports,
)
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    extract_values as typescript_extract_values,
)
from mcp_pipeline.extraction.value_index import (
    ValueExtractor,
    ValueIndex,
    build_value_index,
)

ImportExtractor = Callable[[Node, bytes], ImportIndex]
ToolDetectorFn = Callable[[Node, bytes, str], list[ToolRecord]]
# A "context" tool detector needs repo-wide state (the definition index, the
# value index, every file's imports and source bytes) to resolve cross-file
# references -- unlike ToolDetectorFn, which is single-file. Used by the 2
# new low-level patterns (Python's @list_tools(), TS/JS's setRequestHandler);
# the existing single-file patterns (.tool()/.registerTool()/addTool/etc.)
# stay ToolDetectorFn.
ContextToolDetectorFn = Callable[
    [Node, bytes, str, DefinitionIndex, ValueIndex, dict[str, ImportIndex], dict[str, bytes]],
    list[ToolRecord],
]


def combine_tool_detectors(*fns: ToolDetectorFn) -> ToolDetectorFn:
    """Runs multiple independent ToolDetectorFns against the same file and
    concatenates their results — lets a language register more than one
    pattern (e.g. TS/JS now has the high-level `.tool()`/`.registerTool()`
    detector plus the third-party `fastmcp` npm `.addTool()` detector)
    without LanguageAdapter.detect_tools needing to become a list itself.
    Structural double-matching across patterns is not a practical concern:
    each pattern anchors on a disjoint method/decorator name, and a single
    AST node can't simultaneously match two different names.
    """

    def _combined(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
        results: list[ToolRecord] = []
        for fn in fns:
            results.extend(fn(root, source_bytes, rel_path))
        return results

    return _combined


@dataclass(frozen=True)
class LanguageAdapter:
    """Bundles the language-specific plug-ins the generic Etapa 2 pipeline
    needs. Each new language registers one of these in LANGUAGE_ADAPTERS —
    nothing else in this file changes.

    `extract_values`/`detect_tools_with_context` are optional (default
    None), added for the 2 new low-level patterns that need repo-wide
    cross-file value resolution (Python's @list_tools(), TS/JS's
    setRequestHandler) — Java and C# leave both unset, zero behavior change.
    """

    extract_definitions: DefinitionExtractor
    extract_imports_fn: ImportExtractor
    extract_calls: CallExtractor
    detect_tools: ToolDetectorFn
    extract_values: ValueExtractor | None = None
    detect_tools_with_context: ContextToolDetectorFn | None = None


LANGUAGE_ADAPTERS: dict[str, LanguageAdapter] = {
    "Python": LanguageAdapter(
        extract_definitions=python_extract_definitions,
        extract_imports_fn=python_extract_imports,
        extract_calls=python_extract_calls,
        detect_tools=detect_fastmcp_tools,
        extract_values=python_extract_values,
        detect_tools_with_context=detect_python_lowlevel_list_tools,
    ),
    "TypeScript": LanguageAdapter(
        extract_definitions=typescript_extract_definitions,
        extract_imports_fn=typescript_extract_imports,
        extract_calls=typescript_extract_calls,
        detect_tools=combine_tool_detectors(detect_typescript_tools, detect_typescript_fastmcp_npm_addtool),
        extract_values=typescript_extract_values,
        detect_tools_with_context=detect_typescript_lowlevel_set_request_handler,
    ),
    "JavaScript": LanguageAdapter(
        extract_definitions=javascript_extract_definitions,
        extract_imports_fn=javascript_extract_imports,
        extract_calls=javascript_extract_calls,
        detect_tools=combine_tool_detectors(detect_javascript_tools, detect_javascript_fastmcp_npm_addtool),
        extract_values=javascript_extract_values,
        detect_tools_with_context=detect_javascript_lowlevel_set_request_handler,
    ),
    "Java": LanguageAdapter(
        extract_definitions=java_extract_definitions,
        extract_imports_fn=java_extract_imports,
        extract_calls=java_extract_calls,
        detect_tools=detect_spring_ai_tools,
    ),
    "C#": LanguageAdapter(
        extract_definitions=csharp_extract_definitions,
        extract_imports_fn=csharp_extract_imports,
        extract_calls=csharp_extract_calls,
        detect_tools=detect_csharp_tools,
    ),
}


def _effective_language(repo_src_root: Path, declared_language: str) -> str:
    """GitHub's `primary_language` classification doesn't always match the
    real dominant source language -- confirmed real cases, all declared
    `JavaScript` despite being majority TypeScript in source (likely because
    a committed `dist/`/build output of compiled `.js` skews GitHub's
    byte-count language detection): `firecrawl/firecrawl-mcp-server`,
    `grab/cursor-talk-to-figma-mcp`, `21st-dev/magic-mcp`, and others, found
    while investigating the 135 repos that yielded 0 tools after the full
    Etapa 2 run.

    Deliberately narrow: only the JavaScript-declared-but-actually-TypeScript
    direction is checked (this specific pair is the one where a compiled
    build artifact plausibly causes GitHub's own classifier to disagree with
    the real source) -- this is a targeted correction to which language spec
    to use for ONE already-selected repo, not general multi-language
    scanning (already explicitly out of scope; see the plan).
    """
    if declared_language != "JavaScript":
        return declared_language
    n_js = sum(1 for _ in iter_source_files(repo_src_root, spec_for("JavaScript")))
    n_ts = sum(1 for _ in iter_source_files(repo_src_root, spec_for("TypeScript")))
    return "TypeScript" if n_ts > n_js else declared_language


def detect_tools_with_call_graphs(
    repo_src_root: Path, language: str
) -> list[tuple[ToolRecord, CallGraphNode]]:
    """Runs the full Etapa 2 pipeline for one repo: locate tools, build the
    repo-wide definition/import indices once, then build each tool's 3-level
    call graph. Returns one (ToolRecord, CallGraphNode) pair per tool found;
    a repo with zero tools returns an empty list (a legitimate outcome — see
    the plan's discussion of candidate repos that turn out not to be real
    MCP servers).
    """
    language = _effective_language(repo_src_root, language)
    if language not in LANGUAGE_ADAPTERS:
        raise ValueError(
            f"Etapa 2 ainda não implementa detecção para {language!r}. "
            f"Linguagens suportadas: {sorted(LANGUAGE_ADAPTERS)}."
        )
    adapter = LANGUAGE_ADAPTERS[language]
    spec = spec_for(language)

    definitions = build_definition_index(repo_src_root, spec, adapter.extract_definitions)

    # Phase 1: single pass over every file -- imports, source bytes, and
    # every single-file tool pattern (no repo-wide context needed). Parsed
    # roots are cached so Phase 2 below doesn't re-parse.
    imports_by_file: dict[str, ImportIndex] = {}
    source_bytes_by_file: dict[str, bytes] = {}
    roots_by_file: dict[str, Node] = {}
    tools: list[ToolRecord] = []
    for file_path in iter_source_files(repo_src_root, spec):
        tree, source_bytes = parse_file(file_path, spec)
        rel_path = relative_path(file_path, repo_src_root)
        imports_by_file[rel_path] = adapter.extract_imports_fn(tree.root_node, source_bytes)
        source_bytes_by_file[rel_path] = source_bytes
        roots_by_file[rel_path] = tree.root_node
        tools.extend(adapter.detect_tools(tree.root_node, source_bytes, rel_path))

    # Phase 2: only for languages with a "context" pattern (Python's
    # low-level @list_tools(), TS/JS's setRequestHandler). Deferred until
    # every file's imports/source_bytes are populated repo-wide, because a
    # value reference can point at a file Phase 1 hasn't reached yet in
    # rglob order (real case: blazickjp/arxiv-mcp-server's server.py
    # references a constant defined in tools/search.py).
    if adapter.detect_tools_with_context is not None:
        values = (
            build_value_index(repo_src_root, spec, adapter.extract_values)
            if adapter.extract_values is not None
            else ValueIndex.build([])
        )
        for rel_path, root_node in roots_by_file.items():
            tools.extend(
                adapter.detect_tools_with_context(
                    root_node, source_bytes_by_file[rel_path], rel_path,
                    definitions, values, imports_by_file, source_bytes_by_file,
                )
            )

    results: list[tuple[ToolRecord, CallGraphNode]] = []
    for tool in tools:
        start_def = definitions.by_qualified_name.get(tool.qualified_name)
        if start_def is None:
            # A tool was detected but its own definition didn't make it into
            # the index (e.g. a decorator/def-shape mismatch between the two
            # queries) — skip rather than crash the whole repo's extraction;
            # this should be rare and is worth investigating if it shows up
            # at scale, not silently normal.
            continue
        graph = build_call_graph(start_def, definitions, imports_by_file, source_bytes_by_file, adapter.extract_calls)
        # From start_def/graph, NOT tool.source_location: for JS/TS's
        # .tool()/.registerTool() patterns, source_location is the registration
        # call site, which can have a different line range than the handler
        # itself (see models.py's ToolRecord.loc docstring).
        tool.loc = start_def.end_line - start_def.start_line + 1
        tool.call_graph_depth = call_graph_depth(graph)
        results.append((tool, graph))
    return results
