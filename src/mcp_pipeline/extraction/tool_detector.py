from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import CallExtractor, build_call_graph
from mcp_pipeline.extraction.definition_index import (
    DefinitionExtractor,
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
from mcp_pipeline.extraction.patterns.python_patterns import (
    detect_fastmcp_tools,
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

ImportExtractor = Callable[[Node, bytes], ImportIndex]
ToolDetectorFn = Callable[[Node, bytes, str], list[ToolRecord]]


@dataclass(frozen=True)
class LanguageAdapter:
    """Bundles the 4 language-specific plug-ins the generic Etapa 2 pipeline
    needs. Each new language (Fase 5 of the plan: Java, C#) registers one of
    these in LANGUAGE_ADAPTERS — nothing else in this file changes.
    """

    extract_definitions: DefinitionExtractor
    extract_imports_fn: ImportExtractor
    extract_calls: CallExtractor
    detect_tools: ToolDetectorFn


LANGUAGE_ADAPTERS: dict[str, LanguageAdapter] = {
    "Python": LanguageAdapter(
        extract_definitions=python_extract_definitions,
        extract_imports_fn=python_extract_imports,
        extract_calls=python_extract_calls,
        detect_tools=detect_fastmcp_tools,
    ),
    # JavaScript, TypeScript, Java, C#: not yet implemented — see the plan's
    # Fase 3/5. Requesting one of these raises a clear error below rather
    # than silently returning no tools.
}


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
    if language not in LANGUAGE_ADAPTERS:
        raise ValueError(
            f"Etapa 2 ainda não implementa detecção para {language!r}. "
            f"Linguagens suportadas: {sorted(LANGUAGE_ADAPTERS)}."
        )
    adapter = LANGUAGE_ADAPTERS[language]
    spec = spec_for(language)

    definitions = build_definition_index(repo_src_root, spec, adapter.extract_definitions)

    imports_by_file: dict[str, ImportIndex] = {}
    source_bytes_by_file: dict[str, bytes] = {}
    tools: list[ToolRecord] = []
    for file_path in iter_source_files(repo_src_root, spec):
        tree, source_bytes = parse_file(file_path, spec)
        rel_path = relative_path(file_path, repo_src_root)
        imports_by_file[rel_path] = adapter.extract_imports_fn(tree.root_node, source_bytes)
        source_bytes_by_file[rel_path] = source_bytes
        tools.extend(adapter.detect_tools(tree.root_node, source_bytes, rel_path))

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
        results.append((tool, graph))
    return results
