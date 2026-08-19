from __future__ import annotations

from pathlib import Path

from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree

from mcp_pipeline.extraction.language_registry import LanguageSpec


def parse_file(path: Path, spec: LanguageSpec) -> tuple[Tree, bytes]:
    source_bytes = path.read_bytes()
    parser = Parser(spec.ts_language)
    tree = parser.parse(source_bytes)
    return tree, source_bytes


# Directories whose contents shouldn't count as "the MCP server's tools" —
# found empirically, not assumed: PrefectHQ/fastmcp (the FastMCP framework's
# own repo) has 623 @mcp.tool()-decorated functions in tests/conftest.py
# alone, used to test the framework itself, plus 68 more in examples/ — 89%
# of its raw tool count before this filter. modelcontextprotocol/python-sdk
# similarly has all 98 of its detected tools in docs_src/ (documentation
# code samples, not a server). Excluding these directories is a data-quality
# fix, not a language-detection change: a repo's *production* tool surface
# is what Etapa 2 is meant to measure.
EXCLUDED_DIR_NAMES = frozenset(
    {
        "test", "tests", "__tests__", "spec", "specs", "testing",
        "example", "examples", "demo", "demos", "sample", "samples",
        "doc", "docs", "docs_src", "documentation",
        "node_modules", "vendor", "third_party", "dist", "build", "out",
        ".venv", "venv", ".git",
    }
)


def iter_source_files(repo_src_root: Path, spec: LanguageSpec):
    for ext in spec.extensions:
        for path in repo_src_root.rglob(f"*{ext}"):
            rel_parts = path.relative_to(repo_src_root).parts[:-1]
            if any(part.lower() in EXCLUDED_DIR_NAMES for part in rel_parts):
                continue
            yield path


def node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def string_literal_value(
    node: Node, source_bytes: bytes, content_node_type: str = "string_content", node_type: str = "string"
) -> str | None:
    """Extracts the content of a string literal node, stripping quote/prefix
    delimiters correctly by reading the grammar's own content-child node
    rather than slicing off a fixed number of characters (which breaks on
    triple-quoted strings, f-string prefixes, etc). Both `node_type` and
    `content_node_type` differ by grammar — verified per language, not
    assumed identical across grammars: Python's string node is "string"
    with "string_content" children; JS/TS's is also "string" but with
    "string_fragment" children; Java's is "string_literal" (not "string" —
    caught a real bug here: reusing the JS/TS default silently produced
    quote-wrapped garbage for Java instead of extracting the value) with
    "string_fragment" children. Returns None if `node` isn't a plain string
    literal (e.g. a template literal, a name reference, a concatenation) —
    callers should fall back to capturing the raw source text in that case
    instead of silently dropping the value.
    """
    if node.type != node_type:
        return None
    content_parts = [c for c in node.children if c.type == content_node_type]
    if not content_parts:
        return ""
    return "".join(node_text(c, source_bytes) for c in content_parts)


def run_query(language: Language, query_str: str, root: Node) -> list[tuple[int, dict[str, list[Node]]]]:
    """`Language.query()` + `Query.matches()` (tree-sitter <=0.23.x) was
    replaced by an explicit `Query`/`QueryCursor` split in 0.25.x — this is
    the only place that needed to change; every caller still gets the same
    `list[tuple[int, dict[str, list[Node]]]]` shape, verified identical
    across both API generations.
    """
    query = Query(language, query_str)
    cursor = QueryCursor(query)
    return cursor.matches(root)


def relative_path(file_path: Path, repo_src_root: Path) -> str:
    return file_path.relative_to(repo_src_root).as_posix()


def line_range(node: Node) -> tuple[int, int]:
    """1-indexed, inclusive line range — tree-sitter's start_point/end_point
    rows are 0-indexed."""
    return node.start_point[0] + 1, node.end_point[0] + 1
