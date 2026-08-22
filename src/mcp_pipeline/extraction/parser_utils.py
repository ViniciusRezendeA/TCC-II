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

# Directory-name exclusion (above) doesn't catch test files *colocated* with
# source under an ordinary directory — the standard Vitest/Jest convention
# for TS/JS (`FastMCP.test.ts` next to `FastMCP.ts`) and the standard pytest
# convention for Python (`test_migration.py` next to `migration.py`), neither
# of which lives under a directory literally named test/tests. Found
# empirically, not assumed: punkpeye/fastmcp (the fastmcp npm framework's own
# repo) has 148 `.addTool(` call sites, all inside colocated `*.test.ts`
# files outside any test/-named directory. The identical class of bug is
# already live in the shipped dataset: awslabs/mcp has a real
# `@mcp.tool(name='test-migration')` inside a colocated
# tools/rg/test_migration.py, currently counted as a production tool.
EXCLUDED_FILENAME_SUFFIXES = (
    ".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx",
    ".test.js", ".spec.js", ".test.jsx", ".spec.jsx",
    ".test.mjs", ".spec.mjs", ".test.cjs", ".spec.cjs",
)


def _is_excluded_by_filename(path: Path) -> bool:
    name_lower = path.name.lower()
    if name_lower.endswith(EXCLUDED_FILENAME_SUFFIXES):
        return True
    if path.suffix == ".py":
        stem_lower = path.stem.lower()
        if stem_lower.startswith("test_") or stem_lower.endswith("_test"):
            return True
    return False


def iter_source_files(repo_src_root: Path, spec: LanguageSpec):
    for ext in spec.extensions:
        for path in repo_src_root.rglob(f"*{ext}"):
            # rglob matches directories too, not just files -- real case found running
            # against 206 cloned repos: an uninitialized git submodule left behind an
            # empty placeholder directory whose name happened to end in ".py"
            # (gmh5225/awesome-game-security), which crashed parse_file() with
            # "[Errno 21] Is a directory" instead of being silently skipped like any
            # other non-source path.
            if not path.is_file():
                continue
            rel_parts = path.relative_to(repo_src_root).parts[:-1]
            if any(part.lower() in EXCLUDED_DIR_NAMES for part in rel_parts):
                continue
            if _is_excluded_by_filename(path):
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


def template_string_literal_value(node: Node, source_bytes: bytes) -> str | None:
    """A JS/TS template (backtick) literal with no `${...}` interpolation is
    functionally a plain string literal — real, common case for multi-line
    descriptions (e.g. firecrawl-mcp-server's `firecrawl_scrape` description,
    already in the shipped dataset, mis-flagged `description_is_literal:
    false` before this existed). A template with interpolation is NOT safe
    to treat as literal: naively concatenating only its `string_fragment`
    children would silently drop the interpolated parts and fabricate a
    plausible-but-wrong value, so this returns None whenever a
    `template_substitution` child is present, same as string_literal_value's
    "return None, let the caller fall back to raw-text capture" contract.
    """
    if node.type != "template_string":
        return None
    if any(c.type == "template_substitution" for c in node.children):
        return None
    return "".join(node_text(c, source_bytes) for c in node.children if c.type == "string_fragment")


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
