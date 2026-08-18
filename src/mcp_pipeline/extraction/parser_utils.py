from __future__ import annotations

from pathlib import Path

from tree_sitter import Language, Node, Parser, Tree

from mcp_pipeline.extraction.language_registry import LanguageSpec


def parse_file(path: Path, spec: LanguageSpec) -> tuple[Tree, bytes]:
    source_bytes = path.read_bytes()
    parser = Parser(spec.ts_language)
    tree = parser.parse(source_bytes)
    return tree, source_bytes


def iter_source_files(repo_src_root: Path, spec: LanguageSpec):
    for ext in spec.extensions:
        yield from repo_src_root.rglob(f"*{ext}")


def node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def string_literal_value(node: Node, source_bytes: bytes) -> str | None:
    """Extracts the content of a string literal node, stripping quote/prefix
    delimiters correctly by reading the grammar's own `string_content` child
    rather than slicing off a fixed number of characters (which breaks on
    triple-quoted strings, f-string prefixes, etc). Returns None if `node`
    isn't a plain string literal (e.g. it's an f-string, a name reference, a
    concatenation) — callers should fall back to capturing the raw source
    text in that case instead of silently dropping the value.
    """
    if node.type != "string":
        return None
    content_parts = [c for c in node.children if c.type == "string_content"]
    if not content_parts:
        return ""
    return "".join(node_text(c, source_bytes) for c in content_parts)


def run_query(language: Language, query_str: str, root: Node) -> list[tuple[int, dict[str, list[Node]]]]:
    query = language.query(query_str)
    return query.matches(root)


def relative_path(file_path: Path, repo_src_root: Path) -> str:
    return file_path.relative_to(repo_src_root).as_posix()


def line_range(node: Node) -> tuple[int, int]:
    """1-indexed, inclusive line range — tree-sitter's start_point/end_point
    rows are 0-indexed."""
    return node.start_point[0] + 1, node.end_point[0] + 1
