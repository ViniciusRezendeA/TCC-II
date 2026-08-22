from __future__ import annotations

from pathlib import Path

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.parser_utils import (
    iter_source_files,
    template_string_literal_value,
)

SPEC = spec_for("Python")


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_iter_source_files_excludes_test_and_vendor_directories(tmp_path):
    _touch(tmp_path / "server.py")
    _touch(tmp_path / "lib" / "helpers.py")
    _touch(tmp_path / "tests" / "conftest.py")
    _touch(tmp_path / "examples" / "demo.py")
    _touch(tmp_path / "node_modules" / "pkg" / "script.py")
    _touch(tmp_path / "lib" / "__tests__" / "nested_test.py")
    _touch(tmp_path / "docs_src" / "structured_output" / "server.py")

    found = {p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path, SPEC)}

    assert found == {"server.py", "lib/helpers.py"}


def test_iter_source_files_skips_directories_matching_the_extension_glob(tmp_path):
    _touch(tmp_path / "server.py")
    (tmp_path / "submodule_placeholder.py").mkdir()  # e.g. an uninitialized git submodule

    found = {p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path, SPEC)}

    assert found == {"server.py"}


def test_iter_source_files_matches_all_configured_extensions(tmp_path):
    js_spec = spec_for("JavaScript")
    _touch(tmp_path / "a.js")
    _touch(tmp_path / "b.mjs")
    _touch(tmp_path / "c.cjs")
    _touch(tmp_path / "d.txt")

    found = {p.name for p in iter_source_files(tmp_path, js_spec)}

    assert found == {"a.js", "b.mjs", "c.cjs"}


def test_iter_source_files_excludes_colocated_test_ts_and_spec_ts_files(tmp_path):
    """Real case (punkpeye/fastmcp, the fastmcp npm framework's own repo):
    148 tool-registration call sites live in *.test.ts files sitting
    directly in ordinary source directories, not under anything named
    test/tests -- the directory-name filter alone doesn't catch these."""
    ts_spec = spec_for("TypeScript")
    _touch(tmp_path / "FastMCP.ts")
    _touch(tmp_path / "FastMCP.test.ts")
    _touch(tmp_path / "FastMCP.spec.ts")
    _touch(tmp_path / "tools" / "search.ts")
    _touch(tmp_path / "tools" / "search.test.ts")

    found = {p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path, ts_spec)}

    assert found == {"FastMCP.ts", "tools/search.ts"}


def test_iter_source_files_excludes_colocated_python_test_files(tmp_path):
    """Real case, already in the shipped dataset before this fix:
    awslabs/mcp has tools/rg/test_migration.py, a colocated pytest-style
    file (not under a tests/ directory) containing a real @mcp.tool()."""
    _touch(tmp_path / "server.py")
    _touch(tmp_path / "tools" / "rg" / "test_migration.py")
    _touch(tmp_path / "tools" / "rg" / "migration_test.py")
    _touch(tmp_path / "tools" / "rg" / "migration.py")

    found = {p.relative_to(tmp_path).as_posix() for p in iter_source_files(tmp_path, SPEC)}

    assert found == {"server.py", "tools/rg/migration.py"}


def test_template_string_literal_value_extracts_text_without_interpolation():
    from tree_sitter import Parser

    ts_spec = spec_for("TypeScript")
    parser = Parser(ts_spec.ts_language)
    source_bytes = b"const x = `Fetches a URL from the internet.`;"
    tree = parser.parse(source_bytes)

    template_node = next(
        n for n in _walk(tree.root_node) if n.type == "template_string"
    )

    assert template_string_literal_value(template_node, source_bytes) == "Fetches a URL from the internet."


def test_template_string_literal_value_returns_none_when_interpolated():
    from tree_sitter import Parser

    ts_spec = spec_for("TypeScript")
    parser = Parser(ts_spec.ts_language)
    source_bytes = b"const x = `Value is ${y}.`;"
    tree = parser.parse(source_bytes)

    template_node = next(
        n for n in _walk(tree.root_node) if n.type == "template_string"
    )

    assert template_string_literal_value(template_node, source_bytes) is None


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)
