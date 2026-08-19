from __future__ import annotations

from pathlib import Path

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.parser_utils import iter_source_files

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


def test_iter_source_files_matches_all_configured_extensions(tmp_path):
    js_spec = spec_for("JavaScript")
    _touch(tmp_path / "a.js")
    _touch(tmp_path / "b.mjs")
    _touch(tmp_path / "c.cjs")
    _touch(tmp_path / "d.txt")

    found = {p.name for p in iter_source_files(tmp_path, js_spec)}

    assert found == {"a.js", "b.mjs", "c.cjs"}
