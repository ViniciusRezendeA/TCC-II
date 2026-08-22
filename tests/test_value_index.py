from __future__ import annotations

from mcp_pipeline.extraction.import_index import ImportedName
from mcp_pipeline.extraction.value_index import ValueDef, ValueIndex, resolve_value


def val(bare_name, file):
    return ValueDef(bare_name=bare_name, file=file, value_node=None)


def test_resolve_value_same_file_takes_priority():
    values = ValueIndex.build([val("search_tool", "server.py"), val("search_tool", "other.py")])

    resolved, ambiguous = resolve_value("search_tool", current_file="server.py", values=values, imports_by_file={})

    assert resolved.file == "server.py"
    assert ambiguous is False


def test_resolve_value_cross_file_via_import_stem_match():
    values = ValueIndex.build([val("search_tool", "tools/search.py")])
    imports = {"server.py": {"search_tool": ImportedName(module=".tools.search", original_name="search_tool")}}

    resolved, ambiguous = resolve_value(
        "search_tool", current_file="server.py", values=values, imports_by_file=imports
    )

    assert resolved.file == "tools/search.py"
    assert ambiguous is False


def test_resolve_value_repo_wide_unique_fallback():
    values = ValueIndex.build([val("ping_tool", "tools/ping.py")])

    resolved, ambiguous = resolve_value("ping_tool", current_file="server.py", values=values, imports_by_file={})

    assert resolved.file == "tools/ping.py"
    assert ambiguous is False


def test_resolve_value_ambiguous_when_multiple_candidates_tie():
    values = ValueIndex.build([val("TOOL", "a/x.py"), val("TOOL", "b/y.py")])

    resolved, ambiguous = resolve_value("TOOL", current_file="c/z.py", values=values, imports_by_file={})

    assert resolved is not None
    assert ambiguous is True


def test_resolve_value_returns_none_when_not_found_anywhere():
    values = ValueIndex.build([])

    resolved, ambiguous = resolve_value("missing", current_file="server.py", values=values, imports_by_file={})

    assert resolved is None
    assert ambiguous is False
