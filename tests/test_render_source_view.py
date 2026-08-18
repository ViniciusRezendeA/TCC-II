from __future__ import annotations

from pathlib import Path

from mcp_pipeline.extraction.models import CallGraphNode, SourceLocation
from mcp_pipeline.schema.render_source_view import render_source_view


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_renders_resolved_nodes_with_source_snippets_and_external_leaves_as_comments(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        "def get_weather(city):\n    return fetch(city)\n",
    )

    graph = CallGraphNode(
        level=1, resolved=True, external=False, ambiguous=False,
        qualified_name="get_weather", raw_call_text=None,
        source_location=SourceLocation(file="server.py", start_line=1, end_line=2),
        calls=[
            CallGraphNode(
                level=2, resolved=False, external=True, ambiguous=False,
                qualified_name=None, raw_call_text="fetch(city)", source_location=None,
            )
        ],
    )

    rendered = render_source_view(graph, repo)

    assert "def get_weather(city):" in rendered
    assert "return fetch(city)" in rendered
    assert "server.py:1-2" in rendered
    assert "fetch(city)" in rendered
    assert "não resolvida" in rendered


def test_renders_nested_levels_in_order(tmp_path):
    repo = tmp_path / "src"
    _write(repo / "a.py", "def a():\n    pass\n")
    _write(repo / "b.py", "def b():\n    pass\n")

    graph = CallGraphNode(
        level=1, resolved=True, external=False, ambiguous=False,
        qualified_name="a", raw_call_text=None,
        source_location=SourceLocation(file="a.py", start_line=1, end_line=2),
        calls=[
            CallGraphNode(
                level=2, resolved=True, external=False, ambiguous=False,
                qualified_name="b", raw_call_text="b()",
                source_location=SourceLocation(file="b.py", start_line=1, end_line=2),
            )
        ],
    )

    rendered = render_source_view(graph, repo)

    assert rendered.index("a.py") < rendered.index("b.py")
