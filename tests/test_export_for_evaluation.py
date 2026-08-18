from __future__ import annotations

from pathlib import Path

from mcp_pipeline.extraction.models import CallGraphNode, SourceLocation, ToolRecord
from mcp_pipeline.schema.export_for_evaluation import export_tool_for_evaluation


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sample_tool_and_graph(repo: Path):
    _write(repo / "server.py", "def get_weather(city):\n    return city\n")
    tool = ToolRecord(
        name="get_weather", description="Fetch the weather", description_is_literal=True,
        sdk_pattern="python.fastmcp_decorator",
        source_location=SourceLocation(file="server.py", start_line=1, end_line=2),
        qualified_name="get_weather",
    )
    graph = CallGraphNode(
        level=1, resolved=True, external=False, ambiguous=False,
        qualified_name="get_weather", raw_call_text=None,
        source_location=SourceLocation(file="server.py", start_line=1, end_line=2),
    )
    return tool, graph


def test_scenario_1_omits_source_code_field(tmp_path):
    repo = tmp_path / "src"
    tool, graph = _sample_tool_and_graph(repo)

    payload = export_tool_for_evaluation(tool, graph, repo, server_name="acme/weather-mcp", include_source=False)

    assert payload == {"name": "get_weather", "server_name": "acme/weather-mcp", "description": "Fetch the weather"}
    assert "SOURCE_CODE" not in payload


def test_scenario_2_includes_rendered_source_code(tmp_path):
    repo = tmp_path / "src"
    tool, graph = _sample_tool_and_graph(repo)

    payload = export_tool_for_evaluation(tool, graph, repo, server_name="acme/weather-mcp", include_source=True)

    assert payload["name"] == "get_weather"
    assert payload["server_name"] == "acme/weather-mcp"
    assert payload["description"] == "Fetch the weather"
    assert "def get_weather(city):" in payload["SOURCE_CODE"]
