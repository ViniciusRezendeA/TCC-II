from __future__ import annotations

from mcp_pipeline.extraction.models import CallGraphNode, SourceLocation, ToolRecord


def test_source_location_round_trips():
    loc = SourceLocation(file="src/server.py", start_line=10, end_line=25)
    assert SourceLocation.from_dict(loc.to_dict()) == loc


def test_tool_record_round_trips():
    tool = ToolRecord(
        name="get_weather",
        description="Fetch the weather",
        description_is_literal=True,
        sdk_pattern="python.fastmcp_decorator",
        source_location=SourceLocation(file="server.py", start_line=10, end_line=12),
        qualified_name="WeatherServer.get_weather",
    )
    reloaded = ToolRecord.from_dict(tool.to_dict())
    assert reloaded == tool


def test_call_graph_node_round_trips_with_nested_children():
    leaf_external = CallGraphNode(
        level=3, resolved=False, external=True, ambiguous=False,
        qualified_name=None, raw_call_text="requests.get(city)", source_location=None,
    )
    level2 = CallGraphNode(
        level=2, resolved=True, external=False, ambiguous=False,
        qualified_name="WeatherServer._fetch", raw_call_text="self._fetch(city)",
        source_location=SourceLocation(file="server.py", start_line=14, end_line=16),
        calls=[leaf_external],
    )
    root = CallGraphNode(
        level=1, resolved=True, external=False, ambiguous=False,
        qualified_name="WeatherServer.get_weather", raw_call_text=None,
        source_location=SourceLocation(file="server.py", start_line=10, end_line=12),
        calls=[level2],
    )

    reloaded = CallGraphNode.from_dict(root.to_dict())

    assert reloaded == root
    assert reloaded.calls[0].calls[0].external is True
    assert reloaded.calls[0].calls[0].source_location is None
