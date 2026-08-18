from __future__ import annotations

from pathlib import Path

import pytest

from mcp_pipeline.extraction.tool_detector import detect_tools_with_call_graphs


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_detects_tools_and_builds_call_graphs_end_to_end(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        '''
from . import cache_utils
import requests


class WeatherServer:
    @mcp.tool()
    def get_weather(self, city: str) -> str:
        """Fetch the weather for a city."""
        return self._fetch(city)

    def _fetch(self, city):
        requests.get(city)
        return cache_utils.get_cached(city)


@mcp.tool(description="Second tool")
def other_tool(x):
    return x
''',
    )
    _write(repo / "cache_utils.py", "def get_cached(key):\n    return None\n")

    results = detect_tools_with_call_graphs(repo, "Python")

    assert len(results) == 2
    by_name = {tool.name: (tool, graph) for tool, graph in results}

    weather_tool, weather_graph = by_name["get_weather"]
    assert weather_tool.description == "Fetch the weather for a city."
    assert weather_graph.qualified_name == "WeatherServer.get_weather"
    assert weather_graph.calls[0].qualified_name == "WeatherServer._fetch"
    call_texts = {c.raw_call_text for c in weather_graph.calls[0].calls}
    assert "requests.get(city)" in call_texts
    assert "cache_utils.get_cached(city)" in call_texts

    other_tool_record, other_graph = by_name["other_tool"]
    assert other_tool_record.description == "Second tool"
    assert other_graph.calls == []  # `return x` has no call expressions


def test_repo_with_no_tools_returns_empty_list(tmp_path):
    repo = tmp_path / "src"
    _write(repo / "readme_generator.py", "def build_readme():\n    return 'hello'\n")

    results = detect_tools_with_call_graphs(repo, "Python")

    assert results == []


def test_unsupported_language_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Java"):
        detect_tools_with_call_graphs(tmp_path, "Java")
