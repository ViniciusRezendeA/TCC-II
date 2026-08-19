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


def test_typescript_detects_tools_and_builds_call_graphs_end_to_end(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.ts",
        '''
import { getCached } from "./cache";

server.tool("get_weather", "Fetch the weather", schema, async (args) => {
  const raw = await fetchRaw(args.city);
  return getCached(raw);
});

async function fetchRaw(city) {
  return externalLib.request(city);
}
''',
    )
    _write(repo / "cache.ts", "export function getCached(key) {\n  return null;\n}\n")

    results = detect_tools_with_call_graphs(repo, "TypeScript")

    assert len(results) == 1
    tool, graph = results[0]
    assert tool.name == "get_weather"
    assert tool.description == "Fetch the weather"
    level2_names = {c.qualified_name for c in graph.calls}
    assert level2_names == {"fetchRaw", "getCached"}
    fetch_raw_node = next(c for c in graph.calls if c.qualified_name == "fetchRaw")
    assert fetch_raw_node.calls[0].external is True


def test_javascript_detects_tools_via_registertool_and_named_handler(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.js",
        '''
const { getCached } = require("./cache");

server.registerTool("get_weather", {
  description: "Fetch the weather",
  inputSchema: schema,
}, handleGetWeather);

async function handleGetWeather(args) {
  return getCached(args.city);
}
''',
    )
    _write(repo / "cache.js", "function getCached(key) {\n  return null;\n}\nmodule.exports = { getCached };\n")

    results = detect_tools_with_call_graphs(repo, "JavaScript")

    assert len(results) == 1
    tool, graph = results[0]
    assert tool.name == "get_weather"
    assert tool.description == "Fetch the weather"
    assert tool.sdk_pattern == "javascript.registerTool"
    assert graph.qualified_name == "handleGetWeather"
    assert graph.calls[0].qualified_name == "getCached"


def test_java_detects_springai_tool_and_builds_call_graph(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "WeatherService.java",
        """
public class WeatherService {
    @Tool(description = "Fetch the weather for a city")
    public String getWeather(String city) {
        return this.fetch(city);
    }

    private String fetch(String city) {
        return CacheUtils.get(city);
    }
}
""",
    )
    _write(
        repo / "CacheUtils.java",
        "public class CacheUtils {\n    public static String get(String key) {\n        return null;\n    }\n}\n",
    )

    results = detect_tools_with_call_graphs(repo, "Java")

    assert len(results) == 1
    tool, graph = results[0]
    assert tool.name == "getWeather"
    assert tool.description == "Fetch the weather for a city"
    assert graph.qualified_name == "WeatherService.getWeather"
    assert graph.calls[0].qualified_name == "WeatherService.fetch"
    assert graph.calls[0].calls[0].qualified_name == "CacheUtils.get"


def test_csharp_detects_tool_and_builds_call_graph(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "WeatherTool.cs",
        """
public sealed class WeatherTool
{
    [McpServerTool, Description("Fetch the weather for a city")]
    public static string GetWeather(string city)
    {
        return Fetch(city);
    }

    private static string Fetch(string city)
    {
        return Cache.Get(city);
    }
}
""",
    )
    _write(
        repo / "Cache.cs",
        "public static class Cache {\n    public static string Get(string key) { return null; }\n}\n",
    )

    results = detect_tools_with_call_graphs(repo, "C#")

    assert len(results) == 1
    tool, graph = results[0]
    assert tool.name == "GetWeather"
    assert tool.description == "Fetch the weather for a city"
    assert graph.qualified_name == "WeatherTool.GetWeather"
    assert graph.calls[0].qualified_name == "WeatherTool.Fetch"
    assert graph.calls[0].calls[0].qualified_name == "Cache.Get"


def test_unsupported_language_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Rust"):
        detect_tools_with_call_graphs(tmp_path, "Rust")
