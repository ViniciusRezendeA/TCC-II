from __future__ import annotations

from pathlib import Path

import pytest

from mcp_pipeline.extraction.tool_detector import (
    LANGUAGE_ADAPTERS,
    _effective_language,
    detect_tools_with_call_graphs,
)


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
    assert weather_tool.loc == 3  # def/docstring/return -- decorator line excluded
    assert weather_tool.call_graph_depth == 3  # get_weather -> _fetch -> get_cached

    other_tool_record, other_graph = by_name["other_tool"]
    assert other_tool_record.description == "Second tool"
    assert other_graph.calls == []  # `return x` has no call expressions
    assert other_tool_record.loc == 2  # def/return -- decorator line excluded
    assert other_tool_record.call_graph_depth == 1  # no calls at all


def test_python_lowlevel_list_tools_inline_static_list(tmp_path):
    """Mirrors modelcontextprotocol/servers' fetch server: the official
    low-level @server.list_tools() decorator, alternative to @mcp.tool()."""
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        '''
@server.list_tools()
async def list_tools():
    return [
        Tool(name="fetch", description="Fetches a URL from the internet."),
    ]
''',
    )

    results = detect_tools_with_call_graphs(repo, "Python")

    assert len(results) == 1
    tool, graph = results[0]
    assert tool.name == "fetch"
    assert tool.description == "Fetches a URL from the internet."
    assert tool.sdk_pattern == "python.list_tools_lowlevel"
    assert graph.qualified_name == "list_tools"


def test_python_lowlevel_list_tools_resolves_cross_file_constant(tmp_path):
    """Mirrors blazickjp/arxiv-mcp-server: list_tools() returns a list of
    bare names, each a module-level Tool(...) constant defined in a
    different file, reachable only via the import graph."""
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        """
from .tools.search import search_tool


@server.list_tools()
async def list_tools():
    return [search_tool]
""",
    )
    _write(
        repo / "tools" / "search.py",
        '''
search_tool = Tool(name="search_papers", description="Search for papers on arXiv.")
''',
    )
    _write(repo / "tools" / "__init__.py", "")

    results = detect_tools_with_call_graphs(repo, "Python")

    assert len(results) == 1
    tool, _graph = results[0]
    assert tool.name == "search_papers"
    assert tool.description == "Search for papers on arXiv."


def test_python_lowlevel_list_tools_dynamic_registry_yields_zero_tools(tmp_path):
    """Mirrors vitali87/code-graph-rag: a comprehension over a runtime dict
    populated elsewhere -- must yield zero, not guess."""
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        """
@server.list_tools()
async def list_tools():
    schemas = get_tool_schemas()
    return [Tool(name=schema.name, description=schema.description) for schema in schemas]
""",
    )

    results = detect_tools_with_call_graphs(repo, "Python")

    assert results == []


def test_typescript_lowlevel_set_request_handler_inline_static_array(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.ts",
        """
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{ name: "ping", description: "Ping the server." }],
}));
""",
    )

    results = detect_tools_with_call_graphs(repo, "TypeScript")

    assert len(results) == 1
    tool, _graph = results[0]
    assert tool.name == "ping"
    assert tool.description == "Ping the server."
    assert tool.sdk_pattern == "typescript.set_request_handler_lowlevel"


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
    # handleGetWeather's own 3-line body (lines 9-11 of the fixture), NOT the
    # 4-line server.registerTool(...) call site (lines 4-7) that
    # tool.source_location points to -- this is what pins loc to start_def,
    # not tool.source_location.
    assert tool.loc == 3
    assert tool.call_graph_depth == 2  # handleGetWeather -> getCached (leaf)


def test_javascript_detects_fastmcp_npm_addtool_alongside_registertool(tmp_path):
    """Confirms combine_tool_detectors() wiring: a repo using both the
    high-level .registerTool() pattern and the third-party fastmcp npm
    .addTool() pattern gets tools from both, not just one."""
    repo = tmp_path / "src"
    _write(
        repo / "server.js",
        '''
server.registerTool("get_weather", {
  description: "Fetch the weather",
  inputSchema: schema,
}, async (args) => {
  return fetchWeather(args.city);
});

server.addTool({
  name: "ping",
  description: "Ping the server",
  execute: async () => {
    return "pong";
  },
});
''',
    )

    results = detect_tools_with_call_graphs(repo, "JavaScript")

    names_and_patterns = {tool.name: tool.sdk_pattern for tool, _graph in results}
    assert names_and_patterns == {
        "get_weather": "javascript.registerTool",
        "ping": "javascript.fastmcp_npm_addtool",
    }


def test_effective_language_switches_js_to_ts_when_source_is_majority_typescript(tmp_path):
    """Real case (firecrawl/firecrawl-mcp-server and others): GitHub
    declares primary_language JavaScript, but the real source is majority
    TypeScript (likely a committed dist/ build skewing GitHub's byte-count
    classifier) -- the pipeline should scan the real language, not the
    declared one, when they clearly disagree."""
    repo = tmp_path / "src"
    _write(repo / "index.ts", "export function f() { return 1; }\n")
    _write(repo / "utils.ts", "export function g() { return 2; }\n")
    _write(repo / "legacy.js", "function h() { return 3; }\n")

    assert _effective_language(repo, "JavaScript") == "TypeScript"


def test_effective_language_keeps_js_when_js_is_actually_dominant(tmp_path):
    repo = tmp_path / "src"
    _write(repo / "index.js", "function f() { return 1; }\n")
    _write(repo / "utils.js", "function g() { return 2; }\n")
    _write(repo / "types.ts", "export type X = number;\n")

    assert _effective_language(repo, "JavaScript") == "JavaScript"


def test_effective_language_leaves_other_languages_untouched(tmp_path):
    repo = tmp_path / "src"
    assert _effective_language(repo, "Python") == "Python"
    assert _effective_language(repo, "TypeScript") == "TypeScript"


def test_declared_javascript_repo_detected_as_typescript_end_to_end(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.ts",
        """
server.addTool({
  name: "ping",
  description: "Ping the server.",
  execute: async () => {
    return "pong";
  },
});
""",
    )
    _write(repo / "helpers.ts", "export function noop(): void {}\n")
    _write(repo / "helpers2.ts", "export function noop2(): void {}\n")

    results = detect_tools_with_call_graphs(repo, "JavaScript")

    assert len(results) == 1
    tool, _graph = results[0]
    assert tool.name == "ping"
    assert tool.sdk_pattern == "typescript.fastmcp_npm_addtool"


def test_java_and_csharp_adapters_unaffected_by_new_optional_fields():
    """Regression guard: the Etapa-2-extension additions (extract_values,
    detect_tools_with_context) are optional fields defaulting to None, and
    Java/C# were deliberately left untouched -- this documents that intent
    explicitly rather than relying on it being implicit."""
    assert LANGUAGE_ADAPTERS["Java"].extract_values is None
    assert LANGUAGE_ADAPTERS["Java"].detect_tools_with_context is None
    assert LANGUAGE_ADAPTERS["C#"].extract_values is None
    assert LANGUAGE_ADAPTERS["C#"].detect_tools_with_context is None


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
    with pytest.raises(ValueError, match="Swift"):
        detect_tools_with_call_graphs(tmp_path, "Swift")
