from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.csharp_patterns import (
    detect_csharp_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("C#")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_combined_attribute_form():
    root, source_bytes = parse(
        """
public sealed class WeatherTool
{
    [McpServerTool, Description("Fetch the weather for a city")]
    public static string GetWeather(string city)
    {
        return city;
    }
}
"""
    )
    tools = detect_csharp_tools(root, source_bytes, "WeatherTool.cs")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "GetWeather"
    assert tool.description == "Fetch the weather for a city"
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "csharp.mcpservertool_attribute"
    assert tool.qualified_name == "WeatherTool.GetWeather"


def test_detects_stacked_attribute_form():
    root, source_bytes = parse(
        """
public sealed class WeatherTool
{
    [McpServerTool]
    [Description("Fetch the weather")]
    public static string GetWeather(string city)
    {
        return city;
    }
}
"""
    )
    tools = detect_csharp_tools(root, source_bytes, "WeatherTool.cs")
    assert len(tools) == 1
    assert tools[0].description == "Fetch the weather"


def test_mcpservertool_without_description_yields_empty_string():
    root, source_bytes = parse(
        """
public sealed class WeatherTool
{
    [McpServerTool]
    public static string GetWeather(string city)
    {
        return city;
    }
}
"""
    )
    tools = detect_csharp_tools(root, source_bytes, "WeatherTool.cs")
    assert len(tools) == 1
    assert tools[0].description == ""


def test_methods_without_mcpservertool_attribute_are_ignored():
    root, source_bytes = parse(
        """
public sealed class WeatherTool
{
    [Description("Not a tool")]
    public static string Helper(string city)
    {
        return city;
    }
}
"""
    )
    tools = detect_csharp_tools(root, source_bytes, "WeatherTool.cs")
    assert tools == []


def test_extract_definitions_distinguishes_classes():
    root, source_bytes = parse(
        """
public class WeatherTool
{
    public string GetWeather(string city) { return Fetch(city); }
    private string Fetch(string city) { return city; }
}
public class CacheUtils
{
    public static string Get(string key) { return null; }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "WeatherTool.cs")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {"WeatherTool.GetWeather", "WeatherTool.Fetch", "CacheUtils.Get"}


def test_extract_imports_binds_last_segment_as_alias():
    root, source_bytes = parse(
        """
using ModelContextProtocol.Server;
using System.ComponentModel;

public class WeatherTool {}
"""
    )
    imports = extract_imports(root, source_bytes)
    assert imports["Server"].module == "ModelContextProtocol.Server"
    assert imports["ComponentModel"].module == "System.ComponentModel"


def test_extract_calls_distinguishes_bare_and_qualified_receivers():
    root, source_bytes = parse(
        """
public class WeatherTool
{
    public string GetWeather(string city)
    {
        Fetch(city);
        Cache.Get(city);
        return city;
    }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "WeatherTool.cs")
    body = defs[0].body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["Fetch"].receiver is None
    assert by_name["Get"].receiver == "Cache"
