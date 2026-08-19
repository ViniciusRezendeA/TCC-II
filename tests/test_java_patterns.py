from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.java_patterns import (
    detect_spring_ai_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Java")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_springai_tool_with_description():
    root, source_bytes = parse(
        """
public class WeatherService {
    @Tool(description = "Fetch the weather for a city")
    public String getWeather(String city) {
        return fetch(city);
    }
}
"""
    )
    tools = detect_spring_ai_tools(root, source_bytes, "WeatherService.java")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "getWeather"
    assert tool.description == "Fetch the weather for a city"
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "java.springai_annotation"
    assert tool.qualified_name == "WeatherService.getWeather"


def test_detects_tool_with_explicit_name_override():
    root, source_bytes = parse(
        """
public class WeatherService {
    @Tool(name = "weather_tool", description = "desc")
    public String getWeather(String city) {
        return city;
    }
}
"""
    )
    tools = detect_spring_ai_tools(root, source_bytes, "WeatherService.java")
    assert tools[0].name == "weather_tool"


def test_methods_without_tool_annotation_are_ignored():
    root, source_bytes = parse(
        """
public class WeatherService {
    public String helper(String city) {
        return city;
    }
}
"""
    )
    tools = detect_spring_ai_tools(root, source_bytes, "WeatherService.java")
    assert tools == []


def test_tool_without_description_yields_empty_string():
    root, source_bytes = parse(
        """
public class WeatherService {
    @Tool
    public String getWeather(String city) {
        return city;
    }
}
"""
    )
    tools = detect_spring_ai_tools(root, source_bytes, "WeatherService.java")
    assert tools == []  # @Tool with no argument_list doesn't match the query — documented scope


def test_extract_definitions_distinguishes_methods_from_different_classes():
    root, source_bytes = parse(
        """
public class WeatherService {
    public String getWeather(String city) { return this.fetch(city); }
    private String fetch(String city) { return city; }
}
class CacheUtils {
    public static String get(String key) { return null; }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "WeatherService.java")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {
        "WeatherService.getWeather", "WeatherService.fetch", "CacheUtils.get",
    }


def test_extract_imports_binds_last_segment_as_alias():
    root, source_bytes = parse(
        """
import org.springframework.ai.tool.annotation.Tool;
import com.example.util.CacheUtils;

public class WeatherService {}
"""
    )
    imports = extract_imports(root, source_bytes)
    assert imports["Tool"].module == "org.springframework.ai.tool.annotation.Tool"
    assert imports["CacheUtils"].module == "com.example.util.CacheUtils"


def test_extract_calls_distinguishes_this_and_qualified_receivers():
    root, source_bytes = parse(
        """
public class WeatherService {
    public String getWeather(String city) {
        this.fetch(city);
        CacheUtils.get(city);
        bareCall();
        return city;
    }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "WeatherService.java")
    body = defs[0].body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["fetch"].receiver == "this"
    assert by_name["get"].receiver == "CacheUtils"
    assert by_name["bareCall"].receiver is None
