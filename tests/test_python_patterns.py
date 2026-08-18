from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.python_patterns import (
    detect_fastmcp_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Python")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_module_level_tool_with_docstring_fallback():
    root, source_bytes = parse(
        '''
@mcp.tool()
def get_weather(city: str) -> str:
    """Fetch the current weather for a city."""
    return fetch(city)
'''
    )
    tools = detect_fastmcp_tools(root, source_bytes, "server.py")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "get_weather"
    assert tool.description == "Fetch the current weather for a city."
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "python.fastmcp_decorator"
    assert tool.qualified_name == "get_weather"


def test_detects_tool_with_explicit_description_and_name():
    root, source_bytes = parse(
        '''
@mcp.tool(description="Explicit description", name="weather_tool")
def get_weather(city: str) -> str:
    return fetch(city)
'''
    )
    tools = detect_fastmcp_tools(root, source_bytes, "server.py")
    assert len(tools) == 1
    assert tools[0].name == "weather_tool"
    assert tools[0].description == "Explicit description"
    assert tools[0].description_is_literal is True


def test_non_literal_description_is_captured_but_flagged():
    root, source_bytes = parse(
        '''
DESC = "built elsewhere"

@mcp.tool(description=DESC)
def get_weather(city: str) -> str:
    return fetch(city)
'''
    )
    tools = detect_fastmcp_tools(root, source_bytes, "server.py")
    assert tools[0].description == "DESC"
    assert tools[0].description_is_literal is False


def test_tool_missing_description_and_docstring_yields_empty_string():
    root, source_bytes = parse(
        """
@mcp.tool()
def get_weather(city: str) -> str:
    return fetch(city)
"""
    )
    tools = detect_fastmcp_tools(root, source_bytes, "server.py")
    assert tools[0].description == ""


def test_class_method_tool_gets_qualified_name():
    root, source_bytes = parse(
        '''
class WeatherServer:
    @mcp.tool()
    def get_weather(self, city: str) -> str:
        """Docstring."""
        return self._fetch(city)
'''
    )
    tools = detect_fastmcp_tools(root, source_bytes, "server.py")
    assert tools[0].qualified_name == "WeatherServer.get_weather"


def test_non_tool_decorators_are_ignored():
    root, source_bytes = parse(
        """
@app.route("/weather")
def get_weather(city: str) -> str:
    return fetch(city)
"""
    )
    tools = detect_fastmcp_tools(root, source_bytes, "server.py")
    assert tools == []


def test_extract_definitions_distinguishes_methods_from_module_functions():
    root, source_bytes = parse(
        """
class WeatherServer:
    def get_weather(self, city):
        return self._fetch(city)

    def _fetch(self, city):
        return city

def helper():
    pass
"""
    )
    defs = extract_definitions(root, source_bytes, "server.py")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {"WeatherServer.get_weather", "WeatherServer._fetch", "helper"}
    method = next(d for d in defs if d.qualified_name == "WeatherServer.get_weather")
    assert method.class_name == "WeatherServer"
    func = next(d for d in defs if d.qualified_name == "helper")
    assert func.class_name is None


def test_extract_imports_handles_plain_aliased_and_relative_forms():
    root, source_bytes = parse(
        """
import requests
import cache_utils as cu
from . import helpers
from .utils import fetch_data
from .utils import fetch_data as fd
from mypkg.sub import thing
"""
    )
    imports = extract_imports(root, source_bytes)
    assert imports["requests"].module == "requests"
    assert imports["cu"].module == "cache_utils"
    assert imports["helpers"].module == "."
    assert imports["fetch_data"].module == ".utils"
    assert imports["fd"].original_name == "fetch_data"
    assert imports["thing"].module == "mypkg.sub"


def test_extract_imports_binds_only_top_level_name_for_dotted_plain_import():
    root, source_bytes = parse("import mypkg.sub.deep\n")
    imports = extract_imports(root, source_bytes)
    assert "mypkg" in imports
    assert "deep" not in imports
    assert imports["mypkg"].module == "mypkg.sub.deep"


def test_extract_calls_distinguishes_bare_self_and_qualified_receivers():
    root, source_bytes = parse(
        """
def body_holder():
    self._fetch(city)
    requests.get(url)
    bare_call()
"""
    )
    defs = extract_definitions(root, source_bytes, "server.py")
    body = defs[0].body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["_fetch"].receiver == "self"
    assert by_name["get"].receiver == "requests"
    assert by_name["bare_call"].receiver is None
