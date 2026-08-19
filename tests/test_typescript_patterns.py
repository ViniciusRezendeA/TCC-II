from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    detect_mcp_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
    synthetic_handler_name,
)

SPEC = spec_for("TypeScript")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_v1_tool_form_with_inline_handler():
    root, source_bytes = parse(
        '''
server.tool("get_weather", "Fetch the weather", schema, async (args) => {
  return fetch(args.city);
});
'''
    )
    tools = detect_mcp_tools(root, source_bytes, "server.ts")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "get_weather"
    assert tool.description == "Fetch the weather"
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "typescript.tool_v1"
    assert tool.qualified_name == synthetic_handler_name("server.ts", "get_weather")


def test_detects_registertool_form_with_description_from_config_object():
    root, source_bytes = parse(
        '''
server.registerTool("get_weather", {
  description: "Fetch the weather",
  inputSchema: schema,
}, async (args) => {
  return fetch(args.city);
});
'''
    )
    tools = detect_mcp_tools(root, source_bytes, "server.ts")
    assert len(tools) == 1
    assert tools[0].sdk_pattern == "typescript.registerTool"
    assert tools[0].description == "Fetch the weather"


def test_handler_referenced_by_name_resolves_to_that_identifier():
    root, source_bytes = parse('server.tool("get_weather", "desc", schema, myHandler);')
    tools = detect_mcp_tools(root, source_bytes, "server.ts")
    assert tools[0].qualified_name == "myHandler"


def test_non_tool_method_calls_are_ignored():
    root, source_bytes = parse('app.get("/weather", handler);')
    tools = detect_mcp_tools(root, source_bytes, "server.ts")
    assert tools == []


def test_non_literal_tool_name_is_skipped_not_fabricated():
    """Real-world case (chrome-devtools-mcp): tools built data-driven via a
    `for (const tool of tools) registerTool(tool)` loop, so the name
    argument is `tool.name`, not a string literal. This call site
    represents an unknown number of real tools defined elsewhere — a single
    fake "tool.name" entry would be worse than reporting none.
    """
    root, source_bytes = parse(
        """
function registerTool(tool) {
  server.registerTool(
    tool.name,
    {description: tool.description, inputSchema: tool.schema},
    async (params) => tool.handler(params),
  );
}
"""
    )
    tools = detect_mcp_tools(root, source_bytes, "server.ts")
    assert tools == []


def test_extract_definitions_covers_function_declaration_arrow_const_and_class_method():
    root, source_bytes = parse(
        """
function helper(x) { return x; }
const helper2 = (x) => { return x * 2; };
class WeatherServer {
  getWeather(city) { return this.fetch(city); }
  fetch(city) { return city; }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "server.ts")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {
        "helper", "helper2", "WeatherServer.getWeather", "WeatherServer.fetch",
    }
    method = next(d for d in defs if d.qualified_name == "WeatherServer.getWeather")
    assert method.class_name == "WeatherServer"


def test_extract_definitions_registers_synthetic_entry_for_inline_handler():
    root, source_bytes = parse(
        '''
server.tool("get_weather", "desc", schema, async (args) => {
  return fetch(args.city);
});
'''
    )
    defs = extract_definitions(root, source_bytes, "server.ts")
    assert any(d.qualified_name == synthetic_handler_name("server.ts", "get_weather") for d in defs)


def test_extract_imports_handles_named_default_aliased_and_namespace_forms():
    root, source_bytes = parse(
        """
import { fetchData } from "./utils";
import { fetchData as fd } from "./utils2";
import * as cache from "./cache";
import defaultExport from "./thing";
"""
    )
    imports = extract_imports(root, source_bytes)
    assert imports["fetchData"].module == "./utils"
    assert imports["fd"].original_name == "fetchData"
    assert imports["fd"].module == "./utils2"
    assert imports["cache"].module == "./cache"
    assert imports["cache"].original_name == "*"
    assert imports["defaultExport"].module == "./thing"
    assert imports["defaultExport"].original_name == "default"


def test_extract_calls_distinguishes_bare_this_and_qualified_receivers():
    root, source_bytes = parse(
        """
function bodyHolder() {
  this.fetch(city);
  externalLib.request(url);
  bareCall();
}
"""
    )
    defs = extract_definitions(root, source_bytes, "server.ts")
    body = defs[0].body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["fetch"].receiver == "this"
    assert by_name["request"].receiver == "externalLib"
    assert by_name["bareCall"].receiver is None
