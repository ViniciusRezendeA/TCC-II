from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.kotlin_patterns import (
    detect_kotlin_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Kotlin")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_tool_with_trailing_lambda_handler():
    root, source_bytes = parse(
        """
class Server {
    fun registerTools() {
        addTool(
            name = "get_alerts",
            description = "Get weather alerts for a state.",
        ) { request ->
            fetch(request)
        }
    }
}
"""
    )
    tools = detect_kotlin_tools(root, source_bytes, "Server.kt")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "get_alerts"
    assert tool.description == "Get weather alerts for a state."
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "kotlin.kotlinsdk_addtool"
    assert tool.qualified_name == "<inline_tool_handler>:Server.kt:get_alerts"

    defs = extract_definitions(root, source_bytes, "Server.kt")
    synthetic = next(d for d in defs if d.qualified_name == tool.qualified_name)
    assert synthetic.body_node is not None


def test_detects_tool_with_named_handler_argument():
    root, source_bytes = parse(
        """
fun register() {
    addTool(
        name = "ping",
        description = "Ping.",
        handler = { request ->
            respond(request)
        },
    )
}
"""
    )
    tools = detect_kotlin_tools(root, source_bytes, "server.kt")
    assert len(tools) == 1
    assert tools[0].name == "ping"


def test_non_literal_name_is_ignored():
    root, source_bytes = parse(
        """
fun register() {
    addTool(name = descriptor.id, description = descriptor.description) { request ->
        respond(request)
    }
}
"""
    )
    tools = detect_kotlin_tools(root, source_bytes, "server.kt")
    assert tools == []


def test_interpolated_description_falls_back_to_raw_text():
    root, source_bytes = parse(
        """
fun register() {
    addTool(name = "greet", description = "Greet using $prefix") { request ->
        respond(request)
    }
}
"""
    )
    tools = detect_kotlin_tools(root, source_bytes, "server.kt")
    assert tools[0].description_is_literal is False
    assert "$prefix" in tools[0].description


def test_extract_definitions_distinguishes_classes_and_top_level():
    root, source_bytes = parse(
        """
class Server {
    fun handle() {
        helper()
    }
}

fun helper() {}
"""
    )
    defs = extract_definitions(root, source_bytes, "Server.kt")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {"Server.handle", "helper"}


def test_extract_imports_returns_empty_mapping():
    root, source_bytes = parse(
        """
import io.modelcontextprotocol.kotlin.sdk.server.Server
"""
    )
    assert extract_imports(root, source_bytes) == {}


def test_extract_calls_distinguishes_bare_this_and_qualified_receivers():
    root, source_bytes = parse(
        """
class Server {
    fun handle() {
        fetch(1)
        this.otherMethod()
        Cache.get(1)
    }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "Server.kt")
    body = next(d for d in defs if d.bare_name == "handle").body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["fetch"].receiver is None
    assert by_name["otherMethod"].receiver == "this"
    assert by_name["get"].receiver == "Cache"
