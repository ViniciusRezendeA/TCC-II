from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.dart_patterns import (
    detect_dart_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Dart")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_tool_with_identifier_handler():
    root, source_bytes = parse(
        """
final class Server extends MCPServer with ToolsSupport {
  void _registerTools() {
    registerTool(
      Tool(name: 'create', description: 'Create a project.'),
      _handleCreate,
    );
  }

  Future<CallToolResult> _handleCreate(CallToolRequest request) async {
    return doThing(request);
  }
}
"""
    )
    tools = detect_dart_tools(root, source_bytes, "server.dart")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "create"
    assert tool.description == "Create a project."
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "dart.dart_mcp_registertool"
    assert tool.qualified_name == "Server._handleCreate"


def test_detects_tool_with_inline_closure_handler():
    root, source_bytes = parse(
        """
void main() {
  registerTool(
    Tool(name: 'transcribe', description: 'Transcribe audio.'),
    (call) async {
      return doWork(call);
    },
  );
}
"""
    )
    tools = detect_dart_tools(root, source_bytes, "mcp.dart")
    assert len(tools) == 1
    assert tools[0].name == "transcribe"
    assert tools[0].qualified_name == "<inline_tool_handler>:mcp.dart:transcribe"

    defs = extract_definitions(root, source_bytes, "mcp.dart")
    synthetic = next(d for d in defs if d.qualified_name == tools[0].qualified_name)
    assert synthetic.body_node is not None


def test_variable_referenced_tool_name_is_ignored():
    root, source_bytes = parse(
        """
void register(String name, String desc, Function run) {
  registerTool(Tool(name: name, description: desc), run);
}
"""
    )
    tools = detect_dart_tools(root, source_bytes, "server.dart")
    assert tools == []


def test_triple_quoted_multiline_description():
    root, source_bytes = parse(
        """
void main() {
  registerTool(
    Tool(
      name: 'create',
      description: \'\'\'
Create a very good Dart project.
Multiple lines here.
\'\'\',
    ),
    _handleCreate,
  );
}
"""
    )
    tools = detect_dart_tools(root, source_bytes, "server.dart")
    assert len(tools) == 1
    assert "Create a very good Dart project." in tools[0].description
    assert tools[0].description_is_literal is True


def test_interpolated_description_falls_back_to_raw_text():
    root, source_bytes = parse(
        """
void main() {
  registerTool(
    Tool(name: 'create', description: "Create a $type project."),
    _handleCreate,
  );
}
"""
    )
    tools = detect_dart_tools(root, source_bytes, "server.dart")
    assert tools[0].description_is_literal is False
    assert "$type" in tools[0].description


def test_extract_definitions_distinguishes_classes_and_top_level():
    root, source_bytes = parse(
        """
class Server {
  void handle() {
    helper();
  }
}

void helper() {}
"""
    )
    defs = extract_definitions(root, source_bytes, "server.dart")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {"Server.handle", "helper"}


def test_extract_imports_returns_empty_mapping():
    root, source_bytes = parse(
        """
import 'package:dart_mcp/server.dart';
import 'package:meta/meta.dart' as meta;
"""
    )
    assert extract_imports(root, source_bytes) == {}


def test_extract_calls_distinguishes_bare_and_member_receivers():
    root, source_bytes = parse(
        """
class Server {
  void handle() {
    fetch(1);
    this.otherMethod();
    Cache.get(1);
  }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "server.dart")
    body = next(d for d in defs if d.bare_name == "handle").body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["fetch"].receiver is None
    assert by_name["otherMethod"].receiver == "this"
    assert by_name["get"].receiver == "Cache"
