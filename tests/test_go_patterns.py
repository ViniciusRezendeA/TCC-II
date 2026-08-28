from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.go_patterns import (
    detect_go_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Go")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_tool_with_method_value_handler():
    root, source_bytes = parse(
        """
package main

func (s *mcpServer) registerTools(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{Name: "wails_project_inspect", Description: "Inspect a project."}, s.inspect)
}

func (s *mcpServer) inspect(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return nil, nil
}
"""
    )
    tools = detect_go_tools(root, source_bytes, "server.go")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "wails_project_inspect"
    assert tool.description == "Inspect a project."
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "go.mcp_addtool"
    assert tool.qualified_name == "mcpServer.inspect"


def test_detects_tool_with_bare_function_handler():
    root, source_bytes = parse(
        """
package main

func register(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{Name: "ping", Description: "Ping."}, handlePing)
}

func handlePing(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return nil, nil
}
"""
    )
    tools = detect_go_tools(root, source_bytes, "server.go")
    assert len(tools) == 1
    assert tools[0].qualified_name == "handlePing"


def test_non_literal_name_is_ignored():
    root, source_bytes = parse(
        """
package main

func register(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{Name: listToolName, Description: "List things."}, h.list)
}
"""
    )
    tools = detect_go_tools(root, source_bytes, "server.go")
    assert tools == []


def test_extract_definitions_distinguishes_receiver_types():
    root, source_bytes = parse(
        """
package main

func (s *mcpServer) inspect() {
	helper()
}

func (h *Handler) list() {}

func helper() {}
"""
    )
    defs = extract_definitions(root, source_bytes, "server.go")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {"mcpServer.inspect", "Handler.list", "helper"}


def test_extract_imports_binds_alias_and_explicit_name():
    root, source_bytes = parse(
        """
package main

import (
	"fmt"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)
"""
    )
    imports = extract_imports(root, source_bytes)
    assert imports["fmt"].module == "fmt"
    assert imports["mcpsdk"].module == "mcp"


def test_extract_calls_distinguishes_bare_and_selector_receivers():
    root, source_bytes = parse(
        """
package main

func (s *mcpServer) inspect() {
	helper()
	s.other()
	mcp.AddTool(nil, nil, nil)
}
"""
    )
    defs = extract_definitions(root, source_bytes, "server.go")
    body = next(d for d in defs if d.bare_name == "inspect").body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["helper"].receiver is None
    assert by_name["other"].receiver == "s"
    assert by_name["AddTool"].receiver == "mcp"
