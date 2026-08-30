from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.definition_index import DefinitionIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.go_patterns import (
    detect_go_tools_with_context,
    extract_calls,
    extract_definitions,
    extract_imports,
    extract_values,
)
from mcp_pipeline.extraction.value_index import ValueIndex

SPEC = spec_for("Go")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def detect_tools(root, source_bytes, rel_path="server.go", values=None, imports_by_file=None):
    """Test helper: detect_go_tools_with_context with an empty context by
    default (no cross-reference needed for most patterns tested here)."""
    return detect_go_tools_with_context(
        root, source_bytes, rel_path,
        DefinitionIndex.build([]),
        values if values is not None else ValueIndex.build([]),
        imports_by_file if imports_by_file is not None else {},
        {rel_path: source_bytes},
    )


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
    tools = detect_tools(root, source_bytes)
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
    tools = detect_tools(root, source_bytes)
    assert len(tools) == 1
    assert tools[0].qualified_name == "handlePing"


def test_unresolvable_name_is_ignored():
    root, source_bytes = parse(
        """
package main

func register(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{Name: listToolName, Description: "List things."}, h.list)
}
"""
    )
    # `listToolName` isn't declared anywhere in this snippet -- resolve_value
    # finds nothing, so the call is correctly skipped rather than guessed.
    tools = detect_tools(root, source_bytes)
    assert tools == []


def test_resolves_name_from_same_file_package_constant():
    root, source_bytes = parse(
        """
package main

const (
	listToolName   = "list_agent_instances"
	invokeToolName = "invoke_agent_instance"
)

func register(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{Name: listToolName, Description: "List instances."}, h.list)
	mcp.AddTool(server, &mcp.Tool{Name: invokeToolName, Description: "Invoke an instance."}, h.invoke)
}
"""
    )
    values = ValueIndex.build(extract_values(root, source_bytes, "server.go"))
    tools = detect_tools(root, source_bytes, values=values)
    assert len(tools) == 2
    by_name = {t.name: t for t in tools}
    assert by_name["list_agent_instances"].description_is_literal is True
    assert by_name["list_agent_instances"].description == "List instances."
    assert by_name["invoke_agent_instance"].description == "Invoke an instance."


def test_resolved_constant_that_is_itself_not_a_string_is_ignored():
    root, source_bytes = parse(
        """
package main

const maxRetries = 3

func register(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{Name: maxRetries, Description: "Bogus."}, h.list)
}
"""
    )
    values = ValueIndex.build(extract_values(root, source_bytes, "server.go"))
    tools = detect_tools(root, source_bytes, values=values)
    assert tools == []


def test_extract_values_reads_single_and_grouped_const_forms():
    root, source_bytes = parse(
        """
package main

const singleConst = "single_value"

const (
	listToolName   = "list_agent_instances"
	invokeToolName = "invoke_agent_instance"
)

func f() {
	const notTopLevel = "unreachable"
}
"""
    )
    values = extract_values(root, source_bytes, "server.go")
    names = {v.bare_name for v in values}
    assert names == {"singleConst", "listToolName", "invokeToolName"}


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
