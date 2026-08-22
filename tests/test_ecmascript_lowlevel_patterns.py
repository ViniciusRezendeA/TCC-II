from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.definition_index import DefinitionIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.typescript_patterns import (
    detect_lowlevel_set_request_handler,
    extract_definitions,
    extract_imports,
    extract_values,
)
from mcp_pipeline.extraction.value_index import ValueIndex

SPEC = spec_for("TypeScript")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def detect_single_file(source: str, filename: str = "server.ts"):
    root, source_bytes = parse(source)
    values = ValueIndex.build(extract_values(root, source_bytes, filename))
    definitions = DefinitionIndex.build(extract_definitions(root, source_bytes, filename))
    return detect_lowlevel_set_request_handler(
        root, source_bytes, filename, definitions, values, {}, {filename: source_bytes}
    )


def test_detects_inline_static_array():
    tools = detect_single_file(
        """
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    { name: "gmail_send_email", description: "Sends a new email." },
    { name: "gmail_draft_email", description: "Draft a new email" },
  ],
}));
"""
    )
    assert {t.name for t in tools} == {"gmail_send_email", "gmail_draft_email"}
    assert all(t.sdk_pattern == "typescript.set_request_handler_lowlevel" for t in tools)
    send = next(t for t in tools if t.name == "gmail_send_email")
    assert send.description == "Sends a new email."
    assert send.description_is_literal is True


def test_resolves_same_file_const_with_filter_chain():
    tools = detect_single_file(
        """
const baseTools = [
  { name: "kubectl_get", description: "Get resources." },
  { name: "kubectl_describe", description: "Describe a resource." },
];

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: baseTools.filter((tool) => isAllowed(tool.name)) };
});
"""
    )
    assert {t.name for t in tools} == {"kubectl_get", "kubectl_describe"}


def test_shorthand_return_tools_form():
    """Real case (Flux159/mcp-server-kubernetes): `return { tools }` is
    shorthand for `{ tools: tools }` -- a pair-only object-field lookup
    would miss this entirely."""
    tools = detect_single_file(
        """
const tools = [{ name: "ping", description: "Ping." }];

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools };
});
"""
    )
    assert [t.name for t in tools] == ["ping"]


def test_receiver_is_not_constrained_to_bare_server():
    """Real shapes: target.server.setRequestHandler(...) and
    this.server.setRequestHandler(...), not just server.setRequestHandler(...)."""
    tools = detect_single_file(
        """
target.server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [{ name: "ping", description: "Ping." }],
}));
"""
    )
    assert [t.name for t in tools] == ["ping"]


def test_map_chain_preserves_name_but_marks_description_non_literal():
    """Real case (chrisryugj/korean-law-mcp): a .map() that re-derives
    description via template-literal concatenation must still preserve the
    tool's identity (name), while description falls back to non-literal
    raw text -- not skipped entirely, matching this project's established
    non-literal-description convention."""
    tools = detect_single_file(
        """
const allTools = [
  { name: "search_law", description: "Search laws by keyword." },
];

const exposedTools = allTools.filter((t) => V3_EXPOSED.has(t.name));

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: exposedTools.map((tool) => ({
      name: tool.name,
      description: `${SERVICE_NAME} — ${tool.description}`,
    })),
  };
});
"""
    )
    assert len(tools) == 1
    assert tools[0].name == "search_law"
    assert tools[0].description_is_literal is False
    assert "SERVICE_NAME" in tools[0].description


def test_dynamic_map_lookup_yields_zero_tools_not_guessed():
    """Real case (apify/apify-mcp-server, samanhappy/mcphub): tools
    populated from a runtime Map/registry -- must yield zero, not guess."""
    tools = detect_single_file(
        """
server.setRequestHandler(ListToolsRequestSchema, async () => {
  const tools = Array.from(this.host.tools.values()).map((tool) => getPublicFields(tool));
  return { tools };
});
"""
    )
    assert tools == []


def test_middleware_wrapping_result_yields_zero_tools():
    """Real named false-positive-risk case (mksglu/context-mode): a
    setRequestHandler call that's middleware around the SDK's own dispatch
    (empty array literal, or sanitizing an already-built result), not a
    real tool-definition site."""
    tools = detect_single_file(
        """
target.server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: [] }));
"""
    )
    assert tools == []


def test_cross_file_import_linked_array_elements():
    tools_ts_source = """
export const pingSchema = { name: "ping", description: "Verify liveness." };
"""
    server_ts_source = """
import { pingSchema } from "./tools/ping";

const allTools = [pingSchema];

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: allTools }));
"""
    tools_root, tools_source_bytes = parse(tools_ts_source)
    server_root, server_source_bytes = parse(server_ts_source)

    values = ValueIndex.build(
        extract_values(tools_root, tools_source_bytes, "tools/ping.ts")
        + extract_values(server_root, server_source_bytes, "server.ts")
    )
    definitions = DefinitionIndex.build(
        extract_definitions(tools_root, tools_source_bytes, "tools/ping.ts")
        + extract_definitions(server_root, server_source_bytes, "server.ts")
    )
    imports_by_file = {"server.ts": extract_imports(server_root, server_source_bytes)}
    source_bytes_by_file = {"tools/ping.ts": tools_source_bytes, "server.ts": server_source_bytes}

    tools = detect_lowlevel_set_request_handler(
        server_root, server_source_bytes, "server.ts", definitions, values, imports_by_file, source_bytes_by_file
    )

    assert [t.name for t in tools] == ["ping"]
    assert tools[0].description == "Verify liveness."


def test_extract_values_covers_export_const_and_excludes_function_local(tmp_path):
    root, source_bytes = parse(
        """
const allTools = [1];
export const MAPS_TOOLS = [2];
function f() {
  const local = 3;
  return local;
}
"""
    )
    values = extract_values(root, source_bytes, "server.ts")
    names = {v.bare_name for v in values}
    assert names == {"allTools", "MAPS_TOOLS"}
