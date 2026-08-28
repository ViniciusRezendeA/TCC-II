from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import CallSite
from mcp_pipeline.extraction.definition_index import FunctionDef
from mcp_pipeline.extraction.import_index import ImportedName, ImportIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.models import SourceLocation, ToolRecord
from mcp_pipeline.extraction.parser_utils import (
    line_range,
    node_text,
    run_query,
    string_literal_value,
)

GO_LANGUAGE = spec_for("Go").ts_language


def _go_string_value(node: Node, source_bytes: bytes) -> str | None:
    """Go has no string interpolation (always fmt.Sprintf-style), so unlike
    Ruby/Dart/JS there's no interpolated-string trap to guard against here —
    just two literal shapes with different content-child node types."""
    if node.type == "interpreted_string_literal":
        return string_literal_value(
            node, source_bytes, content_node_type="interpreted_string_literal_content", node_type="interpreted_string_literal"
        )
    if node.type == "raw_string_literal":
        return string_literal_value(
            node, source_bytes, content_node_type="raw_string_literal_content", node_type="raw_string_literal"
        )
    return None


def _receiver_type_name(method_decl: Node, source_bytes: bytes) -> str | None:
    receiver = method_decl.child_by_field_name("receiver")
    if receiver is None:
        return None
    param_decl = next((c for c in receiver.children if c.type == "parameter_declaration"), None)
    if param_decl is None:
        return None
    type_node = param_decl.child_by_field_name("type")
    if type_node is None:
        return None
    if type_node.type == "pointer_type":
        inner = type_node.children[-1]
        return node_text(inner, source_bytes) if inner.type == "type_identifier" else None
    if type_node.type == "type_identifier":
        return node_text(type_node, source_bytes)
    return None


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    for _, caps in run_query(
        GO_LANGUAGE,
        "(function_declaration name: (identifier) @name body: (block) @body) @funcdef",
        root,
    ):
        fn_node = caps["funcdef"][0]
        bare_name = node_text(caps["name"][0], source_bytes)
        start_line, end_line = line_range(fn_node)
        defs.append(
            FunctionDef(
                qualified_name=bare_name, bare_name=bare_name, file=rel_path,
                start_line=start_line, end_line=end_line, body_node=caps["body"][0], class_name=None,
            )
        )
    for _, caps in run_query(
        GO_LANGUAGE,
        "(method_declaration name: (field_identifier) @name body: (block) @body) @funcdef",
        root,
    ):
        fn_node = caps["funcdef"][0]
        bare_name = node_text(caps["name"][0], source_bytes)
        receiver_type = _receiver_type_name(fn_node, source_bytes)
        qualified_name = f"{receiver_type}.{bare_name}" if receiver_type else bare_name
        start_line, end_line = line_range(fn_node)
        defs.append(
            FunctionDef(
                qualified_name=qualified_name, bare_name=bare_name, file=rel_path,
                start_line=start_line, end_line=end_line, body_node=caps["body"][0], class_name=receiver_type,
            )
        )
    return defs


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    result: ImportIndex = {}
    for _, caps in run_query(
        GO_LANGUAGE, "(import_spec path: (interpreted_string_literal) @path) @spec", root
    ):
        spec = caps["spec"][0]
        path_value = _go_string_value(caps["path"][0], source_bytes)
        if path_value is None:
            continue
        name_node = spec.child_by_field_name("name")
        if name_node is not None:
            if name_node.type != "package_identifier":
                continue  # dot (`.`) or blank (`_`) import: no single bindable alias
            alias = node_text(name_node, source_bytes)
        else:
            alias = PurePosixPath(path_value).name
        module = PurePosixPath(path_value).name
        result[alias] = ImportedName(module=module, original_name=alias)
    return result


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for _, caps in run_query(GO_LANGUAGE, "(call_expression function: (_) @fn) @call", body_node):
        fn_node = caps["fn"][0]
        call_node = caps["call"][0]
        raw_text = node_text(call_node, source_bytes)

        if fn_node.type == "identifier":
            call_sites.append(CallSite(callee_name=node_text(fn_node, source_bytes), receiver=None, raw_text=raw_text))
        elif fn_node.type == "selector_expression":
            operand = fn_node.child_by_field_name("operand")
            field = fn_node.child_by_field_name("field")
            if field is None:
                continue
            receiver = node_text(operand, source_bytes) if operand is not None else None
            call_sites.append(
                CallSite(callee_name=node_text(field, source_bytes), receiver=receiver, raw_text=raw_text)
            )
    return call_sites


def _go_named_field(literal_value: Node, key: str, source_bytes: bytes) -> tuple[str | None, bool]:
    for child in literal_value.children:
        if child.type != "keyed_element":
            continue
        key_node = child.child_by_field_name("key")
        value_node = child.child_by_field_name("value")
        if key_node is None or value_node is None or not key_node.children or not value_node.children:
            continue
        if node_text(key_node.children[0], source_bytes) != key:
            continue
        value_inner = value_node.children[0]
        literal = _go_string_value(value_inner, source_bytes)
        if literal is not None:
            return literal, True
        return node_text(value_inner, source_bytes), False
    return None, True


def _enclosing_receiver_type(node: Node, source_bytes: bytes) -> str | None:
    current = node.parent
    while current is not None:
        if current.type == "method_declaration":
            return _receiver_type_name(current, source_bytes)
        if current.type == "function_declaration":
            return None
        current = current.parent
    return None


def detect_go_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    """Detects the official `go-sdk` package's `mcp.AddTool(server,
    &mcp.Tool{Name:, Description:, ...}, handler)` pattern -- verified
    against real parses in 2 independent real MCP-server repos
    (wailsapp/wails, kagent-dev/kagent), both depending on
    `github.com/modelcontextprotocol/go-sdk`. `handler` is always a
    function *value* (never called at the registration site): either a
    bare top-level function name, or a `receiver.method` selector -- for
    the latter, since Go has no `self`/`this` to anchor on, the receiver's
    *type* is taken from the enclosing method_declaration the AddTool call
    itself sits in (true in both real repos found: registration happens in
    a method, and the handler is a sibling method of that same receiver
    type) -- a registration made from a plain function assigning a handler
    method on a locally-constructed struct (not observed as the dominant
    shape) would need real local type inference to resolve, out of scope
    here same as everywhere else in this pipeline (name-based, not
    type-resolved).
    """
    tools: list[ToolRecord] = []
    for _, caps in run_query(
        GO_LANGUAGE,
        "(call_expression function: (selector_expression field: (field_identifier) @method) arguments: (argument_list) @args) @call",
        root,
    ):
        if node_text(caps["method"][0], source_bytes) != "AddTool":
            continue
        args_node = caps["args"][0]
        arg_nodes = [c for c in args_node.children if c.type not in ("(", ")", ",")]
        if len(arg_nodes) < 3:
            continue
        tool_arg, handler_arg = arg_nodes[1], arg_nodes[2]

        composite = tool_arg
        if composite.type == "unary_expression":
            operand = composite.child_by_field_name("operand")
            if operand is None:
                continue
            composite = operand
        if composite.type != "composite_literal":
            continue
        literal_value = composite.child_by_field_name("body")
        if literal_value is None:
            continue

        tool_name, tool_name_is_literal = _go_named_field(literal_value, "Name", source_bytes)
        if tool_name is None or not tool_name_is_literal:
            continue
        description, description_is_literal = _go_named_field(literal_value, "Description", source_bytes)

        call_node = caps["call"][0]
        if handler_arg.type == "identifier":
            qualified_name = node_text(handler_arg, source_bytes)
        elif handler_arg.type == "selector_expression":
            field = handler_arg.child_by_field_name("field")
            if field is None:
                continue
            handler_name = node_text(field, source_bytes)
            receiver_type = _enclosing_receiver_type(call_node, source_bytes)
            qualified_name = f"{receiver_type}.{handler_name}" if receiver_type else handler_name
        else:
            continue

        start_line, end_line = line_range(call_node)
        tools.append(
            ToolRecord(
                name=tool_name,
                description=description or "",
                description_is_literal=description_is_literal,
                sdk_pattern="go.mcp_addtool",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=qualified_name,
            )
        )
    return tools
