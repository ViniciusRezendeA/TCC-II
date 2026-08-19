from __future__ import annotations

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

CS_LANGUAGE = spec_for("C#").ts_language


def _cs_string_value(node: Node, source_bytes: bytes) -> str | None:
    return string_literal_value(node, source_bytes, content_node_type="string_literal_content", node_type="string_literal")


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    for _, caps in run_query(
        CS_LANGUAGE,
        "(method_declaration name: (identifier) @name body: (block) @body) @funcdef",
        root,
    ):
        method_node = caps["funcdef"][0]
        bare_name = node_text(caps["name"][0], source_bytes)
        class_name = _enclosing_class_name(method_node, source_bytes)
        qualified_name = f"{class_name}.{bare_name}" if class_name else bare_name
        start_line, end_line = line_range(method_node)
        defs.append(
            FunctionDef(
                qualified_name=qualified_name, bare_name=bare_name, file=rel_path,
                start_line=start_line, end_line=end_line, body_node=caps["body"][0], class_name=class_name,
            )
        )
    return defs


def _enclosing_class_name(method_node: Node, source_bytes: bytes) -> str | None:
    container = method_node.parent
    if container is None or container.type != "declaration_list" or container.parent is None:
        return None
    class_node = container.parent
    if class_node.type != "class_declaration":
        return None
    name_node = class_node.child_by_field_name("name")
    return node_text(name_node, source_bytes) if name_node is not None else None


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    result: ImportIndex = {}
    for _, caps in run_query(CS_LANGUAGE, "(using_directive) @imp", root):
        imp_node = caps["imp"][0]
        path_node = next((c for c in imp_node.children if c.type in ("qualified_name", "identifier")), None)
        if path_node is None:
            continue
        full_path = node_text(path_node, source_bytes)
        alias = full_path.rsplit(".", 1)[-1]
        result[alias] = ImportedName(module=full_path, original_name=alias)
    return result


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for _, caps in run_query(CS_LANGUAGE, "(invocation_expression function: (_) @fn) @call", body_node):
        fn_node = caps["fn"][0]
        call_node = caps["call"][0]
        raw_text = node_text(call_node, source_bytes)

        if fn_node.type == "identifier":
            call_sites.append(CallSite(callee_name=node_text(fn_node, source_bytes), receiver=None, raw_text=raw_text))
        elif fn_node.type == "member_access_expression":
            object_node = fn_node.child_by_field_name("expression")
            name_node = fn_node.child_by_field_name("name")
            if name_node is None:
                continue
            receiver = node_text(object_node, source_bytes) if object_node is not None else None
            call_sites.append(
                CallSite(callee_name=node_text(name_node, source_bytes), receiver=receiver, raw_text=raw_text)
            )
    return call_sites


def detect_csharp_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    """Detects the official C# SDK's `[McpServerTool]` marker + `[Description("...")]`
    attribute pattern (combined `[McpServerTool, Description("...")]` or
    stacked on separate attribute_lists — both forms verified against real
    parses, see the plan)."""
    tools: list[ToolRecord] = []
    for _, caps in run_query(
        CS_LANGUAGE,
        "(method_declaration name: (identifier) @name body: (block) @body) @funcdef",
        root,
    ):
        method_node = caps["funcdef"][0]
        attribute_lists = [c for c in method_node.children if c.type == "attribute_list"]
        attributes = [a for al in attribute_lists for a in al.children if a.type == "attribute"]
        attr_names = {node_text(a.child_by_field_name("name"), source_bytes) for a in attributes}

        if "McpServerTool" not in attr_names:
            continue

        description_attr = next(
            (a for a in attributes if node_text(a.child_by_field_name("name"), source_bytes) == "Description"), None
        )
        description, description_is_literal = _first_attribute_argument(description_attr, source_bytes)

        bare_name = node_text(caps["name"][0], source_bytes)
        class_name = _enclosing_class_name(method_node, source_bytes)
        qualified_name = f"{class_name}.{bare_name}" if class_name else bare_name
        start_line, end_line = line_range(method_node)

        tools.append(
            ToolRecord(
                name=bare_name,
                description=description or "",
                description_is_literal=description_is_literal,
                sdk_pattern="csharp.mcpservertool_attribute",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=qualified_name,
            )
        )
    return tools


def _first_attribute_argument(attribute_node: Node | None, source_bytes: bytes) -> tuple[str | None, bool]:
    if attribute_node is None:
        return None, True
    arg_list = next((c for c in attribute_node.children if c.type == "attribute_argument_list"), None)
    if arg_list is None:
        return None, True
    arg = next((c for c in arg_list.children if c.type == "attribute_argument"), None)
    if arg is None or arg.child_count == 0:
        return None, True
    value_node = arg.children[0]
    literal = _cs_string_value(value_node, source_bytes)
    if literal is not None:
        return literal, True
    return node_text(value_node, source_bytes), False
