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

JAVA_LANGUAGE = spec_for("Java").ts_language


def _java_string_value(node: Node, source_bytes: bytes) -> str | None:
    return string_literal_value(node, source_bytes, content_node_type="string_fragment", node_type="string_literal")


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    for _, caps in run_query(
        JAVA_LANGUAGE,
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
    if container is None or container.type != "class_body" or container.parent is None:
        return None
    class_node = container.parent
    if class_node.type != "class_declaration":
        return None
    name_node = class_node.child_by_field_name("name")
    return node_text(name_node, source_bytes) if name_node is not None else None


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    result: ImportIndex = {}
    for _, caps in run_query(JAVA_LANGUAGE, "(import_declaration) @imp", root):
        imp_node = caps["imp"][0]
        path_node = next((c for c in imp_node.children if c.type in ("scoped_identifier", "identifier")), None)
        if path_node is None:
            continue
        full_path = node_text(path_node, source_bytes)
        alias = full_path.rsplit(".", 1)[-1]
        if alias == "*":
            continue  # wildcard import — no single alias to bind
        result[alias] = ImportedName(module=full_path, original_name=alias)
    return result


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for _, caps in run_query(JAVA_LANGUAGE, "(method_invocation name: (identifier) @name) @call", body_node):
        call_node = caps["call"][0]
        name_node = caps["name"][0]
        object_node = call_node.child_by_field_name("object")
        receiver = node_text(object_node, source_bytes) if object_node is not None else None
        call_sites.append(
            CallSite(
                callee_name=node_text(name_node, source_bytes),
                receiver=receiver,
                raw_text=node_text(call_node, source_bytes),
            )
        )
    return call_sites


def detect_spring_ai_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    """Detects the Spring AI `@Tool(description = "...")` annotation
    pattern — per the plan's research, the dominant real-world pattern for
    Java MCP servers (more common in practice than the official SDK's
    builder API, which is a documented follow-up)."""
    tools: list[ToolRecord] = []
    query_str = """
    (method_declaration
      (modifiers
        (annotation
          name: (identifier) @annotation_name
          arguments: (annotation_argument_list) @annotation_args))
      name: (identifier) @method_name) @method
    """
    for _, caps in run_query(JAVA_LANGUAGE, query_str, root):
        if node_text(caps["annotation_name"][0], source_bytes) != "Tool":
            continue

        method_node = caps["method"][0]
        method_name = node_text(caps["method_name"][0], source_bytes)
        description, description_is_literal = _annotation_arg(caps["annotation_args"][0], "description", source_bytes)
        tool_name, _ = _annotation_arg(caps["annotation_args"][0], "name", source_bytes)

        class_name = _enclosing_class_name(method_node, source_bytes)
        qualified_name = f"{class_name}.{method_name}" if class_name else method_name
        start_line, end_line = line_range(method_node)

        tools.append(
            ToolRecord(
                name=tool_name or method_name,
                description=description or "",
                description_is_literal=description_is_literal,
                sdk_pattern="java.springai_annotation",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=qualified_name,
            )
        )
    return tools


def _annotation_arg(args_node: Node, key: str, source_bytes: bytes) -> tuple[str | None, bool]:
    for child in args_node.children:
        if child.type != "element_value_pair":
            continue
        key_node = child.child_by_field_name("key")
        value_node = child.child_by_field_name("value")
        if key_node is None or value_node is None or node_text(key_node, source_bytes) != key:
            continue
        literal = _java_string_value(value_node, source_bytes)
        if literal is not None:
            return literal, True
        return node_text(value_node, source_bytes), False
    return None, True
