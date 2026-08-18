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

PY_LANGUAGE = spec_for("Python").ts_language


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    matches = run_query(PY_LANGUAGE, "(function_definition name: (identifier) @name) @funcdef", root)
    for _, caps in matches:
        func_node = caps["funcdef"][0]
        name_node = caps["name"][0]
        func_name = node_text(name_node, source_bytes)

        class_name = _enclosing_class_name(func_node, source_bytes)
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name

        start_line, end_line = line_range(func_node)
        body_node = func_node.child_by_field_name("body")
        defs.append(
            FunctionDef(
                qualified_name=qualified_name,
                bare_name=func_name,
                file=rel_path,
                start_line=start_line,
                end_line=end_line,
                body_node=body_node,
                class_name=class_name,
            )
        )
    return defs


def _enclosing_class_name(func_node: Node, source_bytes: bytes) -> str | None:
    container = func_node.parent
    if container is not None and container.type == "decorated_definition":
        container = container.parent
    if container is None or container.type != "block" or container.parent is None:
        return None
    class_node = container.parent
    if class_node.type != "class_definition":
        return None
    name_node = class_node.child_by_field_name("name")
    return node_text(name_node, source_bytes) if name_node is not None else None


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    result: ImportIndex = {}
    _walk_imports(root, source_bytes, result)
    return result


def _walk_imports(node: Node, source_bytes: bytes, result: ImportIndex) -> None:
    if node.type == "import_statement":
        for child in node.children:
            if child.type == "dotted_name":
                mod = node_text(child, source_bytes)
                alias = mod.split(".")[0]  # `import x.y.z` binds only the top-level name `x`
                result[alias] = ImportedName(module=mod, original_name=alias)
            elif child.type == "aliased_import":
                dotted, alias_ident = child.children[0], child.children[-1]
                mod = node_text(dotted, source_bytes)
                alias = node_text(alias_ident, source_bytes)
                result[alias] = ImportedName(module=mod, original_name=mod.split(".")[-1])
    elif node.type == "import_from_statement":
        module_node = next(
            (c for c in node.children if c.type in ("dotted_name", "relative_import")), None
        )
        module_text = node_text(module_node, source_bytes) if module_node is not None else ""
        for child in node.children:
            if child is module_node:
                continue
            if child.type == "dotted_name":
                name = node_text(child, source_bytes)
                result[name] = ImportedName(module=module_text, original_name=name)
            elif child.type == "aliased_import":
                dotted, alias_ident = child.children[0], child.children[-1]
                orig_name = node_text(dotted, source_bytes)
                alias = node_text(alias_ident, source_bytes)
                result[alias] = ImportedName(module=module_text, original_name=orig_name)

    for child in node.children:
        _walk_imports(child, source_bytes, result)


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    matches = run_query(PY_LANGUAGE, "(call function: (_) @fn) @call", body_node)
    for _, caps in matches:
        fn_node = caps["fn"][0]
        call_node = caps["call"][0]
        raw_text = node_text(call_node, source_bytes)

        if fn_node.type == "identifier":
            call_sites.append(CallSite(callee_name=node_text(fn_node, source_bytes), receiver=None, raw_text=raw_text))
        elif fn_node.type == "attribute":
            object_node = fn_node.child_by_field_name("object")
            attribute_node = fn_node.child_by_field_name("attribute")
            if attribute_node is None:
                continue
            receiver = node_text(object_node, source_bytes) if object_node is not None else None
            call_sites.append(
                CallSite(callee_name=node_text(attribute_node, source_bytes), receiver=receiver, raw_text=raw_text)
            )
    return call_sites


def detect_fastmcp_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    """Detects the FastMCP high-level decorator pattern (`@mcp.tool()`),
    the dominant pattern in the Python SDK as of the 2026-08 preliminary
    research — see the plan's SDK generation table. Low-level
    `@server.list_tools()` / `Server(..., on_list_tools=...)` patterns are a
    documented follow-up, not yet implemented.
    """
    tools: list[ToolRecord] = []
    query_str = """
    (decorated_definition
      (decorator
        (call
          function: (attribute
            object: (identifier)
            attribute: (identifier) @decorator_method)
          arguments: (argument_list) @decorator_args))
      definition: (function_definition
        name: (identifier) @tool_func_name
        body: (block) @tool_func_body) @tool_func_def)
    """
    for _, caps in run_query(PY_LANGUAGE, query_str, root):
        if node_text(caps["decorator_method"][0], source_bytes) != "tool":
            continue

        func_node = caps["tool_func_def"][0]
        func_name = node_text(caps["tool_func_name"][0], source_bytes)
        body_node = caps["tool_func_body"][0]
        args_node = caps["decorator_args"][0]

        tool_name, _ = _keyword_arg_value(args_node, "name", source_bytes)
        description, description_is_literal = _keyword_arg_value(args_node, "description", source_bytes)
        if description is None:
            description, description_is_literal = _docstring_of(body_node, source_bytes)

        class_name = _enclosing_class_name(func_node, source_bytes)
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name
        start_line, end_line = line_range(func_node)

        tools.append(
            ToolRecord(
                name=tool_name or func_name,
                description=description or "",
                description_is_literal=description_is_literal,
                sdk_pattern="python.fastmcp_decorator",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=qualified_name,
            )
        )
    return tools


def _keyword_arg_value(args_node: Node, key: str, source_bytes: bytes) -> tuple[str | None, bool]:
    """Returns (value, is_literal). `is_literal` is False (but the raw text
    is still returned, never dropped) when the value expression isn't a
    plain string literal — e.g. an f-string or a name reference."""
    for child in args_node.children:
        if child.type != "keyword_argument":
            continue
        name_node = child.child_by_field_name("name")
        value_node = child.child_by_field_name("value")
        if name_node is None or value_node is None or node_text(name_node, source_bytes) != key:
            continue
        literal = string_literal_value(value_node, source_bytes)
        if literal is not None:
            return literal, True
        return node_text(value_node, source_bytes), False
    return None, True


def _docstring_of(body_node: Node, source_bytes: bytes) -> tuple[str | None, bool]:
    """Replicates the FastMCP SDK's own fallback, verified in source:
    `description or fn.__doc__`."""
    if body_node.child_count == 0:
        return None, True
    first_stmt = body_node.children[0]
    if first_stmt.type != "expression_statement" or first_stmt.child_count == 0:
        return None, True
    expr = first_stmt.children[0]
    literal = string_literal_value(expr, source_bytes)
    if literal is not None:
        return literal.strip(), True
    return None, True
