from __future__ import annotations

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import CallSite
from mcp_pipeline.extraction.definition_index import DefinitionIndex, FunctionDef
from mcp_pipeline.extraction.import_index import ImportedName, ImportIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.models import SourceLocation, ToolRecord
from mcp_pipeline.extraction.parser_utils import (
    line_range,
    node_text,
    run_query,
    string_literal_value,
)
from mcp_pipeline.extraction.value_index import (
    MAX_VALUE_RESOLUTION_HOPS,
    ValueDef,
    ValueIndex,
    resolve_value,
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


def extract_values(root: Node, source_bytes: bytes, rel_path: str) -> list[ValueDef]:
    """Module-top-level assignments only (`search_tool = types.Tool(...)`,
    `TOOLS = [search_tool, ...]`) -- verified live that anchoring on
    `(module (expression_statement (assignment ...)))` correctly excludes
    assignments nested inside a function body or a class body, which aren't
    reachable by bare-name reference the way module-level names are.
    """
    values: list[ValueDef] = []
    query_str = """
    (module
      (expression_statement
        (assignment left: (identifier) @name right: (_) @value)))
    """
    for _, caps in run_query(PY_LANGUAGE, query_str, root):
        name = node_text(caps["name"][0], source_bytes)
        values.append(ValueDef(bare_name=name, file=rel_path, value_node=caps["value"][0]))
    return values


def _first_return_expr(body_node: Node) -> Node | None:
    """The first top-level `return <expr>` in a function body. All confirmed
    real list_tools()/helper-function examples have a single simple return
    at the top level -- a function with multiple conditional returns isn't
    a confirmed case and isn't handled (falls through to "give up cleanly"
    at the call site, same as any other unresolvable shape)."""
    for child in body_node.children:
        if child.type == "return_statement":
            value_children = [c for c in child.children if c.type != "return"]
            if value_children:
                return value_children[0]
    return None


def _unwrap_await(node: Node) -> Node:
    if node.type == "await":
        inner = [c for c in node.children if c.type != "await"]
        if inner:
            return inner[0]
    return node


def _resolve_list_tools_return_expr(
    node: Node,
    current_file: str,
    source_bytes: bytes,
    definitions: DefinitionIndex,
    values: ValueIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
    hops_remaining: int,
) -> tuple[Node, str, bytes] | None:
    """Phase A of the plan's 2-phase value resolution: locate the literal
    `list` expression a `list_tools()` handler (or a same-file helper it
    delegates to) ultimately returns. Returns (list_node, its_file,
    its_source_bytes), or None if it can't be resolved statically within
    the hop budget -- dynamic construction (a comprehension over a runtime
    registry), an unresolvable call, or budget exhausted all fall through
    to None here; callers must treat that as "zero tools from this path",
    never guess.
    """
    node = _unwrap_await(node)

    if node.type == "list":
        return node, current_file, source_bytes

    if hops_remaining <= 0:
        return None

    if node.type == "identifier":
        name = node_text(node, source_bytes)
        resolved, _ambiguous = resolve_value(name, current_file, values, imports_by_file)
        if resolved is None:
            return None
        resolved_source_bytes = source_bytes_by_file[resolved.file]
        return _resolve_list_tools_return_expr(
            resolved.value_node, resolved.file, resolved_source_bytes,
            definitions, values, imports_by_file, source_bytes_by_file, hops_remaining - 1,
        )

    if node.type == "call":
        # One hop of same-file helper-function-call indirection (real case:
        # jgravelle/jcodemunch-mcp's list_tools() calling _build_tools_list()).
        # A call that isn't a same-file function -- e.g. Tool(...) itself
        # shouldn't appear here (list_tools() always returns a list, never a
        # bare Tool), or a genuinely dynamic/external call -- gives up.
        function_node = node.child_by_field_name("function")
        if function_node is None or function_node.type != "identifier":
            return None
        callee_name = node_text(function_node, source_bytes)
        same_file_fns = [d for d in definitions.by_bare_name.get(callee_name, []) if d.file == current_file]
        if len(same_file_fns) != 1:
            return None
        helper = same_file_fns[0]
        return_expr = _first_return_expr(helper.body_node)
        if return_expr is None:
            return None
        return _resolve_list_tools_return_expr(
            return_expr, helper.file, source_bytes_by_file[helper.file],
            definitions, values, imports_by_file, source_bytes_by_file, hops_remaining - 1,
        )

    return None


def _resolve_list_element(
    node: Node,
    current_file: str,
    source_bytes: bytes,
    values: ValueIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
) -> tuple[Node, bytes] | None:
    """Phase B: resolve one element of an already-located tool list to a
    Tool(...)-shaped call node. Separate, small per-element cap (same-file,
    then at most 1 import hop) independent of Phase A's budget -- these are
    independent per-element lookups, not a chain. An element that fails to
    resolve is skipped individually by the caller, not treated as aborting
    the whole list.
    """
    if node.type == "call":
        return node, source_bytes

    if node.type == "identifier":
        name = node_text(node, source_bytes)
        resolved, _ambiguous = resolve_value(name, current_file, values, imports_by_file)
        if resolved is None:
            return None
        resolved_node = resolved.value_node
        if resolved_node.type == "call":
            return resolved_node, source_bytes_by_file[resolved.file]
        return None

    return None


LIST_TOOLS_DECORATOR_QUERY = """
(decorated_definition
  (decorator
    (call
      function: (attribute
        object: (identifier)
        attribute: (identifier) @decorator_method)
      arguments: (argument_list)))
  definition: (function_definition
    name: (identifier) @tool_func_name
    body: (block) @tool_func_body) @tool_func_def)
"""


def detect_lowlevel_list_tools(
    root: Node,
    source_bytes: bytes,
    rel_path: str,
    definitions: DefinitionIndex,
    values: ValueIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
) -> list[ToolRecord]:
    """Detects the official MCP Python SDK's low-level `@<obj>.list_tools()`
    decorator -- the alternative to the already-detected `@mcp.tool()`
    high-level decorator sugar. Confirmed in 14 of 206 repos in the Etapa 2
    pilot corpus, 8 with zero overlap with the high-level pattern.

    Structural trade-off, explicit: the resulting call graph's level 1 is
    the `list_tools()` function itself (metadata/schema construction), not
    the tool's actual execution logic -- that lives in a separate
    `@server.call_tool()` handler this pattern does not attempt to link to.
    """
    tools: list[ToolRecord] = []
    for _, caps in run_query(PY_LANGUAGE, LIST_TOOLS_DECORATOR_QUERY, root):
        if node_text(caps["decorator_method"][0], source_bytes) != "list_tools":
            continue

        func_node = caps["tool_func_def"][0]
        func_name = node_text(caps["tool_func_name"][0], source_bytes)
        body_node = caps["tool_func_body"][0]
        class_name = _enclosing_class_name(func_node, source_bytes)
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name

        return_expr = _first_return_expr(body_node)
        if return_expr is None:
            continue

        located = _resolve_list_tools_return_expr(
            return_expr, rel_path, source_bytes,
            definitions, values, imports_by_file, source_bytes_by_file,
            MAX_VALUE_RESOLUTION_HOPS,
        )
        if located is None:
            continue
        list_node, list_file, list_source_bytes = located

        for element in list_node.children:
            if element.type in ("[", "]", ","):
                continue
            resolved = _resolve_list_element(
                element, list_file, list_source_bytes, values, imports_by_file, source_bytes_by_file
            )
            if resolved is None:
                continue
            tool_call_node, tool_source_bytes = resolved
            args_node = tool_call_node.child_by_field_name("arguments")
            if args_node is None:
                continue

            tool_name, tool_name_is_literal = _keyword_arg_value(args_node, "name", tool_source_bytes)
            if tool_name is None or not tool_name_is_literal:
                continue  # no literal name -- skip rather than fabricate
            description, description_is_literal = _keyword_arg_value(args_node, "description", tool_source_bytes)

            start_line, end_line = line_range(func_node)
            tools.append(
                ToolRecord(
                    name=tool_name,
                    description=description or "",
                    description_is_literal=description_is_literal,
                    sdk_pattern="python.list_tools_lowlevel",
                    source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                    qualified_name=qualified_name,
                )
            )
    return tools
