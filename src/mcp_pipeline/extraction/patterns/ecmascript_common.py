from __future__ import annotations

"""Shared implementation for TypeScript and JavaScript — verified identical
grammar shapes (function_declaration, variable_declarator+arrow/function
expression, class method_definition, call_expression, import_statement,
string/string_fragment) across tree-sitter-typescript and
tree-sitter-javascript for every construct used here. `typescript_patterns.py`
and `javascript_patterns.py` are thin bindings of this module to their
respective `Language` object — this is what architecture principle #6 in the
plan means by "shared logic parametrized by a per-language plug-in", applied
one level further: two languages close enough to share the plug-in itself.
"""

from dataclasses import dataclass

from tree_sitter import Language, Node

from mcp_pipeline.extraction.call_graph_builder import CallSite
from mcp_pipeline.extraction.definition_index import DefinitionIndex, FunctionDef
from mcp_pipeline.extraction.import_index import ImportedName, ImportIndex
from mcp_pipeline.extraction.models import SourceLocation, ToolRecord
from mcp_pipeline.extraction.parser_utils import (
    line_range,
    node_text,
    run_query,
    string_literal_value,
    template_string_literal_value,
)
from mcp_pipeline.extraction.value_index import (
    MAX_VALUE_RESOLUTION_HOPS,
    ValueDef,
    ValueIndex,
    resolve_value,
)

TOOL_CALL_QUERY = """
(call_expression
  function: (member_expression
    object: (identifier)
    property: (property_identifier) @method_name)
  arguments: (arguments) @args) @call
"""


def _string_value(node: Node, source_bytes: bytes) -> str | None:
    """Single choke-point for TS/JS string-literal extraction -- every
    caller in this module (and the 2 new low-level patterns) funnels through
    here, so the template-literal fix below applies uniformly without
    touching call sites individually."""
    value = string_literal_value(node, source_bytes, content_node_type="string_fragment")
    if value is not None:
        return value
    return template_string_literal_value(node, source_bytes)


def synthetic_handler_name(rel_path: str, tool_name: str) -> str:
    """Inline arrow-function/function-expression handlers (the idiomatic
    style for `.tool()`/`.registerTool()`) aren't named definitions, so they
    can't be looked up by name like every other language's tools. This
    synthetic key lets them live in the same DefinitionIndex as everything
    else — both extract_definitions and detect_mcp_tools use this exact
    naming convention so the two stay linked.
    """
    return f"<inline_tool_handler>:{rel_path}:{tool_name}"


@dataclass
class _ToolCallMatch:
    tool_name: str
    description: str | None
    description_is_literal: bool
    sdk_pattern: str
    handler_node: Node | None  # set when the handler is an inline arrow/function expression
    handler_identifier: str | None  # set when the handler is a bare identifier reference
    call_node: Node


def _find_tool_registration_calls(language: Language, root: Node, source_bytes: bytes) -> list[_ToolCallMatch]:
    matches: list[_ToolCallMatch] = []
    for _, caps in run_query(language, TOOL_CALL_QUERY, root):
        method_name = node_text(caps["method_name"][0], source_bytes)
        if method_name not in ("tool", "registerTool"):
            continue

        args_node = caps["args"][0]
        arg_nodes = [c for c in args_node.children if c.type not in ("(", ")", ",")]
        if len(arg_nodes) < 2:
            continue

        name_node = arg_nodes[0]
        tool_name = _string_value(name_node, source_bytes)
        if tool_name is None:
            # The name argument isn't a literal string (e.g. `tool.name` from
            # a data-driven `for (const tool of tools) registerTool(tool)`
            # loop, seen in real repos — chrome-devtools-mcp). This call site
            # represents an unknown number of actual tools built elsewhere,
            # not one — a single fabricated "tool" named after the raw
            # expression text would be worse than finding none, so skip it
            # entirely rather than guess.
            continue

        second_arg = arg_nodes[1]
        if second_arg.type == "string":
            # old v1 form: server.tool(name, description, schema, handler)
            sdk_pattern = "tool_v1"
            description = _string_value(second_arg, source_bytes)
            description_is_literal = description is not None
            if description is None:
                description = node_text(second_arg, source_bytes)
        elif second_arg.type == "object":
            # registerTool(name, {description, inputSchema}, handler)
            sdk_pattern = "registerTool"
            description, description_is_literal = _object_field(second_arg, "description", source_bytes)
        else:
            continue

        handler_node = None
        handler_identifier = None
        for arg in arg_nodes[2:]:
            if arg.type in ("arrow_function", "function_expression"):
                handler_node = arg
                break
            if arg.type == "identifier":
                handler_identifier = node_text(arg, source_bytes)

        matches.append(
            _ToolCallMatch(
                tool_name=tool_name,
                description=description,
                description_is_literal=description_is_literal,
                sdk_pattern=sdk_pattern,
                handler_node=handler_node,
                handler_identifier=handler_identifier,
                call_node=caps["call"][0],
            )
        )
    return matches


def _object_field(object_node: Node, key: str, source_bytes: bytes) -> tuple[str | None, bool]:
    for child in object_node.children:
        if child.type != "pair":
            continue
        key_node = child.child_by_field_name("key")
        value_node = child.child_by_field_name("value")
        if key_node is None or value_node is None or node_text(key_node, source_bytes) != key:
            continue
        literal = _string_value(value_node, source_bytes)
        if literal is not None:
            return literal, True
        return node_text(value_node, source_bytes), False
    return None, True


def _object_field_node(object_node: Node, key: str, source_bytes: bytes) -> Node | None:
    """Like _object_field, but returns the raw value Node instead of an
    extracted string -- needed for the `execute:` field, where the caller
    has to distinguish an inline function from a bare identifier reference,
    not just read text."""
    for child in object_node.children:
        if child.type != "pair":
            continue
        key_node = child.child_by_field_name("key")
        value_node = child.child_by_field_name("value")
        if key_node is None or value_node is None or node_text(key_node, source_bytes) != key:
            continue
        return value_node
    return None


FASTMCP_ADDTOOL_CALL_QUERY = """
(call_expression
  function: [
    (identifier) @bare_name
    (member_expression property: (property_identifier) @method_name)
  ]
  arguments: (arguments) @args) @call
"""

# Only the one confirmed real-world case (brightdata/brightdata-mcp defines a
# local `const addTool = (tool) => {...}` wrapper and calls it bare, forwarding
# to `server.addTool(tool)` internally) -- same documented,
# evidence-only-extend-if-needed convention as EXCLUDED_DIR_NAMES.
_ADDTOOL_WRAPPER_NAMES = frozenset({"addTool"})


def detect_fastmcp_npm_addtool(
    language: Language, language_tag: str, root: Node, source_bytes: bytes, rel_path: str
) -> list[ToolRecord]:
    """Detects the third-party `fastmcp` npm package's `.addTool({...})`
    idiom -- a real, named, shared community SDK (confirmed via package.json
    dependency declarations across multiple real repos), distinct from both
    the official TS/JS SDK and Python's FastMCP. Every confirmed real call
    site has the tool object literal directly inline, so unlike the other 2
    new patterns this needs no value-index/cross-file resolution -- it's
    structurally closer to the existing `.tool()`/`.registerTool()` pattern.

    Matches both `<obj>.addTool({...})` and a bare-identifier wrapper call
    (the brightdata-mcp shape above) -- the wrapper-internal
    `server.addTool(tool)` calls pass a non-literal `tool` parameter and are
    excluded by the object-literal-first-argument check below, the same
    "don't fabricate" principle `_find_tool_registration_calls` already uses
    for non-literal tool names.
    """
    tools: list[ToolRecord] = []
    for _, caps in run_query(language, FASTMCP_ADDTOOL_CALL_QUERY, root):
        method_name_caps = caps.get("method_name")
        bare_name_caps = caps.get("bare_name")
        if method_name_caps is not None:
            if node_text(method_name_caps[0], source_bytes) != "addTool":
                continue
        elif bare_name_caps is not None:
            if node_text(bare_name_caps[0], source_bytes) not in _ADDTOOL_WRAPPER_NAMES:
                continue
        else:
            continue

        args_node = caps["args"][0]
        arg_nodes = [c for c in args_node.children if c.type not in ("(", ")", ",")]
        if not arg_nodes or arg_nodes[0].type != "object":
            continue
        tool_obj = arg_nodes[0]

        tool_name, tool_name_is_literal = _object_field(tool_obj, "name", source_bytes)
        if tool_name is None or not tool_name_is_literal:
            # No `name` key, or a non-literal value (e.g. a loop variable's
            # `.name` property) -- skip rather than fabricate, same
            # principle as the existing pattern's non-literal-name handling.
            continue
        description, description_is_literal = _object_field(tool_obj, "description", source_bytes)

        handler_node = None
        handler_identifier = None
        execute_value = _object_field_node(tool_obj, "execute", source_bytes)
        if execute_value is not None:
            if execute_value.type == "identifier":
                handler_identifier = node_text(execute_value, source_bytes)
            else:
                # arrow_function/function_expression directly, OR a wrapped
                # call like `execute: tool_fn('search_engine', async (...)
                # => {...})` (real case: brightdata-mcp's telemetry
                # wrapper). Either way, register the whole execute:
                # expression as the body to scan for call sites -- looser
                # than requiring a bare function, but losing the tool's
                # name/description entirely because its handler is wrapped
                # would be worse than an imprecise call graph for it.
                handler_node = execute_value

        if handler_node is not None:
            qualified_name = synthetic_handler_name(rel_path, tool_name)
        elif handler_identifier is not None:
            qualified_name = handler_identifier
        else:
            continue  # no discoverable handler -- skip rather than guess

        start_line, end_line = line_range(caps["call"][0])
        tools.append(
            ToolRecord(
                name=tool_name,
                description=description or "",
                description_is_literal=description_is_literal,
                sdk_pattern=f"{language_tag}.fastmcp_npm_addtool",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=qualified_name,
            )
        )
    return tools


def extract_fastmcp_addtool_definitions(
    language: Language, root: Node, source_bytes: bytes, rel_path: str
) -> list[FunctionDef]:
    """Registers a FunctionDef for each addTool call's inline `execute`
    handler, mirroring extract_definitions()'s existing loop over
    _find_tool_registration_calls() for the .tool()/.registerTool() pattern
    -- needed so detect_tools_with_call_graphs() can look the tool's
    qualified_name up and build its call graph."""
    defs: list[FunctionDef] = []
    for _, caps in run_query(language, FASTMCP_ADDTOOL_CALL_QUERY, root):
        method_name_caps = caps.get("method_name")
        bare_name_caps = caps.get("bare_name")
        if method_name_caps is not None:
            if node_text(method_name_caps[0], source_bytes) != "addTool":
                continue
        elif bare_name_caps is not None:
            if node_text(bare_name_caps[0], source_bytes) not in _ADDTOOL_WRAPPER_NAMES:
                continue
        else:
            continue

        args_node = caps["args"][0]
        arg_nodes = [c for c in args_node.children if c.type not in ("(", ")", ",")]
        if not arg_nodes or arg_nodes[0].type != "object":
            continue
        tool_obj = arg_nodes[0]

        tool_name, tool_name_is_literal = _object_field(tool_obj, "name", source_bytes)
        if tool_name is None or not tool_name_is_literal:
            continue

        execute_value = _object_field_node(tool_obj, "execute", source_bytes)
        if execute_value is None or execute_value.type == "identifier":
            continue  # identifier-referenced handlers are registered under their own name elsewhere

        # For a bare arrow/function expression, use just its own body (less
        # noise, matches existing precedent). For anything else (e.g. a
        # wrapped call like `tool_fn('name', async (...) => {...})`), fall
        # back to the whole expression -- extract_calls()'s query works
        # over any node, not just a statement_block.
        if execute_value.type in ("arrow_function", "function_expression"):
            body = execute_value.child_by_field_name("body")
            if body is None:
                continue
        else:
            body = execute_value

        start_line, end_line = line_range(execute_value)
        synthetic_name = synthetic_handler_name(rel_path, tool_name)
        defs.append(
            FunctionDef(
                qualified_name=synthetic_name, bare_name=synthetic_name, file=rel_path,
                start_line=start_line, end_line=end_line, body_node=body, class_name=None,
            )
        )
    return defs


def detect_mcp_tools(
    language: Language, language_tag: str, root: Node, source_bytes: bytes, rel_path: str
) -> list[ToolRecord]:
    """Detects the high-level `.tool()` (v1) / `.registerTool()` (v1
    registerTool alias and v2) patterns — see the plan's SDK generation
    table. Low-level `setRequestHandler(...)` is a documented follow-up.
    `language_tag` (e.g. "typescript"/"javascript") namespaces `sdk_pattern`
    so the two languages stay distinguishable in the final dataset even
    though the detection logic is shared.
    """
    tools: list[ToolRecord] = []
    for match in _find_tool_registration_calls(language, root, source_bytes):
        if match.handler_node is not None:
            qualified_name = synthetic_handler_name(rel_path, match.tool_name)
        elif match.handler_identifier is not None:
            qualified_name = match.handler_identifier
        else:
            continue  # no discoverable handler — skip rather than guess

        start_line, end_line = line_range(match.call_node)
        tools.append(
            ToolRecord(
                name=match.tool_name,
                description=match.description or "",
                description_is_literal=match.description_is_literal,
                sdk_pattern=f"{language_tag}.{match.sdk_pattern}",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=qualified_name,
            )
        )
    return tools


def extract_definitions(language: Language, root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []

    for _, caps in run_query(
        language,
        "(function_declaration name: (identifier) @name body: (statement_block) @body) @funcdef",
        root,
    ):
        defs.append(_function_def_from(caps["name"][0], caps["funcdef"][0], caps["body"][0], rel_path, source_bytes, class_name=None))

    for _, caps in run_query(
        language,
        "(variable_declarator name: (identifier) @name value: [(arrow_function) (function_expression)] @func)",
        root,
    ):
        func_node = caps["func"][0]
        body = func_node.child_by_field_name("body")
        if body is None or body.type != "statement_block":
            continue  # expression-bodied arrow function (`x => x * 2`) has no call sites to scan meaningfully here
        defs.append(_function_def_from(caps["name"][0], func_node, body, rel_path, source_bytes, class_name=None))

    for _, caps in run_query(
        language,
        "(method_definition name: (property_identifier) @name body: (statement_block) @body) @funcdef",
        root,
    ):
        method_node = caps["funcdef"][0]
        class_name = _enclosing_class_name(method_node, source_bytes)
        defs.append(_function_def_from(caps["name"][0], method_node, caps["body"][0], rel_path, source_bytes, class_name))

    for match in _find_tool_registration_calls(language, root, source_bytes):
        if match.handler_node is None:
            continue
        body = match.handler_node.child_by_field_name("body")
        if body is None or body.type != "statement_block":
            continue
        start_line, end_line = line_range(match.handler_node)
        synthetic_name = synthetic_handler_name(rel_path, match.tool_name)
        defs.append(
            FunctionDef(
                qualified_name=synthetic_name, bare_name=synthetic_name, file=rel_path,
                start_line=start_line, end_line=end_line, body_node=body, class_name=None,
            )
        )

    defs.extend(extract_fastmcp_addtool_definitions(language, root, source_bytes, rel_path))
    defs.extend(register_lowlevel_list_tools_definitions(language, root, source_bytes, rel_path))

    return defs


def _function_def_from(name_node: Node, func_node: Node, body_node: Node, rel_path: str, source_bytes: bytes, class_name: str | None) -> FunctionDef:
    bare_name = node_text(name_node, source_bytes)
    qualified_name = f"{class_name}.{bare_name}" if class_name else bare_name
    start_line, end_line = line_range(func_node)
    return FunctionDef(
        qualified_name=qualified_name, bare_name=bare_name, file=rel_path,
        start_line=start_line, end_line=end_line, body_node=body_node, class_name=class_name,
    )


def _enclosing_class_name(node: Node, source_bytes: bytes) -> str | None:
    container = node.parent
    if container is None or container.type != "class_body" or container.parent is None:
        return None
    class_node = container.parent
    if class_node.type not in ("class_declaration", "class"):
        return None
    name_node = class_node.child_by_field_name("name")
    return node_text(name_node, source_bytes) if name_node is not None else None


def extract_imports(language: Language, root: Node, source_bytes: bytes) -> ImportIndex:
    """Only handles ES module `import` statements — CommonJS `require()` is
    a plain call expression, not a declaration, so it isn't captured here.
    This is a known gap for older-style JS servers; call sites that would
    have resolved via a `require()`-bound alias still fall through to the
    resolver's same-file/repo-wide-unique steps, which handles the common
    case (bare destructured imports) but not `const cache = require(...);
    cache.getCached()` qualified-call forms.
    """
    result: ImportIndex = {}
    for _, caps in run_query(language, "(import_statement source: (string) @source) @stmt", root):
        module = _string_value(caps["source"][0], source_bytes) or ""
        clause = next((c for c in caps["stmt"][0].children if c.type == "import_clause"), None)
        if clause is None:
            continue
        _collect_import_clause(clause, module, source_bytes, result)
    return result


def _collect_import_clause(clause: Node, module: str, source_bytes: bytes, result: ImportIndex) -> None:
    for child in clause.children:
        if child.type == "identifier":
            # default import: `import Foo from "./thing"`
            alias = node_text(child, source_bytes)
            result[alias] = ImportedName(module=module, original_name="default")
        elif child.type == "namespace_import":
            alias_node = child.children[-1]  # `* as alias`
            alias = node_text(alias_node, source_bytes)
            result[alias] = ImportedName(module=module, original_name="*")
        elif child.type == "named_imports":
            for spec in (c for c in child.children if c.type == "import_specifier"):
                idents = [c for c in spec.children if c.type == "identifier"]
                orig_name = node_text(idents[0], source_bytes)
                alias = node_text(idents[-1], source_bytes)
                result[alias] = ImportedName(module=module, original_name=orig_name)


def extract_calls(language: Language, body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for _, caps in run_query(language, "(call_expression function: (_) @fn) @call", body_node):
        fn_node = caps["fn"][0]
        call_node = caps["call"][0]
        raw_text = node_text(call_node, source_bytes)

        if fn_node.type == "identifier":
            call_sites.append(CallSite(callee_name=node_text(fn_node, source_bytes), receiver=None, raw_text=raw_text))
        elif fn_node.type == "member_expression":
            object_node = fn_node.child_by_field_name("object")
            property_node = fn_node.child_by_field_name("property")
            if property_node is None:
                continue
            receiver = node_text(object_node, source_bytes) if object_node is not None else None
            call_sites.append(
                CallSite(callee_name=node_text(property_node, source_bytes), receiver=receiver, raw_text=raw_text)
            )
    return call_sites


# --- setRequestHandler(ListToolsRequestSchema, ...) low-level pattern -----

VALUE_QUERY = """
(program [
  (lexical_declaration (variable_declarator name: (identifier) @name value: (_) @value))
  (export_statement declaration: (lexical_declaration (variable_declarator name: (identifier) @name value: (_) @value)))
])
"""


def extract_values(language: Language, root: Node, source_bytes: bytes, rel_path: str) -> list[ValueDef]:
    """Module/program-top-level `const`/`let` bindings only (including
    `export const ...`) -- verified live that this excludes declarations
    nested inside a function body, matching definition_index.py's own
    "module-level only" scoping for the same reason (a locally-scoped
    binding isn't reachable by bare-name reference the way a top-level one
    is).
    """
    values: list[ValueDef] = []
    for _, caps in run_query(language, VALUE_QUERY, root):
        name = node_text(caps["name"][0], source_bytes)
        values.append(ValueDef(bare_name=name, file=rel_path, value_node=caps["value"][0]))
    return values


def synthetic_list_tools_handler_name(rel_path: str, start_line: int) -> str:
    """Like synthetic_handler_name(), but for setRequestHandler(...) sites,
    where one call site produces N tools sharing one handler -- keyed by
    location instead of a per-tool name."""
    return f"<inline_list_tools_handler>:{rel_path}:{start_line}"


def _get_body_expr(node: Node) -> Node | None:
    """Given an arrow_function/function_expression, returns the single
    expression it evaluates to: the inner expression of a
    parenthesized_expression body (`() => ({...})`), the value of the first
    top-level return in a statement_block body (`() => { return {...}; }`),
    or the body itself for any other (bare-expression) form -- verified
    live for both confirmed real shapes."""
    body = node.child_by_field_name("body")
    if body is None:
        return None
    if body.type == "parenthesized_expression":
        inner = [c for c in body.children if c.type not in ("(", ")")]
        return inner[0] if inner else None
    if body.type == "statement_block":
        for child in body.children:
            if child.type == "return_statement":
                value_children = [c for c in child.children if c.type != "return"]
                return value_children[0] if value_children else None
        return None
    return body


def _object_field_node_or_shorthand(object_node: Node, key: str, source_bytes: bytes) -> Node | None:
    """Like _object_field_node, but also recognizes ES2015 shorthand
    property syntax (`return { tools }`, equivalent to `{ tools: tools }`)
    -- real case (Flux159/mcp-server-kubernetes) a pair-only lookup misses
    entirely."""
    for child in object_node.children:
        if child.type == "pair":
            key_node = child.child_by_field_name("key")
            value_node = child.child_by_field_name("value")
            if key_node is not None and value_node is not None and node_text(key_node, source_bytes) == key:
                return value_node
        elif child.type == "shorthand_property_identifier" and node_text(child, source_bytes) == key:
            return child  # the identifier itself doubles as the value reference
    return None


def _sole_param_name(arrow_node: Node, source_bytes: bytes) -> str | None:
    bare_param = arrow_node.child_by_field_name("parameter")
    if bare_param is not None and bare_param.type == "identifier":
        return node_text(bare_param, source_bytes)
    params_node = arrow_node.child_by_field_name("parameters")
    if params_node is not None:
        param_nodes = [c for c in params_node.children if c.type not in ("(", ")", ",")]
        if len(param_nodes) == 1:
            inner = param_nodes[0]
            if inner.type == "identifier":
                return node_text(inner, source_bytes)
            if inner.type == "required_parameter":
                ident = next((c for c in inner.children if c.type == "identifier"), None)
                if ident is not None:
                    return node_text(ident, source_bytes)
    return None


def _sole_arrow_callback(args_node: Node | None) -> Node | None:
    if args_node is None:
        return None
    arg_nodes = [c for c in args_node.children if c.type not in ("(", ")", ",")]
    if len(arg_nodes) != 1 or arg_nodes[0].type not in ("arrow_function", "function_expression"):
        return None
    return arg_nodes[0]


def _verify_safe_map_callback(callback: Node, source_bytes: bytes) -> Node | None:
    """A `.map(callback)` unwrap is only safe (per the plan's explicit
    trade-off) when the callback's shape is verified to not rename which
    tool a `name` refers to -- single bare parameter, returns an object
    whose `name:` is exactly `<param>.name`. Returns the callback's own
    `description:` value node (applied uniformly to every element this
    call site produces, per the accepted korean-law-mcp trade-off) if
    verified safe, else None -- callers must give up on the whole .map(),
    not guess.
    """
    param_name = _sole_param_name(callback, source_bytes)
    if param_name is None:
        return None
    body_expr = _get_body_expr(callback)
    if body_expr is None or body_expr.type != "object":
        return None
    name_value = _object_field_node_or_shorthand(body_expr, "name", source_bytes)
    if name_value is None or name_value.type != "member_expression":
        return None
    name_obj = name_value.child_by_field_name("object")
    name_prop = name_value.child_by_field_name("property")
    if name_obj is None or name_prop is None:
        return None
    if name_obj.type != "identifier" or node_text(name_obj, source_bytes) != param_name:
        return None
    if node_text(name_prop, source_bytes) != "name":
        return None
    return _object_field_node_or_shorthand(body_expr, "description", source_bytes)


def _find_local_const_value(block_node: Node, name: str, source_bytes: bytes) -> Node | None:
    """Searches a statement_block's own top-level statements for a
    `const <name> = <expr>` (or `let`), returning <expr>'s node if found.
    Real case (Flux159/mcp-server-kubernetes): `const baseTools =
    allowOnlyReadonlyTools ? readonlyTools : allTools;` declared inside the
    setRequestHandler handler itself and used later in that same handler --
    not resolvable via the module-level ValueIndex, since it's
    function-local, not module-level.
    """
    for child in block_node.children:
        if child.type != "lexical_declaration":
            continue
        for decl in child.children:
            if decl.type != "variable_declarator":
                continue
            name_node = decl.child_by_field_name("name")
            value_node = decl.child_by_field_name("value")
            if name_node is not None and value_node is not None and node_text(name_node, source_bytes) == name:
                return value_node
    return None


def _unwrap_ts_list_expr(
    language: Language,
    node: Node,
    current_file: str,
    source_bytes: bytes,
    values: ValueIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
    hops_remaining: int,
    local_scope: Node | None = None,
) -> list[tuple[Node, str, bytes, Node | None]]:
    """Phase A: locate every literal `array` expression a setRequestHandler
    handler (or a local/same-file/imported const it references, optionally
    through a trailing .filter()/.map() chain, or a ternary between two
    such paths) ultimately evaluates to.

    Returns a list of (array_node, its_file, its_source_bytes,
    map_description_override_node_or_None) tuples -- normally 0 or 1, but
    more than 1 when a ternary is encountered (real case,
    Flux159/mcp-server-kubernetes above: each branch contributes its own
    resolved array, unioned by the caller rather than picked arbitrarily,
    since which branch fires depends on runtime config this static pass
    can't evaluate). An empty list means "give up cleanly", never guess.

    `local_scope`, when given, is the handler's own statement_block --
    checked before the module-level ValueIndex when resolving a bare name,
    since a function-local const isn't in that index at all.
    """
    if node.type == "array":
        return [(node, current_file, source_bytes, None)]

    if node.type == "ternary_expression":
        consequence = node.child_by_field_name("consequence")
        alternative = node.child_by_field_name("alternative")
        results: list[tuple[Node, str, bytes, Node | None]] = []
        for branch in (consequence, alternative):
            if branch is None:
                continue
            results.extend(
                _unwrap_ts_list_expr(
                    language, branch, current_file, source_bytes,
                    values, imports_by_file, source_bytes_by_file, hops_remaining, local_scope,
                )
            )
        return results

    if hops_remaining <= 0:
        return []

    if node.type in ("identifier", "shorthand_property_identifier"):
        name = node_text(node, source_bytes)
        if local_scope is not None:
            local_value = _find_local_const_value(local_scope, name, source_bytes)
            if local_value is not None:
                return _unwrap_ts_list_expr(
                    language, local_value, current_file, source_bytes,
                    values, imports_by_file, source_bytes_by_file, hops_remaining - 1, local_scope,
                )
        resolved, _ambiguous = resolve_value(name, current_file, values, imports_by_file)
        if resolved is None:
            return []
        return _unwrap_ts_list_expr(
            language, resolved.value_node, resolved.file, source_bytes_by_file[resolved.file],
            values, imports_by_file, source_bytes_by_file, hops_remaining - 1, local_scope,
        )

    if node.type == "call_expression":
        function_node = node.child_by_field_name("function")
        if function_node is None or function_node.type != "member_expression":
            return []
        property_node = function_node.child_by_field_name("property")
        object_node = function_node.child_by_field_name("object")
        if property_node is None or object_node is None:
            return []
        method = node_text(property_node, source_bytes)

        if method == "filter":
            # A filter only removes elements, never renames a survivor's
            # name/description -- always safe to pass through structurally
            # without evaluating the predicate (accepted over-reporting
            # trade-off: elements filtered out at runtime are still counted).
            return _unwrap_ts_list_expr(
                language, object_node, current_file, source_bytes,
                values, imports_by_file, source_bytes_by_file, hops_remaining - 1, local_scope,
            )

        if method == "map":
            args_node = node.child_by_field_name("arguments")
            callback = _sole_arrow_callback(args_node)
            if callback is None:
                return []
            override = _verify_safe_map_callback(callback, source_bytes)
            if override is None:
                return []  # callback shape not verified safe -- give up, don't guess
            results = _unwrap_ts_list_expr(
                language, object_node, current_file, source_bytes,
                values, imports_by_file, source_bytes_by_file, hops_remaining - 1, local_scope,
            )
            return [(arr, f, sb, override) for (arr, f, sb, _prior_override) in results]

        return []

    return []


def _resolve_ts_list_element(
    node: Node,
    current_file: str,
    source_bytes: bytes,
    values: ValueIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
) -> tuple[Node, bytes] | None:
    """Phase B: resolve one array element to a tool-shaped object literal.
    Separate, small per-element cap (same-file, then at most 1 import hop)
    independent of Phase A's budget. An element that fails is skipped
    individually, not treated as aborting the whole array."""
    if node.type == "object":
        return node, source_bytes

    if node.type == "identifier":
        name = node_text(node, source_bytes)
        resolved, _ambiguous = resolve_value(name, current_file, values, imports_by_file)
        if resolved is None:
            return None
        resolved_node = resolved.value_node
        if resolved_node.type == "object":
            return resolved_node, source_bytes_by_file[resolved.file]
        return None

    return None


SET_REQUEST_HANDLER_QUERY = """
(call_expression
  function: (member_expression
    property: (property_identifier) @method_name)
  arguments: (arguments) @args) @call
"""


def _handler_returns_tools_field(handler: Node, source_bytes: bytes) -> Node | None:
    handler_body_expr = _get_body_expr(handler)
    if handler_body_expr is None or handler_body_expr.type != "object":
        return None
    return _object_field_node_or_shorthand(handler_body_expr, "tools", source_bytes)


def detect_lowlevel_set_request_handler(
    language: Language,
    language_tag: str,
    root: Node,
    source_bytes: bytes,
    rel_path: str,
    definitions: DefinitionIndex,
    values: ValueIndex,
    imports_by_file: dict[str, ImportIndex],
    source_bytes_by_file: dict[str, bytes],
) -> list[ToolRecord]:
    """Detects the official MCP TS/JS SDK's low-level
    `<server>.setRequestHandler(ListToolsRequestSchema, handler)` API --
    the alternative to the already-detected high-level `.tool()`/
    `.registerTool()` sugar. Confirmed in 24 of 206 repos in the Etapa 2
    pilot corpus, 20 with zero overlap with the high-level pattern.

    The receiver is deliberately unconstrained (real cases include
    `target.server.setRequestHandler(...)` and
    `this.server.setRequestHandler(...)`, not just a bare `server`).

    Same structural trade-off as the Python low-level pattern: the
    resulting call graph's level 1 is the ListToolsRequestSchema handler
    itself (metadata/filter/map code), not the tool's CallToolRequestSchema
    execution dispatch, which this pattern does not attempt to link to.
    """
    tools: list[ToolRecord] = []
    for _, caps in run_query(language, SET_REQUEST_HANDLER_QUERY, root):
        if node_text(caps["method_name"][0], source_bytes) != "setRequestHandler":
            continue

        args_node = caps["args"][0]
        arg_nodes = [c for c in args_node.children if c.type not in ("(", ")", ",")]
        if len(arg_nodes) < 2:
            continue
        schema_arg, handler_arg = arg_nodes[0], arg_nodes[1]
        schema_ok = (schema_arg.type == "identifier" and node_text(schema_arg, source_bytes) == "ListToolsRequestSchema") or (
            _string_value(schema_arg, source_bytes) == "tools/list"
        )
        if not schema_ok:
            continue
        if handler_arg.type not in ("arrow_function", "function_expression"):
            continue

        tools_field = _handler_returns_tools_field(handler_arg, source_bytes)
        if tools_field is None:
            continue

        start_line, end_line = line_range(caps["call"][0])
        handler_body = handler_arg.child_by_field_name("body")
        local_scope = handler_body if handler_body is not None and handler_body.type == "statement_block" else None
        located = _unwrap_ts_list_expr(
            language, tools_field, rel_path, source_bytes,
            values, imports_by_file, source_bytes_by_file, MAX_VALUE_RESOLUTION_HOPS, local_scope,
        )
        if not located:
            continue

        qualified_name = synthetic_list_tools_handler_name(rel_path, start_line)
        seen_names: set[str] = set()  # a ternary can union the same tool from both branches
        for array_node, array_file, array_source_bytes, description_override in located:
            for element in array_node.children:
                if element.type in ("[", "]", ","):
                    continue
                resolved = _resolve_ts_list_element(
                    element, array_file, array_source_bytes, values, imports_by_file, source_bytes_by_file
                )
                if resolved is None:
                    continue
                tool_obj, tool_source_bytes = resolved

                tool_name, tool_name_is_literal = _object_field(tool_obj, "name", tool_source_bytes)
                if tool_name is None or not tool_name_is_literal:
                    continue  # no literal name -- skip rather than fabricate
                if tool_name in seen_names:
                    continue
                seen_names.add(tool_name)

                if description_override is not None:
                    description = _string_value(description_override, source_bytes)
                    description_is_literal = description is not None
                    if description is None:
                        description = node_text(description_override, source_bytes)
                else:
                    description, description_is_literal = _object_field(tool_obj, "description", tool_source_bytes)

                tools.append(
                    ToolRecord(
                        name=tool_name,
                        description=description or "",
                        description_is_literal=description_is_literal,
                        sdk_pattern=f"{language_tag}.set_request_handler_lowlevel",
                        source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                        qualified_name=qualified_name,
                    )
                )
    return tools


def register_lowlevel_list_tools_definitions(
    language: Language, root: Node, source_bytes: bytes, rel_path: str
) -> list[FunctionDef]:
    """Registers a synthetic FunctionDef for each setRequestHandler(...)
    handler site, mirroring the existing loop that does this for
    .tool()/.registerTool() inline handlers -- needed so
    detect_tools_with_call_graphs() can look each tool's qualified_name up
    and build its call graph."""
    defs: list[FunctionDef] = []
    for _, caps in run_query(language, SET_REQUEST_HANDLER_QUERY, root):
        if node_text(caps["method_name"][0], source_bytes) != "setRequestHandler":
            continue
        args_node = caps["args"][0]
        arg_nodes = [c for c in args_node.children if c.type not in ("(", ")", ",")]
        if len(arg_nodes) < 2:
            continue
        schema_arg, handler_arg = arg_nodes[0], arg_nodes[1]
        schema_ok = (schema_arg.type == "identifier" and node_text(schema_arg, source_bytes) == "ListToolsRequestSchema") or (
            _string_value(schema_arg, source_bytes) == "tools/list"
        )
        if not schema_ok or handler_arg.type not in ("arrow_function", "function_expression"):
            continue

        body = handler_arg.child_by_field_name("body")
        if body is None:
            continue
        # Unlike the .tool()/.registerTool()/addTool execution handlers
        # (which do real work and are almost always statement_block-bodied,
        # so requiring that shape there costs little real coverage), this
        # handler only returns metadata -- the terse expression form
        # (`() => ({ tools: [...] })`) is the DOMINANT real-world style for
        # it, not a rare edge case, so both shapes are registered here.
        # extract_calls()'s query works over any node, not just
        # statement_block, so this doesn't lose call-graph coverage either.

        start_line, _end_line = line_range(caps["call"][0])
        synthetic_name = synthetic_list_tools_handler_name(rel_path, start_line)
        handler_start, handler_end = line_range(handler_arg)
        defs.append(
            FunctionDef(
                qualified_name=synthetic_name, bare_name=synthetic_name, file=rel_path,
                start_line=handler_start, end_line=handler_end, body_node=body, class_name=None,
            )
        )
    return defs
