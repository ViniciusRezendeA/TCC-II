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
from mcp_pipeline.extraction.definition_index import FunctionDef
from mcp_pipeline.extraction.import_index import ImportedName, ImportIndex
from mcp_pipeline.extraction.models import SourceLocation, ToolRecord
from mcp_pipeline.extraction.parser_utils import (
    line_range,
    node_text,
    run_query,
    string_literal_value,
)

TOOL_CALL_QUERY = """
(call_expression
  function: (member_expression
    object: (identifier)
    property: (property_identifier) @method_name)
  arguments: (arguments) @args) @call
"""


def _string_value(node: Node, source_bytes: bytes) -> str | None:
    return string_literal_value(node, source_bytes, content_node_type="string_fragment")


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
