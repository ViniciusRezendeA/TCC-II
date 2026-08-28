from __future__ import annotations

import re
from dataclasses import dataclass

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import CallSite
from mcp_pipeline.extraction.definition_index import FunctionDef
from mcp_pipeline.extraction.import_index import ImportIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.models import SourceLocation, ToolRecord
from mcp_pipeline.extraction.parser_utils import line_range, node_text, run_query

KOTLIN_LANGUAGE = spec_for("Kotlin").ts_language

_STRING_NODE_TYPES = ("string_literal", "multiline_string_literal")

# tree-sitter-kotlin (1.1.0) doesn't expose Kotlin's `$name`/`${expr}` string
# interpolation as its own node type -- confirmed empirically: `"interp
# $name here"` parses to three flat `string_content` siblings ('interp ',
# '$', 'name here'), the '$' and what follows it aren't distinguished from
# literal text at all. Concatenating string_content children therefore
# reconstructs the source text losslessly but can't tell "literal $" from
# "interpolation" via node type the way every other language here can. This
# regex is a text-level stand-in for that missing structural signal, same
# purpose as the interpolation checks in ruby_patterns.py/ecmascript_common
# (falls back to raw text, description_is_literal=False) -- a bare digit or
# punctuation after `$` isn't valid Kotlin interpolation syntax, so it's
# deliberately not matched (`$5` is really just a dollar sign).
_INTERPOLATION_RE = re.compile(r"(?<!\\)\$(\{|[A-Za-z_])")


def _kotlin_string_value(node: Node, source_bytes: bytes) -> str | None:
    if node.type not in _STRING_NODE_TYPES:
        return None
    content_parts = [c for c in node.children if c.type == "string_content"]
    text = "".join(node_text(c, source_bytes) for c in content_parts)
    if _INTERPOLATION_RE.search(text):
        return None
    return text


def _synthetic_handler_name(rel_path: str, tool_name: str) -> str:
    """Same convention as ecmascript_common.py's/dart_patterns.py's
    synthetic_handler_name -- an inline trailing-lambda handler isn't a
    named definition, so extract_definitions and detect_kotlin_tools need a
    shared synthetic key to link the tool record back to its body."""
    return f"<inline_tool_handler>:{rel_path}:{tool_name}"


def _enclosing_class_name(fn_node: Node, source_bytes: bytes) -> str | None:
    container = fn_node.parent
    if container is None or container.type != "class_body" or container.parent is None:
        return None
    class_node = container.parent
    if class_node.type != "class_declaration":
        return None
    name_node = class_node.child_by_field_name("name")
    return node_text(name_node, source_bytes) if name_node is not None else None


def _function_body_node(fn_node: Node) -> Node | None:
    # function_declaration has no `body` field in this grammar version
    # (child_by_field_name("body") returns None even though the child
    # exists -- verified empirically), so this is a positional lookup.
    return next((c for c in fn_node.children if c.type == "function_body"), None)


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    for _, caps in run_query(KOTLIN_LANGUAGE, "(function_declaration name: (identifier) @name) @funcdef", root):
        fn_node = caps["funcdef"][0]
        body = _function_body_node(fn_node)
        if body is None:
            continue
        bare_name = node_text(caps["name"][0], source_bytes)
        class_name = _enclosing_class_name(fn_node, source_bytes)
        qualified_name = f"{class_name}.{bare_name}" if class_name else bare_name
        start_line, end_line = line_range(fn_node)
        defs.append(
            FunctionDef(
                qualified_name=qualified_name, bare_name=bare_name, file=rel_path,
                start_line=start_line, end_line=end_line, body_node=body, class_name=class_name,
            )
        )

    for match in _find_add_tool_calls(root, source_bytes):
        synthetic_name = _synthetic_handler_name(rel_path, match.tool_name)
        defs.append(
            FunctionDef(
                qualified_name=synthetic_name, bare_name=synthetic_name, file=rel_path,
                start_line=match.handler_lambda.start_point[0] + 1, end_line=match.handler_lambda.end_point[0] + 1,
                body_node=match.handler_lambda, class_name=None,
            )
        )
    return defs


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    """Kotlin's `import foo.bar.Baz` binds `Baz` (or an `as`-aliased name)
    globally within the file, but MCP tool call sites resolved here are
    almost always same-class/same-file (the SDK builder pattern keeps tool
    registration and handler lambdas colocated) -- same non-goal as
    ruby_patterns.py/dart_patterns.py's extract_imports: call resolution
    falls back to same-class/same-file/repo-wide bare-name matching in
    call_graph_builder.resolve_call.
    """
    return {}


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for _, caps in run_query(KOTLIN_LANGUAGE, "(call_expression) @call", body_node):
        call_node = caps["call"][0]
        if not call_node.children:
            continue
        first = call_node.children[0]
        raw_text = node_text(call_node, source_bytes)
        if first.type == "identifier":
            call_sites.append(CallSite(callee_name=node_text(first, source_bytes), receiver=None, raw_text=raw_text))
        elif first.type == "navigation_expression" and len(first.children) >= 3:
            receiver_node = first.children[0]
            member_node = first.children[-1]
            if member_node.type != "identifier":
                continue
            call_sites.append(
                CallSite(
                    callee_name=node_text(member_node, source_bytes),
                    receiver=node_text(receiver_node, source_bytes),
                    raw_text=raw_text,
                )
            )
    return call_sites


@dataclass
class _AddToolMatch:
    tool_name: str
    description: str | None
    description_is_literal: bool
    call_node: Node  # the `addTool(...)` call itself, for source_location
    handler_lambda: Node  # lambda_literal


def _named_value_arg(value_arguments: Node, key: str, source_bytes: bytes) -> tuple[str | None, bool]:
    for child in value_arguments.children:
        if child.type != "value_argument" or len(child.children) < 3:
            continue
        name_node, eq_node, value_node = child.children[0], child.children[1], child.children[2]
        if name_node.type != "identifier" or eq_node.type != "=" or node_text(name_node, source_bytes) != key:
            continue
        literal = _kotlin_string_value(value_node, source_bytes)
        if literal is not None:
            return literal, True
        return node_text(value_node, source_bytes), False
    return None, True


def _named_lambda_arg(value_arguments: Node, key: str, source_bytes: bytes) -> Node | None:
    for child in value_arguments.children:
        if child.type != "value_argument" or len(child.children) < 3:
            continue
        name_node, eq_node, value_node = child.children[0], child.children[1], child.children[2]
        if name_node.type != "identifier" or eq_node.type != "=" or node_text(name_node, source_bytes) != key:
            continue
        return value_node if value_node.type == "lambda_literal" else None
    return None


def _match_add_tool(call_node: Node, source_bytes: bytes) -> _AddToolMatch | None:
    """Detects the official `kotlin-sdk`'s `addTool(name = "...",
    description = "...", ...) { request -> ... }` pattern -- verified
    against real parses in modelcontextprotocol/kotlin-sdk's own samples
    and 2 independent real MCP-server repos (six2dez/burp-ai-agent,
    platonai/Browser4), both depending on the official SDK. `addTool` is
    called as a bare (receiver-less) identifier in every real case found --
    it's an extension function invoked inside a server-builder lambda with
    an implicit receiver, not `server.addTool(...)`.

    The handler is a lambda, in one of two real shapes: Kotlin's trailing-
    lambda sugar (`addTool(...) { request -> ... }`, which the grammar
    desugars into an *outer* call_expression wrapping [this inner
    call_expression, an annotated_lambda] -- checked via call_node's
    parent) or an explicit `handler = { request -> ... }` named argument
    (six2dez's style) inside call_node's own value_arguments.
    """
    value_args = next((c for c in call_node.children if c.type == "value_arguments"), None)
    if value_args is None:
        return None

    tool_name, tool_name_is_literal = _named_value_arg(value_args, "name", source_bytes)
    if tool_name is None or not tool_name_is_literal:
        return None
    description, description_is_literal = _named_value_arg(value_args, "description", source_bytes)

    handler_lambda: Node | None = None
    parent = call_node.parent
    if parent is not None and parent.type == "call_expression" and parent.children and parent.children[0] == call_node:
        annotated = next((c for c in parent.children if c.type == "annotated_lambda"), None)
        if annotated is not None:
            handler_lambda = next((c for c in annotated.children if c.type == "lambda_literal"), None)
    if handler_lambda is None:
        handler_lambda = _named_lambda_arg(value_args, "handler", source_bytes)
    if handler_lambda is None:
        return None

    return _AddToolMatch(
        tool_name=tool_name, description=description, description_is_literal=description_is_literal,
        call_node=call_node, handler_lambda=handler_lambda,
    )


def _find_add_tool_calls(root: Node, source_bytes: bytes) -> list[_AddToolMatch]:
    matches: list[_AddToolMatch] = []
    _walk_add_tool_calls(root, source_bytes, matches)
    return matches


def _walk_add_tool_calls(node: Node, source_bytes: bytes, out: list[_AddToolMatch]) -> None:
    if (
        node.type == "call_expression"
        and node.children
        and node.children[0].type == "identifier"
        and node_text(node.children[0], source_bytes) == "addTool"
    ):
        match = _match_add_tool(node, source_bytes)
        if match is not None:
            out.append(match)
    for child in node.children:
        _walk_add_tool_calls(child, source_bytes, out)


def detect_kotlin_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    tools: list[ToolRecord] = []
    for match in _find_add_tool_calls(root, source_bytes):
        start_line, end_line = line_range(match.call_node)
        tools.append(
            ToolRecord(
                name=match.tool_name,
                description=match.description or "",
                description_is_literal=match.description_is_literal,
                sdk_pattern="kotlin.kotlinsdk_addtool",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=_synthetic_handler_name(rel_path, match.tool_name),
            )
        )
    return tools
