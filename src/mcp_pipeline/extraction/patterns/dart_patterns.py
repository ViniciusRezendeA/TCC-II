from __future__ import annotations

from dataclasses import dataclass

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import CallSite
from mcp_pipeline.extraction.definition_index import FunctionDef
from mcp_pipeline.extraction.import_index import ImportIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.models import SourceLocation, ToolRecord
from mcp_pipeline.extraction.parser_utils import node_text

DART_LANGUAGE = spec_for("Dart").ts_language

# tree-sitter-dart (0.1.0) defines no named fields on any of the nodes this
# module cares about (call_by_field_name is None everywhere -- confirmed
# empirically, not assumed) -- unlike every other language here, so this
# module walks by node *type* and sibling position instead of run_query +
# field: constraints. Two grammar quirks drive most of the shape below:
# 1. A call expression isn't one node -- `foo(x)` is a flat sibling run
#    [identifier 'foo', selector '(x)'], and `a.b(x)` is
#    [identifier 'a', selector '.b', selector '(x)']. A "call" is
#    recognized by finding a `selector` whose own first child is
#    `argument_part`, then looking at *its* previous sibling.
# 2. A method isn't one node either -- `void f() {...}` is two adjacent
#    siblings, `method_signature` (or bare `function_signature` for a
#    top-level function) followed by `function_body`.


def _synthetic_handler_name(rel_path: str, tool_name: str) -> str:
    """Same convention as ecmascript_common.py's synthetic_handler_name (not
    imported from there -- Dart's grammar is unrelated) for the identical
    reason: an inline closure passed as `registerTool`'s 2nd argument isn't
    a named definition, so both extract_definitions and detect_dart_tools
    need a shared synthetic key to link the tool record back to its body."""
    return f"<inline_tool_handler>:{rel_path}:{tool_name}"


def _dart_string_value(node: Node, source_bytes: bytes) -> str | None:
    """tree-sitter-dart's `string_literal` has no content child at all for a
    non-interpolated string (verified empirically: `"x"` parses to just two
    quote-token children, nothing in between) -- the actual text lives in
    the *byte gap* between the opening and closing delimiter tokens, which
    also transparently handles triple-quoted (`'''...'''`) and raw (`r'...'`)
    strings without hardcoding delimiter lengths. Returns None (falls back
    to raw text upstream) for any interpolated string, e.g. `"$x"`/`"${x}"`.
    """
    if node.type != "string_literal" or node.child_count < 2:
        return None
    if any(c.type == "template_substitution" for c in node.children):
        return None
    children = node.children
    return source_bytes[children[0].end_byte : children[-1].start_byte].decode("utf-8", errors="replace")


def _class_definition_name(class_def: Node, source_bytes: bytes) -> str | None:
    name_node = next((c for c in class_def.children if c.type == "identifier"), None)
    return node_text(name_node, source_bytes) if name_node is not None else None


def _enclosing_class_name(node: Node, source_bytes: bytes) -> str | None:
    current = node.parent
    while current is not None:
        if current.type == "class_definition":
            return _class_definition_name(current, source_bytes)
        current = current.parent
    return None


def _signature_name(function_signature: Node, source_bytes: bytes) -> str | None:
    children = function_signature.children
    for i, c in enumerate(children):
        if c.type == "formal_parameter_list" and i > 0 and children[i - 1].type == "identifier":
            return node_text(children[i - 1], source_bytes)
    return None


def _body_block(function_body: Node) -> Node | None:
    for c in function_body.children:
        if c.type == "block":
            return c
        if c.type == "function_expression_body":
            return _body_block(c)
    return None


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    _walk_definitions(root, source_bytes, rel_path, defs)
    for match in _find_register_tool_calls(root, source_bytes):
        if match.handler_kind != "inline":
            continue
        body = _body_block(match.handler_node)
        if body is None:
            continue
        synthetic_name = _synthetic_handler_name(rel_path, match.tool_name)
        defs.append(
            FunctionDef(
                qualified_name=synthetic_name, bare_name=synthetic_name, file=rel_path,
                start_line=match.handler_node.start_point[0] + 1, end_line=match.handler_node.end_point[0] + 1,
                body_node=body, class_name=None,
            )
        )
    return defs


def _walk_definitions(node: Node, source_bytes: bytes, rel_path: str, out: list[FunctionDef]) -> None:
    if node.type == "function_signature":
        # Inside a class, `function_body` is a sibling of the *wrapping*
        # method_signature, not of function_signature itself (method_signature
        # has exactly one child: the function_signature) -- confirmed against
        # a real parse. A top-level function has no such wrapper, so
        # function_signature and function_body are direct siblings there.
        in_method = node.parent is not None and node.parent.type == "method_signature"
        container = node.parent if in_method else node
        sibling = container.next_sibling
        name = _signature_name(node, source_bytes)
        if sibling is not None and sibling.type == "function_body" and name is not None:
            body = _body_block(sibling)
            if body is not None:
                class_name = _enclosing_class_name(node, source_bytes) if in_method else None
                qualified_name = f"{class_name}.{name}" if class_name else name
                out.append(
                    FunctionDef(
                        qualified_name=qualified_name, bare_name=name, file=rel_path,
                        start_line=container.start_point[0] + 1, end_line=sibling.end_point[0] + 1,
                        body_node=body, class_name=class_name,
                    )
                )
    for child in node.children:
        _walk_definitions(child, source_bytes, rel_path, out)


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    """Dart's `import 'package:foo/bar.dart';` binds no per-symbol alias the
    way Python/JS/Rust imports do (everything imported becomes globally
    visible, optionally behind a whole-library `as` prefix that isn't a
    per-function alias either) -- there's nothing meaningful to key an
    ImportIndex by here. Same non-goal as ruby_patterns.py's extract_imports:
    call resolution falls back to same-class/same-file/repo-wide bare-name
    matching in call_graph_builder.resolve_call.
    """
    return {}


def _dot_selector_member(selector_node: Node, source_bytes: bytes) -> str | None:
    inner = next((c for c in selector_node.children if c.type == "unconditional_assignable_selector"), None)
    if inner is None:
        return None
    name_node = next((c for c in inner.children if c.type == "identifier"), None)
    return node_text(name_node, source_bytes) if name_node is not None else None


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    _walk_calls(body_node, source_bytes, call_sites)
    return call_sites


def _walk_calls(node: Node, source_bytes: bytes, out: list[CallSite]) -> None:
    if node.type == "selector" and node.child_count > 0 and node.children[0].type == "argument_part":
        callee_sel = node.prev_sibling
        if callee_sel is not None:
            if callee_sel.type == "selector":
                member = _dot_selector_member(callee_sel, source_bytes)
                receiver_node = callee_sel.prev_sibling
                if member is not None and receiver_node is not None:
                    out.append(
                        CallSite(
                            callee_name=member, receiver=node_text(receiver_node, source_bytes),
                            raw_text=source_bytes[receiver_node.start_byte : node.end_byte].decode("utf-8", errors="replace"),
                        )
                    )
            elif callee_sel.type == "identifier":
                out.append(
                    CallSite(
                        callee_name=node_text(callee_sel, source_bytes), receiver=None,
                        raw_text=source_bytes[callee_sel.start_byte : node.end_byte].decode("utf-8", errors="replace"),
                    )
                )
    for child in node.children:
        _walk_calls(child, source_bytes, out)


@dataclass
class _RegisterToolMatch:
    tool_name: str
    description: str | None
    description_is_literal: bool
    call_node: Node  # the outer `selector` (registerTool's own call-parens)
    handler_kind: str  # "inline" | "identifier"
    handler_node: Node  # function_expression (inline) or identifier node


def _named_arg(arguments_node: Node, key: str, source_bytes: bytes) -> tuple[str | None, bool]:
    for child in arguments_node.children:
        if child.type != "named_argument" or child.child_count < 2:
            continue
        label_node = next((c for c in child.children if c.type == "label"), None)
        if label_node is None or node_text(label_node, source_bytes).rstrip(":") != key:
            continue
        value_node = child.children[-1]
        literal = _dart_string_value(value_node, source_bytes)
        if literal is not None:
            return literal, True
        return node_text(value_node, source_bytes), False
    return None, True


def _find_register_tool_calls(root: Node, source_bytes: bytes) -> list[_RegisterToolMatch]:
    matches: list[_RegisterToolMatch] = []
    _walk_register_tool_calls(root, source_bytes, matches)
    return matches


def _walk_register_tool_calls(node: Node, source_bytes: bytes, out: list[_RegisterToolMatch]) -> None:
    if node.type == "selector" and node.child_count > 0 and node.children[0].type == "argument_part":
        callee = node.prev_sibling
        if callee is not None and callee.type == "identifier" and node_text(callee, source_bytes) == "registerTool":
            match = _match_register_tool_call(node, source_bytes)
            if match is not None:
                out.append(match)
    for child in node.children:
        _walk_register_tool_calls(child, source_bytes, out)


def _match_register_tool_call(call_selector: Node, source_bytes: bytes) -> _RegisterToolMatch | None:
    """Detects the official `dart_mcp` package's `registerTool(Tool(name:,
    description:, ...), handler)` pattern -- verified against real parses in
    2 independent real MCP-server repos (leoafarias/fvm, VeryGoodOpenSource/
    very_good_cli), both depending on `package:dart_mcp/server.dart`. Both
    real `handler` forms are handled: an inline closure (`(call) async {
    ... }`, fvm's style) and a bare reference to a sibling method
    (`_handleCreate`, very_good_cli's style) -- the latter is resolved
    against the *call site's* enclosing class, since in both real repos the
    registration itself happens inside a `_registerTools()` method of the
    same class the handler method lives in.
    """
    argument_part = call_selector.children[0]
    arguments = next((c for c in argument_part.children if c.type == "arguments"), None)
    if arguments is None:
        return None
    top_args = [c for c in arguments.children if c.type == "argument"]
    if len(top_args) < 2:
        return None
    tool_arg, handler_arg = top_args[0], top_args[1]

    tool_ctor = next((c for c in tool_arg.children if c.type == "identifier"), None)
    if tool_ctor is None or node_text(tool_ctor, source_bytes) != "Tool":
        return None
    tool_selector = next((c for c in tool_arg.children if c.type == "selector"), None)
    if tool_selector is None:
        return None
    tool_argument_part = next((c for c in tool_selector.children if c.type == "argument_part"), None)
    tool_arguments = (
        next((c for c in tool_argument_part.children if c.type == "arguments"), None) if tool_argument_part else None
    )
    if tool_arguments is None:
        return None

    tool_name, tool_name_is_literal = _named_arg(tool_arguments, "name", source_bytes)
    if tool_name is None or not tool_name_is_literal:
        return None
    description, description_is_literal = _named_arg(tool_arguments, "description", source_bytes)

    if not handler_arg.children:
        return None
    handler_child = handler_arg.children[0]
    if handler_child.type == "function_expression":
        return _RegisterToolMatch(
            tool_name=tool_name, description=description, description_is_literal=description_is_literal,
            call_node=call_selector, handler_kind="inline", handler_node=handler_child,
        )
    if handler_child.type == "identifier":
        return _RegisterToolMatch(
            tool_name=tool_name, description=description, description_is_literal=description_is_literal,
            call_node=call_selector, handler_kind="identifier", handler_node=handler_child,
        )
    return None


def detect_dart_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    tools: list[ToolRecord] = []
    for match in _find_register_tool_calls(root, source_bytes):
        if match.handler_kind == "inline":
            qualified_name = _synthetic_handler_name(rel_path, match.tool_name)
        else:
            class_name = _enclosing_class_name(match.call_node, source_bytes)
            handler_name = node_text(match.handler_node, source_bytes)
            qualified_name = f"{class_name}.{handler_name}" if class_name else handler_name

        tools.append(
            ToolRecord(
                name=match.tool_name,
                description=match.description or "",
                description_is_literal=match.description_is_literal,
                sdk_pattern="dart.dart_mcp_registertool",
                source_location=SourceLocation(
                    file=rel_path, start_line=match.call_node.start_point[0] + 1, end_line=match.call_node.end_point[0] + 1
                ),
                qualified_name=qualified_name,
            )
        )
    return tools
