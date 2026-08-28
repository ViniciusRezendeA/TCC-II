from __future__ import annotations

from tree_sitter import Node

from mcp_pipeline.extraction.call_graph_builder import CallSite
from mcp_pipeline.extraction.definition_index import FunctionDef
from mcp_pipeline.extraction.import_index import ImportIndex
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.models import SourceLocation, ToolRecord
from mcp_pipeline.extraction.parser_utils import (
    line_range,
    node_text,
    run_query,
    string_literal_value,
)

RUBY_LANGUAGE = spec_for("Ruby").ts_language


def _ruby_string_value(node: Node, source_bytes: bytes) -> str | None:
    """Like parser_utils.string_literal_value, but first rejects a string
    with `#{...}` interpolation -- naively joining only its `string_content`
    children would silently drop the interpolated part and fabricate a
    plausible-but-wrong literal (the same trap template_string_literal_value
    guards against for JS/TS template strings)."""
    if node.type != "string":
        return None
    if any(c.type == "interpolation" for c in node.children):
        return None
    return string_literal_value(node, source_bytes, content_node_type="string_content", node_type="string")


def _enclosing_container_name(fn_node: Node, source_bytes: bytes) -> str | None:
    """A method/singleton_method's container (class OR module) is its
    grandparent via a body_statement wrapper -- confirmed present even for a
    single-statement body, unlike C#'s direct attribute_list-on-node shape."""
    container = fn_node.parent
    if container is None or container.type != "body_statement" or container.parent is None:
        return None
    owner = container.parent
    if owner.type not in ("class", "module"):
        return None
    name_node = owner.child_by_field_name("name")
    return node_text(name_node, source_bytes) if name_node is not None else None


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    for node_type in ("method", "singleton_method"):
        for _, caps in run_query(
            RUBY_LANGUAGE,
            f"({node_type} name: (identifier) @name body: (body_statement) @body) @funcdef",
            root,
        ):
            fn_node = caps["funcdef"][0]
            bare_name = node_text(caps["name"][0], source_bytes)
            class_name = _enclosing_container_name(fn_node, source_bytes)
            qualified_name = f"{class_name}.{bare_name}" if class_name else bare_name
            start_line, end_line = line_range(fn_node)
            defs.append(
                FunctionDef(
                    qualified_name=qualified_name, bare_name=bare_name, file=rel_path,
                    start_line=start_line, end_line=end_line, body_node=caps["body"][0], class_name=class_name,
                )
            )
    return defs


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    """Ruby has no import-with-alias syntax -- `require`/`require_relative`
    are plain method calls that load a file and make its globally-namespaced
    constants available everywhere, they don't bind a local name the way
    Python's `import`/JS's `import`/Rust's `use` do. There is nothing to
    populate here; call resolution falls back to same-class (tier 1),
    same-file (tier 2), and repo-wide bare-name (tier 4) matching in
    call_graph_builder.resolve_call, same as every other language when tier
    3 (import-alias) has nothing to offer.
    """
    return {}


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for _, caps in run_query(RUBY_LANGUAGE, "(call method: (identifier) @name) @call", body_node):
        call_node = caps["call"][0]
        name_node = caps["name"][0]
        receiver_node = call_node.child_by_field_name("receiver")
        receiver = node_text(receiver_node, source_bytes) if receiver_node is not None else None
        call_sites.append(
            CallSite(callee_name=node_text(name_node, source_bytes), receiver=receiver, raw_text=node_text(call_node, source_bytes))
        )
    return call_sites


def _superclass_tail_name(superclass_node: Node, source_bytes: bytes) -> str | None:
    expr = next((c for c in superclass_node.children if c.type != "<"), None)
    if expr is None:
        return None
    if expr.type == "scope_resolution":
        name_node = expr.child_by_field_name("name")
        return node_text(name_node, source_bytes) if name_node is not None else None
    if expr.type == "constant":
        return node_text(expr, source_bytes)
    return None


def _find_call_method(body_node: Node, source_bytes: bytes) -> Node | None:
    for child in body_node.children:
        if child.type == "method":
            name_node = child.child_by_field_name("name")
            if name_node is not None and node_text(name_node, source_bytes) == "call":
                return child
    return None


def _class_dsl_string_arg(body_node: Node, source_bytes: bytes, key: str) -> tuple[str | None, bool]:
    """Reads a class-body-level `key 'value'` (or `key('value')` -- both
    forms produce the same `call` node shape in tree-sitter-ruby) DSL
    call, e.g. `description 'Greet someone by name'`. Only looks at
    body_node's *direct* children (the class-body statement list), not
    nested calls inside `def call`/`arguments do ... end` blocks, so a
    same-named method/local call deeper in the class can't be
    mismatched for the DSL setter.
    """
    for child in body_node.children:
        if child.type != "call":
            continue
        method_node = child.child_by_field_name("method")
        if method_node is None or node_text(method_node, source_bytes) != key:
            continue
        args_node = child.child_by_field_name("arguments")
        if args_node is None or not args_node.named_children:
            continue
        value_node = args_node.named_children[0]
        literal = _ruby_string_value(value_node, source_bytes)
        if literal is not None:
            return literal, True
        return node_text(value_node, source_bytes), False
    return None, True


def detect_ruby_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    """Detects the `fast-mcp` gem's dominant class-based pattern: a class
    whose superclass chain ends in `...Tool` (`FastMcp::Tool` directly, or
    an app-level `ApplicationTool < FastMcp::Tool` indirection -- the
    standard Rails-generator convention, mirroring `ApplicationRecord`/
    `ApplicationController`) that overrides `call`. Verified against real
    parses in yjacquin/fast-mcp itself (the framework's own generator
    templates, `lib/generators/fast_mcp/install/templates/sample_tool.rb`)
    -- the `call` method IS the tool's real implementation (unlike Swift's
    SDK, deliberately not implemented -- see the plan), so it doubles as
    the call graph's root definition. Requiring both the superclass-name
    suffix AND a `call` override (rather than either alone) keeps this from
    matching an unrelated same-repo class that merely happens to be named
    `...Tool`.
    """
    tools: list[ToolRecord] = []
    for _, caps in run_query(
        RUBY_LANGUAGE,
        "(class name: (constant) @name superclass: (superclass) @super body: (body_statement) @body) @classdef",
        root,
    ):
        tail = _superclass_tail_name(caps["super"][0], source_bytes)
        if tail is None or not tail.endswith("Tool"):
            continue

        body_node = caps["body"][0]
        call_method = _find_call_method(body_node, source_bytes)
        if call_method is None:
            continue

        class_name = node_text(caps["name"][0], source_bytes)
        name_override, _ = _class_dsl_string_arg(body_node, source_bytes, "tool_name")
        description, description_is_literal = _class_dsl_string_arg(body_node, source_bytes, "description")

        start_line, end_line = line_range(call_method)
        tools.append(
            ToolRecord(
                name=name_override or class_name,
                description=description or "",
                description_is_literal=description_is_literal,
                sdk_pattern="ruby.fastmcp_tool_class",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=f"{class_name}.call",
            )
        )
    return tools
