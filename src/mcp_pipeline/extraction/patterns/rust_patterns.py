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

RUST_LANGUAGE = spec_for("Rust").ts_language


def _rust_string_value(node: Node, source_bytes: bytes) -> str | None:
    return string_literal_value(node, source_bytes, content_node_type="string_content", node_type="string_literal")


def extract_definitions(root: Node, source_bytes: bytes, rel_path: str) -> list[FunctionDef]:
    defs: list[FunctionDef] = []
    for _, caps in run_query(
        RUST_LANGUAGE,
        "(function_item name: (identifier) @name body: (block) @body) @funcdef",
        root,
    ):
        fn_node = caps["funcdef"][0]
        bare_name = node_text(caps["name"][0], source_bytes)
        class_name = _enclosing_impl_name(fn_node, source_bytes)
        qualified_name = f"{class_name}.{bare_name}" if class_name else bare_name
        start_line, end_line = line_range(fn_node)
        defs.append(
            FunctionDef(
                qualified_name=qualified_name, bare_name=bare_name, file=rel_path,
                start_line=start_line, end_line=end_line, body_node=caps["body"][0], class_name=class_name,
            )
        )
    return defs


def _enclosing_impl_name(fn_node: Node, source_bytes: bytes) -> str | None:
    """A function_item's attributes/doc-comments are siblings, not children
    (unlike C#'s method_declaration, whose attribute_list IS a child) --
    but the impl-block containment shape (function_item -> declaration_list
    -> impl_item) is otherwise identical to csharp_patterns.py's
    _enclosing_class_name. `impl_item`'s `type` field is the impl target
    even for a trait impl (`impl ServerHandler for Server`) -- verified
    against a real parse, it's "Server", not "ServerHandler".
    """
    container = fn_node.parent
    if container is None or container.type != "declaration_list" or container.parent is None:
        return None
    impl_node = container.parent
    if impl_node.type != "impl_item":
        return None
    type_node = impl_node.child_by_field_name("type")
    return node_text(type_node, source_bytes) if type_node is not None else None


def extract_imports(root: Node, source_bytes: bytes) -> ImportIndex:
    result: ImportIndex = {}
    for _, caps in run_query(RUST_LANGUAGE, "(use_declaration) @imp", root):
        imp_node = caps["imp"][0]
        target = next(
            (c for c in imp_node.children if c.type in ("scoped_identifier", "scoped_use_list", "identifier")),
            None,
        )
        if target is not None:
            _collect_use_bindings(target, source_bytes, result)
    return result


def _last_path_segment(path_node: Node, source_bytes: bytes) -> str:
    if path_node.type == "scoped_identifier":
        name_node = path_node.child_by_field_name("name")
        if name_node is not None:
            return node_text(name_node, source_bytes)
    return node_text(path_node, source_bytes)


def _collect_use_bindings(node: Node, source_bytes: bytes, result: dict[str, ImportedName]) -> None:
    """Best-effort `use` binding -> (module, original_name), same non-goal
    of full symbol resolution as import_index.py's docstring -- `module` is
    the path segment immediately before the imported name(s), a heuristic
    stand-in for "which file this probably came from" (Rust's convention of
    `mod foo;` pairing with `foo.rs`/`foo/mod.rs`), not real crate/module
    resolution. Glob imports (`use foo::*;`) bind no specific name and are
    skipped -- there's nothing to add to a per-alias map.
    """
    if node.type == "scoped_identifier":
        name_node = node.child_by_field_name("name")
        path_node = node.child_by_field_name("path")
        if name_node is None:
            return
        alias = node_text(name_node, source_bytes)
        module = _last_path_segment(path_node, source_bytes) if path_node is not None else ""
        result[alias] = ImportedName(module=module, original_name=alias)
    elif node.type == "identifier":
        alias = node_text(node, source_bytes)
        result[alias] = ImportedName(module=alias, original_name=alias)
    elif node.type == "scoped_use_list":
        path_node = node.child_by_field_name("path")
        list_node = node.child_by_field_name("list")
        if list_node is None:
            return
        module = _last_path_segment(path_node, source_bytes) if path_node is not None else ""
        for item in list_node.named_children:
            if item.type == "identifier":
                alias = node_text(item, source_bytes)
                result[alias] = ImportedName(module=module, original_name=alias)
            elif item.type == "use_as_clause":
                orig_node = item.child_by_field_name("path")
                alias_node = item.child_by_field_name("alias")
                if orig_node is None or alias_node is None:
                    continue
                result[node_text(alias_node, source_bytes)] = ImportedName(
                    module=module, original_name=node_text(orig_node, source_bytes)
                )
            elif item.type == "scoped_identifier":
                # nested grouped path, e.g. `use foo::{bar::Baz, qux::Quux};`.
                _collect_use_bindings(item, source_bytes, result)


def extract_calls(body_node: Node, source_bytes: bytes) -> list[CallSite]:
    call_sites: list[CallSite] = []
    for _, caps in run_query(RUST_LANGUAGE, "(call_expression function: (_) @fn) @call", body_node):
        fn_node = caps["fn"][0]
        call_node = caps["call"][0]
        raw_text = node_text(call_node, source_bytes)

        if fn_node.type == "identifier":
            call_sites.append(CallSite(callee_name=node_text(fn_node, source_bytes), receiver=None, raw_text=raw_text))
        elif fn_node.type == "field_expression":
            value_node = fn_node.child_by_field_name("value")
            field_node = fn_node.child_by_field_name("field")
            if field_node is None:
                continue
            receiver = node_text(value_node, source_bytes) if value_node is not None else None
            call_sites.append(
                CallSite(callee_name=node_text(field_node, source_bytes), receiver=receiver, raw_text=raw_text)
            )
        elif fn_node.type == "scoped_identifier":
            # `Type::method(...)` / `module::func(...)` -- associated-function
            # or module-qualified call. `Self::method(...)` is normalized to
            # receiver "self" so it rides resolve_call()'s existing
            # same-class tier (call_graph_builder.py) instead of needing a
            # Rust-specific tier in the shared resolver.
            path_node = fn_node.child_by_field_name("path")
            name_node = fn_node.child_by_field_name("name")
            if name_node is None:
                continue
            receiver = node_text(path_node, source_bytes) if path_node is not None else None
            if receiver == "Self":
                receiver = "self"
            call_sites.append(
                CallSite(callee_name=node_text(name_node, source_bytes), receiver=receiver, raw_text=raw_text)
            )
    return call_sites


def _preceding_attributes_and_docs(fn_node: Node) -> tuple[list[Node], list[Node]]:
    """Rust attributes and `///` doc comments are siblings of the function
    they annotate, not children of it (unlike C#'s attribute_list) -- walk
    backward through the contiguous run of attribute_item/line_comment
    siblings immediately above `fn_node`, in source order."""
    attributes: list[Node] = []
    doc_comments: list[Node] = []
    sib = fn_node.prev_sibling
    while sib is not None and sib.type in ("attribute_item", "line_comment"):
        if sib.type == "attribute_item":
            attributes.append(sib)
        else:
            doc_comment = next((c for c in sib.children if c.type == "doc_comment"), None)
            if doc_comment is not None:
                doc_comments.append(doc_comment)
        sib = sib.prev_sibling
    attributes.reverse()
    doc_comments.reverse()
    return attributes, doc_comments


def _attribute_name_and_args(attribute_item: Node, source_bytes: bytes) -> tuple[str | None, Node | None]:
    attribute = next((c for c in attribute_item.children if c.type == "attribute"), None)
    if attribute is None or attribute.child_count == 0:
        return None, None
    name_node = attribute.children[0]
    if name_node.type != "identifier":
        return None, None
    token_tree = next((c for c in attribute.children if c.type == "token_tree"), None)
    return node_text(name_node, source_bytes), token_tree


def _string_arg(token_tree: Node, source_bytes: bytes, key: str) -> tuple[str | None, bool]:
    """Reads a `key = value` pair out of a macro attribute's raw token
    tree -- tree-sitter-rust doesn't parse `#[tool(name = "x", ...)]`'s
    parenthesized args into structured key/value nodes (it's an opaque
    token_tree, since macro attribute syntax is macro-defined, not part of
    the language grammar), but `=`/`,`/parens are unnamed tokens, so
    `named_children` already gives a flat [identifier, value, identifier,
    value, ...] sequence -- confirmed against a real multi-key parse
    (name/title/description/annotations(...)/meta = f()), including that a
    nested group like `annotations(...)` or a call like `f()` doesn't
    desync the scan, since we only look at the *immediate* next named
    sibling of a matching key, not a globally-paired index.
    """
    named = token_tree.named_children
    for i, child in enumerate(named):
        if child.type == "identifier" and node_text(child, source_bytes) == key:
            if i + 1 >= len(named):
                return None, True
            value_node = named[i + 1]
            literal = _rust_string_value(value_node, source_bytes)
            if literal is not None:
                return literal, True
            return node_text(value_node, source_bytes), False
    return None, True


def _join_doc_comments(doc_comments: list[Node], source_bytes: bytes) -> str | None:
    """Mirrors the `rmcp` macro's own doc-comment-as-description fallback
    (rmcp-macros/src/common.rs's `extract_doc_line`, verified against the
    official SDK's source): each `///` line is stripped and joined with
    "\\n", blank lines contribute nothing.
    """
    lines = [node_text(c, source_bytes).strip() for c in doc_comments]
    lines = [line for line in lines if line]
    return "\n".join(lines) if lines else None


def detect_rust_tools(root: Node, source_bytes: bytes, rel_path: str) -> list[ToolRecord]:
    """Detects the official `rmcp` crate's `#[tool(...)]`/bare `#[tool]`
    attribute-macro pattern on methods inside a `#[tool_router]`-marked impl
    block. Verified against real parses across 10+ independent real
    MCP-server repos depending on `rmcp` (chroma-core/chroma,
    gitbutlerapp/gitbutler, apache/iggy, rerun-io/rerun, stakpak/agent,
    0xPlaygrounds/rig and others) plus the SDK's own macro source
    (modelcontextprotocol/rust-sdk) -- both the `name`/`description`
    attribute-arg form and the doc-comment-as-description fallback for a
    bare `#[tool]`/`#[tool(name = "...")]` are the SDK's real, documented
    behavior (rmcp-macros/src/common.rs's `extract_doc_line`), not a guess.
    """
    tools: list[ToolRecord] = []
    for _, caps in run_query(
        RUST_LANGUAGE,
        "(function_item name: (identifier) @name body: (block) @body) @funcdef",
        root,
    ):
        fn_node = caps["funcdef"][0]
        attributes, doc_comments = _preceding_attributes_and_docs(fn_node)

        has_tool = False
        tool_args: Node | None = None
        for attr_item in attributes:
            attr_name, attr_args = _attribute_name_and_args(attr_item, source_bytes)
            if attr_name == "tool":
                has_tool = True
                tool_args = attr_args
                break
        if not has_tool:
            continue

        name_override, description, description_is_literal = None, None, True
        if tool_args is not None:
            name_override, _ = _string_arg(tool_args, source_bytes, "name")
            description, description_is_literal = _string_arg(tool_args, source_bytes, "description")
        if description is None:
            description = _join_doc_comments(doc_comments, source_bytes)
            description_is_literal = True

        bare_name = node_text(caps["name"][0], source_bytes)
        class_name = _enclosing_impl_name(fn_node, source_bytes)
        qualified_name = f"{class_name}.{bare_name}" if class_name else bare_name
        start_line, end_line = line_range(fn_node)

        tools.append(
            ToolRecord(
                name=name_override or bare_name,
                description=description or "",
                description_is_literal=description_is_literal,
                sdk_pattern="rust.rmcp_tool_attribute",
                source_location=SourceLocation(file=rel_path, start_line=start_line, end_line=end_line),
                qualified_name=qualified_name,
            )
        )
    return tools
