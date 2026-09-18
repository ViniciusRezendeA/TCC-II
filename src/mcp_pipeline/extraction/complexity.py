from __future__ import annotations

from tree_sitter import Node

from mcp_pipeline.extraction.parser_utils import node_text

# McCabe (1976) classic cyclomatic complexity: CC = decision points + 1.
# Short-circuit boolean operators (&&, ||, and/or) are deliberately NOT
# counted as extra decision points -- that's the "extended"/radon-style
# variant, not classic McCabe (see the plan discussion this was decided
# against).
#
# Node type names below were verified per language by parsing representative
# snippets with the exact tree-sitter grammar versions pinned in
# pyproject.toml, not assumed from memory -- grammars disagree on naming
# (e.g. Python's ternary is "conditional_expression", C#'s is also
# "conditional_expression", but JS/TS/Java call the same construct
# "ternary_expression").
#
# A plain `else`/`default` branch adds no decision point of its own (it has
# no predicate -- it's the "no other condition matched" fallthrough), so it's
# excluded everywhere. For switch/match/when constructs this requires
# distinguishing "case" labels from "default" labels; some grammars give
# them distinct node types (JS/TS's switch_case vs switch_default, Go's
# expression_case vs default_case, Dart's switch_statement_case vs
# switch_statement_default) so a flat type-membership set is enough. Others
# don't (Java's switch_label and C#'s switch_section cover both, Kotlin's
# when_entry covers both, Rust's match_arm and Python's case_clause don't
# structurally distinguish a wildcard `_`/`case _` from a normal arm) --
# those are handled by `_is_extra_decision` below via a per-node check.
_SIMPLE_DECISION_TYPES: dict[str, frozenset[str]] = {
    "Python": frozenset({
        "if_statement", "elif_clause", "for_statement", "while_statement",
        "except_clause", "conditional_expression",
    }),
    "JavaScript": frozenset({
        "if_statement", "for_statement", "for_in_statement", "while_statement",
        "do_statement", "catch_clause", "switch_case", "ternary_expression",
    }),
    "TypeScript": frozenset({
        "if_statement", "for_statement", "for_in_statement", "while_statement",
        "do_statement", "catch_clause", "switch_case", "ternary_expression",
    }),
    "Java": frozenset({
        "if_statement", "for_statement", "enhanced_for_statement", "while_statement",
        "do_statement", "catch_clause", "ternary_expression",
    }),
    "C#": frozenset({
        "if_statement", "for_statement", "foreach_statement", "while_statement",
        "do_statement", "catch_clause", "conditional_expression",
    }),
    "Go": frozenset({
        # No ternary, no exceptions/catch in Go.
        "if_statement", "for_statement", "expression_case", "communication_case",
    }),
    "Rust": frozenset({
        # `loop { ... }` has no predicate of its own, but it's still an extra
        # back-edge in the control-flow graph (same reason do-while counts
        # elsewhere despite checking its condition at the end) -- +1 under
        # the graph-theoretic edges-minus-nodes-plus-2 reading of McCabe.
        "if_expression", "for_expression", "while_expression", "loop_expression",
    }),
    "Ruby": frozenset({
        "if", "elsif", "unless", "for", "while", "until", "when", "rescue", "conditional",
    }),
    "Dart": frozenset({
        "if_statement", "for_statement", "while_statement", "do_statement",
        "catch_clause", "switch_statement_case", "conditional_expression",
    }),
    "Kotlin": frozenset({
        "if_expression", "for_statement", "while_statement", "do_while_statement",
        "catch_block",
    }),
}

# Nested function/closure/lambda boundaries: the walk stops descending here
# so a tool's complexity isn't inflated by decision points that belong to an
# inner function it merely defines (never calls) -- mirrors how `loc` is
# scoped to one FunctionDef, not whatever happens to be lexically inside it.
# Ruby's "block" (do...end / {}) is deliberately NOT a boundary: an
# each-style block is inline control flow belonging to the enclosing method,
# consistent with how RuboCop's own CyclomaticComplexity cop treats blocks.
_FUNCTION_BOUNDARY_TYPES: dict[str, frozenset[str]] = {
    "Python": frozenset({"function_definition", "lambda"}),
    "JavaScript": frozenset({
        "function_declaration", "function_expression", "arrow_function",
        "generator_function", "generator_function_declaration", "method_definition",
    }),
    "TypeScript": frozenset({
        "function_declaration", "function_expression", "arrow_function",
        "generator_function", "generator_function_declaration", "method_definition",
    }),
    "Java": frozenset({"method_declaration", "lambda_expression"}),
    "C#": frozenset({
        "method_declaration", "lambda_expression", "anonymous_method_expression",
        "local_function_statement",
    }),
    "Go": frozenset({"func_literal", "function_declaration", "method_declaration"}),
    "Rust": frozenset({"closure_expression", "function_item"}),
    "Ruby": frozenset({"method", "singleton_method", "lambda"}),
    "Dart": frozenset({"function_expression", "function_signature"}),
    "Kotlin": frozenset({"function_declaration", "lambda_literal", "anonymous_function"}),
}


def _is_extra_decision(node: Node, language: str, source_bytes: bytes) -> bool:
    """Node types that need a per-node check instead of a flat type-membership
    test -- see the module docstring's explanation of why these five can't be
    handled by `_SIMPLE_DECISION_TYPES` alone."""
    if language == "Python" and node.type == "case_clause":
        pattern = next((c for c in node.children if c.type == "case_pattern"), None)
        return pattern is None or node_text(pattern, source_bytes).strip() != "_"
    if language == "Rust" and node.type == "match_arm":
        pattern = next((c for c in node.children if c.type == "match_pattern"), None)
        return pattern is None or node_text(pattern, source_bytes).strip() != "_"
    if language == "Kotlin" and node.type == "when_entry":
        return not any(c.type == "else" for c in node.children)
    if language == "Java" and node.type == "switch_label":
        return bool(node.children) and node.children[0].type == "case"
    if language == "C#" and node.type == "switch_section":
        return bool(node.children) and node.children[0].type == "case"
    return False


def cyclomatic_complexity(body_node: Node, language: str, source_bytes: bytes) -> int:
    """McCabe cyclomatic complexity of one function's body, scoped exactly
    like `loc` (see models.py's ToolRecord.loc): counts decision points
    inside `body_node` but not inside any nested function/lambda it defines.
    """
    simple_types = _SIMPLE_DECISION_TYPES.get(language, frozenset())
    boundary_types = _FUNCTION_BOUNDARY_TYPES.get(language, frozenset())
    decision_points = 0

    def walk(node: Node) -> None:
        nonlocal decision_points
        if node.type in simple_types or _is_extra_decision(node, language, source_bytes):
            decision_points += 1
        for child in node.children:
            if child.type in boundary_types:
                continue
            walk(child)

    walk(body_node)
    return decision_points + 1
