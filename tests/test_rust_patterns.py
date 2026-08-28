from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.rust_patterns import (
    detect_rust_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Rust")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_tool_with_explicit_name_and_description():
    root, source_bytes = parse(
        """
#[tool_router]
impl Server {
    #[tool(name = "get-weather", description = "Get the weather of a city.")]
    pub async fn get_weather(&self, city: String) -> String {
        city
    }
}
"""
    )
    tools = detect_rust_tools(root, source_bytes, "server.rs")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "get-weather"
    assert tool.description == "Get the weather of a city."
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "rust.rmcp_tool_attribute"
    assert tool.qualified_name == "Server.get_weather"


def test_bare_tool_attribute_falls_back_to_function_name_and_doc_comment():
    root, source_bytes = parse(
        """
#[tool_router]
impl Server {
    /// Get the weather of a city.
    #[tool]
    pub async fn get_weather(&self, city: String) -> String {
        city
    }
}
"""
    )
    tools = detect_rust_tools(root, source_bytes, "server.rs")
    assert len(tools) == 1
    assert tools[0].name == "get_weather"
    assert tools[0].description == "Get the weather of a city."
    assert tools[0].description_is_literal is True


def test_multiline_doc_comment_is_joined_with_newlines():
    root, source_bytes = parse(
        """
#[tool_router]
impl Server {
    /// Get the weather of a city.
    /// Returns a plain string.
    #[tool]
    pub async fn get_weather(&self) -> String {
        String::new()
    }
}
"""
    )
    tools = detect_rust_tools(root, source_bytes, "server.rs")
    assert tools[0].description == "Get the weather of a city.\nReturns a plain string."


def test_tool_without_description_or_doc_comment_yields_empty_string():
    root, source_bytes = parse(
        """
#[tool_router]
impl Server {
    #[tool]
    pub async fn empty_param(&self) {}
}
"""
    )
    tools = detect_rust_tools(root, source_bytes, "server.rs")
    assert len(tools) == 1
    assert tools[0].description == ""


def test_annotations_and_meta_args_dont_break_name_description_extraction():
    root, source_bytes = parse(
        """
#[tool_router]
impl Server {
    #[tool(
        name = "gitbutler_workspace",
        title = "View GitButler workspace",
        description = "Returns the workspace.",
        annotations(
            read_only_hint = true,
            destructive_hint = false
        ),
        meta = workspace_tool_meta()
    )]
    async fn gitbutler_workspace(&self) {}
}
"""
    )
    tools = detect_rust_tools(root, source_bytes, "server.rs")
    assert len(tools) == 1
    assert tools[0].name == "gitbutler_workspace"
    assert tools[0].description == "Returns the workspace."


def test_functions_without_tool_attribute_are_ignored():
    root, source_bytes = parse(
        """
#[tool_router]
impl Server {
    fn helper(&self) -> String {
        String::new()
    }
}
"""
    )
    tools = detect_rust_tools(root, source_bytes, "server.rs")
    assert tools == []


def test_extract_definitions_distinguishes_impls_and_free_functions():
    root, source_bytes = parse(
        """
impl Server {
    pub async fn get_weather(&self, city: String) -> String { fetch(city) }
}
impl ServerHandler for Server {
    fn handshake(&self) {}
}
fn fetch(city: String) -> String { city }
"""
    )
    defs = extract_definitions(root, source_bytes, "server.rs")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {"Server.get_weather", "Server.handshake", "fetch"}


def test_extract_imports_handles_single_grouped_and_aliased_forms():
    root, source_bytes = parse(
        """
use crate::tools::weather::get_forecast;
use super::helpers::{format_output, parse_input as parse_it};
use crate::utils::*;
"""
    )
    imports = extract_imports(root, source_bytes)
    assert imports["get_forecast"].module == "weather"
    assert imports["format_output"].module == "helpers"
    assert imports["parse_it"].module == "helpers"
    assert imports["parse_it"].original_name == "parse_input"
    assert "utils" not in imports  # glob import binds no specific name


def test_extract_calls_distinguishes_bare_self_and_scoped_receivers():
    root, source_bytes = parse(
        """
impl Server {
    pub async fn get_weather(&self, city: String) -> String {
        helper_call(city.clone());
        self.other_method();
        Self::new();
        String::from(city)
    }
}
"""
    )
    defs = extract_definitions(root, source_bytes, "server.rs")
    body = next(d for d in defs if d.bare_name == "get_weather").body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["helper_call"].receiver is None
    assert by_name["other_method"].receiver == "self"
    assert by_name["new"].receiver == "self"  # `Self::` normalized to same-class tier
    assert by_name["from"].receiver == "String"
