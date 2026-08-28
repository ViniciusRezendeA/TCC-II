from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.ruby_patterns import (
    detect_ruby_tools,
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Ruby")


def parse(source: str):
    parser = Parser(SPEC.ts_language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    return tree.root_node, source_bytes


def test_detects_tool_with_explicit_name_and_description():
    root, source_bytes = parse(
        """
class GreetTool < FastMcp::Tool
  tool_name 'greet'
  description 'Greet someone by name'

  def call(args)
    args[:name]
  end
end
"""
    )
    tools = detect_ruby_tools(root, source_bytes, "greet_tool.rb")
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "greet"
    assert tool.description == "Greet someone by name"
    assert tool.description_is_literal is True
    assert tool.sdk_pattern == "ruby.fastmcp_tool_class"
    assert tool.qualified_name == "GreetTool.call"


def test_defaults_name_to_class_name_when_tool_name_not_called():
    root, source_bytes = parse(
        """
class GreetTool < FastMcp::Tool
  description 'Greet someone'

  def call(args)
    args[:name]
  end
end
"""
    )
    tools = detect_ruby_tools(root, source_bytes, "greet_tool.rb")
    assert tools[0].name == "GreetTool"


def test_indirect_application_tool_superclass_is_detected():
    root, source_bytes = parse(
        """
class SampleTool < ApplicationTool
  description 'Greet a user'

  def call(id:)
    id
  end
end
"""
    )
    tools = detect_ruby_tools(root, source_bytes, "sample_tool.rb")
    assert len(tools) == 1
    assert tools[0].qualified_name == "SampleTool.call"


def test_class_without_call_method_is_ignored():
    root, source_bytes = parse(
        """
class AbstractTool < FastMcp::Tool
  description 'Base tool, no call override'
end
"""
    )
    tools = detect_ruby_tools(root, source_bytes, "abstract_tool.rb")
    assert tools == []


def test_class_not_inheriting_a_tool_base_is_ignored():
    root, source_bytes = parse(
        """
class GreetService < ApplicationService
  def call(args)
    args[:name]
  end
end
"""
    )
    tools = detect_ruby_tools(root, source_bytes, "greet_service.rb")
    assert tools == []


def test_interpolated_description_falls_back_to_raw_text():
    root, source_bytes = parse(
        """
class GreetTool < FastMcp::Tool
  description "Greet using #{PREFIX}"

  def call(args)
    args[:name]
  end
end
"""
    )
    tools = detect_ruby_tools(root, source_bytes, "greet_tool.rb")
    assert tools[0].description_is_literal is False
    assert "PREFIX" in tools[0].description


def test_extract_definitions_distinguishes_classes_and_modules():
    root, source_bytes = parse(
        """
class GreetTool < FastMcp::Tool
  def call(args)
    fetch(args)
  end

  def fetch(args)
    args
  end
end

module CacheUtils
  def self.get(key)
    key
  end
end
"""
    )
    defs = extract_definitions(root, source_bytes, "greet_tool.rb")
    qualified_names = {d.qualified_name for d in defs}
    assert qualified_names == {"GreetTool.call", "GreetTool.fetch", "CacheUtils.get"}


def test_extract_imports_returns_empty_mapping():
    root, source_bytes = parse(
        """
require 'json'
require_relative './helpers'
"""
    )
    assert extract_imports(root, source_bytes) == {}


def test_extract_calls_distinguishes_bare_self_and_constant_receivers():
    root, source_bytes = parse(
        """
class GreetTool < FastMcp::Tool
  def call(args)
    fetch(args)
    self.other_method
    Cache.get(args)
  end
end
"""
    )
    defs = extract_definitions(root, source_bytes, "greet_tool.rb")
    body = next(d for d in defs if d.bare_name == "call").body_node
    calls = extract_calls(body, source_bytes)
    by_name = {c.callee_name: c for c in calls}
    assert by_name["fetch"].receiver is None
    assert by_name["other_method"].receiver == "self"
    assert by_name["get"].receiver == "Cache"
