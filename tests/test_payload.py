from __future__ import annotations

from pathlib import Path

from mcp_pipeline.evaluation.payload import (
    MAX_SOURCE_CODE_CHARS,
    build_payload,
    repo_src_root_for,
)
from mcp_pipeline.extraction.models import CallGraphNode, SourceLocation, ToolRecord


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _sample_tool_and_graph(end_line: int = 2):
    tool = ToolRecord(
        name="get_weather",
        description="Fetch the weather",
        description_is_literal=True,
        sdk_pattern="python.fastmcp_decorator",
        source_location=SourceLocation(file="server.py", start_line=1, end_line=end_line),
        qualified_name="get_weather",
    )
    graph = CallGraphNode(
        level=1,
        resolved=True,
        external=False,
        ambiguous=False,
        qualified_name="get_weather",
        source_location=SourceLocation(file="server.py", start_line=1, end_line=end_line),
    )
    return tool, graph


def test_repo_src_root_for_matches_clone_manager_slug_convention(tmp_path):
    root = repo_src_root_for("acme/weather-mcp", repos_root=tmp_path)

    assert root == tmp_path / "acme__weather-mcp" / "src"


def test_build_payload_omits_source_code_for_description_only_scenario(tmp_path):
    repo = tmp_path / "src"
    _write(repo / "server.py", "def get_weather(city):\n    return city\n")
    tool, graph = _sample_tool_and_graph()

    payload = build_payload(tool, graph, repo, "acme/weather-mcp", include_source=False)

    assert "SOURCE_CODE" not in payload


def test_build_payload_keeps_small_source_code_untouched(tmp_path):
    repo = tmp_path / "src"
    _write(repo / "server.py", "def get_weather(city):\n    return city\n")
    tool, graph = _sample_tool_and_graph()

    payload = build_payload(tool, graph, repo, "acme/weather-mcp", include_source=True)

    assert "def get_weather(city):" in payload["SOURCE_CODE"]
    assert "truncated" not in payload["SOURCE_CODE"]


def test_build_payload_truncates_oversized_source_code(tmp_path):
    repo = tmp_path / "src"
    n_lines = 20_000
    body = "\n".join(f"    line_{i} = {i}" for i in range(n_lines))
    _write(repo / "server.py", f"def get_weather(city):\n{body}\n    return city\n")
    tool, graph = _sample_tool_and_graph(end_line=n_lines + 2)

    payload = build_payload(tool, graph, repo, "acme/weather-mcp", include_source=True)

    assert len(payload["SOURCE_CODE"]) > MAX_SOURCE_CODE_CHARS  # confirms the fixture is actually oversized
    truncation_notice = "// [truncated -- call graph exceeds evaluation size cap]"
    assert payload["SOURCE_CODE"].endswith(truncation_notice)
    assert len(payload["SOURCE_CODE"]) == MAX_SOURCE_CODE_CHARS + len("\n\n" + truncation_notice)
