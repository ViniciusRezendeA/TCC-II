from __future__ import annotations

from pathlib import Path

from mcp_pipeline.extraction.models import CallGraphNode, ToolRecord
from mcp_pipeline.schema.render_source_view import render_source_view


def export_tool_for_evaluation(
    tool: ToolRecord, call_graph: CallGraphNode, repo_src_root: Path, server_name: str, include_source: bool
) -> dict:
    """Packages one tool into the exact JSON shape Hasan et al.'s LLM-jury
    prompt expects (`evaluate_tool_descriptions.py`, see the plan): `{name,
    server_name, description, SOURCE_CODE}`. `include_source=False` matches
    the TCC's cenário 1 (description only); `include_source=True` matches
    cenário 2 — the object is otherwise identical, so Etapa 3 can drive both
    scenarios from the same export function.
    """
    payload = {"name": tool.name, "server_name": server_name, "description": tool.description}
    if include_source:
        payload["SOURCE_CODE"] = render_source_view(call_graph, repo_src_root)
    return payload
