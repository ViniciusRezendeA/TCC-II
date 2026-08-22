from __future__ import annotations

from pathlib import Path

from mcp_pipeline.clone.clone_manager import slug_for_name_with_owner
from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.extraction.models import CallGraphNode, ToolRecord
from mcp_pipeline.schema.export_for_evaluation import export_tool_for_evaluation

# Measured on the real Etapa 2 pilot (747 tools): median rendered SOURCE_CODE ~10.2k chars
# (~2.5k tokens), p90 ~47k chars (~11.8k tokens), max ~345k chars (~86k tokens). The cap
# bounds that tail; it's a starting value, not an empirically validated limit.
MAX_SOURCE_CODE_CHARS = 32_000
TRUNCATION_NOTICE = "\n\n// [truncated -- call graph exceeds evaluation size cap]"


def repo_src_root_for(name_with_owner: str, repos_root: Path | None = None) -> Path:
    repos_root = repos_root if repos_root is not None else (DATA_DIR / "repos")
    return repos_root / slug_for_name_with_owner(name_with_owner) / "src"


def build_payload(
    tool: ToolRecord,
    call_graph: CallGraphNode,
    repo_src_root: Path,
    server_name: str,
    include_source: bool,
) -> dict:
    """Wraps export_tool_for_evaluation() (Etapa 2, frozen -- not modified here) with a size
    cap on SOURCE_CODE. Etapa 2's own export function stays untouched because it's already
    validated against real data and mirrors Hasan et al.'s payload shape exactly; the cap is
    an Etapa-3-specific concern (bounding judge cost/attention), so it lives here instead.
    """
    payload = export_tool_for_evaluation(tool, call_graph, repo_src_root, server_name, include_source)
    source_code = payload.get("SOURCE_CODE")
    if source_code and len(source_code) > MAX_SOURCE_CODE_CHARS:
        payload["SOURCE_CODE"] = source_code[:MAX_SOURCE_CODE_CHARS] + TRUNCATION_NOTICE
    return payload
