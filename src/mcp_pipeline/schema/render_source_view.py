from __future__ import annotations

from pathlib import Path

from mcp_pipeline.extraction.models import CallGraphNode, SourceLocation


def render_source_view(call_graph: CallGraphNode, repo_src_root: Path) -> str:
    """Flattens the structured call graph tree into a single annotated
    string — the same spirit as Hasan et al.'s `SOURCE_CODE` field (see the
    plan's "Achados sobre o formato de dados de Hasan et al."), but built
    from our richer 3-level tree instead of their ~1-level regex expansion.
    Each snippet is read fresh from the cloned repo on disk via
    `source_location` — this view is derived, not the source of truth
    (`dataset.jsonl`'s structured `call_graph` is).
    """
    chunks: list[str] = []
    _collect_chunks(call_graph, repo_src_root, chunks)
    return "\n\n".join(chunks)


def _collect_chunks(node: CallGraphNode, repo_src_root: Path, chunks: list[str]) -> None:
    if node.resolved and node.source_location is not None:
        snippet = _read_snippet(repo_src_root, node.source_location)
        label = f"// Fonte (nível {node.level}): {node.source_location.file}:{node.source_location.start_line}-{node.source_location.end_line}"
        if node.raw_call_text:
            label = f"// Chamada: {node.raw_call_text}\n{label}"
        chunks.append(f"{label}\n{snippet}")
    elif node.external:
        chunks.append(f"// Chamada externa/não resolvida (nível {node.level}): {node.raw_call_text}")

    for child in node.calls:
        _collect_chunks(child, repo_src_root, chunks)


def _read_snippet(repo_src_root: Path, loc: SourceLocation) -> str:
    lines = (repo_src_root / loc.file).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[loc.start_line - 1 : loc.end_line])
