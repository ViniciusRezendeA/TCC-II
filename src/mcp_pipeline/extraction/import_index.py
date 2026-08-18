from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImportedName:
    module: str  # e.g. "cache_utils" or ".cache_utils" for a relative import
    original_name: str  # the name as defined in the source module


# Per-file map: local alias -> where it came from. A deliberate scoping
# heuristic (architecture principle in the plan: "não é resolução completa
# de símbolos") — used only to help call_graph_builder narrow down which
# file's definitions to search first, never as a guarantee of correctness.
ImportIndex = dict[str, ImportedName]
