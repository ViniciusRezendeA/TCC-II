from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Checkpoint:
    """Small JSON-file-backed progress store, shared by Step 1 (per sub-query
    search cursors) and Step 2 (per-repo completion status).

    Writes are atomic (write to a temp file, then rename) so a crash mid-write
    never corrupts previously saved progress.

    _save() merges with whatever is currently on disk before writing, instead of blindly
    overwriting from this instance's own in-memory `_data` -- confirmed live to matter: Etapa 3
    runs one Checkpoint instance per OS process (run_sequential_step3.py for Gemini,
    run_step3.py for every other judge), and both point at the SAME state/step3_progress.json
    (checkpoint_key already namespaces by judge_id, so one shared file across judges is
    intentional). Running two such processes at once (e.g. Gemini + DeepSeek in parallel) used
    to be silently destructive: each set() wrote out THIS process's full _data snapshot (loaded
    once at __init__), so whichever process saved last erased every entry the other had added
    to the file since its own load -- caught live via state/step3_progress.json holding only
    9,720 deepseek-flash keys while data/evaluations/deepseek-flash.jsonl already had 12,902
    unique (tool_uid, scenario) pairs recorded, and ~2,500 of those pairs billed twice because
    the vanished checkpoint entry made should_skip() treat already-paid-for work as pending
    again on the next restart. Re-reading the file on every save closes that window down to the
    ordinary last-write-wins-per-key race (two processes touching the exact same key at the
    exact same instant), which is rare and, unlike the whole-file case, loses at most that one
    key -- not everything the other process had done since it started.
    """

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

    def all(self) -> dict[str, Any]:
        return dict(self._data)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = self._load()
        merged.update(self._data)
        self._data = merged
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
