from __future__ import annotations

"""Repeatable version of the manual SDK syntax check done during the design
of this pipeline (2026-08-17), which found that the Python, TypeScript and
Java official MCP SDKs were mid-rewrite at that time (two syntax generations
coexisting). Re-run this before a full Etapa 1/2 build to catch further
drift — see plan risk #1.

Shallow-clones the four official SDK repos into a throwaway temp directory
and grep-checks for the anchor strings each language's tool-detection
patterns rely on (see src/mcp_pipeline/extraction/patterns/*.py once those
exist). Does not modify anything in this project; safe to run any time.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

SDKS = {
    "python": "https://github.com/modelcontextprotocol/python-sdk.git",
    "typescript": "https://github.com/modelcontextprotocol/typescript-sdk.git",
    "java": "https://github.com/modelcontextprotocol/java-sdk.git",
    "csharp": "https://github.com/modelcontextprotocol/csharp-sdk.git",
}

# (language, description, anchor substring) — presence is checked with `grep -r`
# against the freshly cloned SDK source. This is a coarse smoke check, not a
# guarantee the detection patterns in extraction/patterns/ still match byte
# for byte — treat a missing anchor as "go re-read that SDK's source before
# trusting the existing tree-sitter patterns".
ANCHORS = [
    ("python", "v1 FastMCP decorator", "mcp.server.fastmcp"),
    ("python", "v1 low-level Server.list_tools", "list_tools"),
    ("python", "v2 MCPServer constructor kwargs", "on_list_tools"),
    ("typescript", "v1 @modelcontextprotocol/sdk package", "@modelcontextprotocol/sdk"),
    ("typescript", "v1/v2 registerTool", "registerTool"),
    ("typescript", "v2 split server package", "@modelcontextprotocol/server"),
    ("java", "official builder Tool.builder", "Tool.builder"),
    ("java", "SyncToolSpecification", "SyncToolSpecification"),
    ("csharp", "McpServerToolType attribute", "McpServerToolType"),
    ("csharp", "McpServerTool attribute", "McpServerTool"),
]


def clone_shallow(url: str, dest: Path) -> bool:
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        print(f"  FALHA ao clonar {url}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def check_anchor(repo_dir: Path, anchor: str) -> bool:
    result = subprocess.run(
        ["grep", "-r", "-l", "-F", anchor, str(repo_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mcp_sdk_verify_") as tmp:
        tmp_path = Path(tmp)
        repo_dirs: dict[str, Path] = {}

        for lang, url in SDKS.items():
            dest = tmp_path / lang
            print(f"Clonando SDK {lang}...")
            if clone_shallow(url, dest):
                repo_dirs[lang] = dest

        print("\n=== Verificação de âncoras de sintaxe ===")
        any_missing = False
        for lang, description, anchor in ANCHORS:
            if lang not in repo_dirs:
                print(f"[PULADO] {lang}: {description} (clone falhou)")
                continue
            found = check_anchor(repo_dirs[lang], anchor)
            status = "OK" if found else "AUSENTE"
            if not found:
                any_missing = True
            print(f"[{status}] {lang}: {description} (\"{anchor}\")")

        if any_missing:
            print(
                "\nAlgumas âncoras não foram encontradas — revisar o SDK "
                "correspondente manualmente antes de confiar nos padrões de "
                "extração existentes (podem ter mudado de nome/local)."
            )
            sys.exit(1)
        print("\nTodas as âncoras conhecidas ainda estão presentes.")


if __name__ == "__main__":
    main()
