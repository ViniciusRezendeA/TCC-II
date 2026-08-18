from __future__ import annotations

from dataclasses import dataclass

import tree_sitter_c_sharp
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language


@dataclass(frozen=True)
class LanguageSpec:
    name: str  # matches RepoCandidate.primary_language values
    ts_language: Language
    extensions: tuple[str, ...]


# tree_sitter_typescript exposes both language_typescript() and language_tsx();
# MCP server code is backend code, so only the plain TS/JS grammars are needed —
# JSX/TSX support is deliberately out of scope.
LANGUAGES: dict[str, LanguageSpec] = {
    "Python": LanguageSpec("Python", Language(tree_sitter_python.language()), (".py",)),
    "JavaScript": LanguageSpec("JavaScript", Language(tree_sitter_javascript.language()), (".js", ".mjs", ".cjs")),
    "TypeScript": LanguageSpec(
        "TypeScript", Language(tree_sitter_typescript.language_typescript()), (".ts",)
    ),
    "Java": LanguageSpec("Java", Language(tree_sitter_java.language()), (".java",)),
    "C#": LanguageSpec("C#", Language(tree_sitter_c_sharp.language()), (".cs",)),
}


def spec_for(language_name: str) -> LanguageSpec:
    try:
        return LANGUAGES[language_name]
    except KeyError:
        raise ValueError(f"Linguagem não suportada pela Etapa 2: {language_name!r}") from None
