from __future__ import annotations

import json

# Bumping this invalidates every checkpoint key that embeds it (see run_step3.py's
# checkpoint_key()) whenever the rubric text or the RubricScores schema changes materially
# -- old results stay in the output JSONL tagged with their own prompt_version instead of
# being silently skipped or overwritten.
PROMPT_VERSION = "v1"

RUBRIC_COMPONENTS: list[tuple[str, str, str]] = [
    (
        "purpose",
        "Purpose",
        "How clearly and completely the description explains what the tool does.",
    ),
    (
        "guidelines",
        "Guidelines",
        (
            "Whether the description gives both decision-making activation criteria (when to "
            "use the tool) and operational instructions (how to use it)."
        ),
    ),
    (
        "limitations",
        "Limitations",
        (
            "Whether known constraints, caveats, or corner cases where the tool may fail are "
            "disclosed."
        ),
    ),
    (
        "parameter_explanation",
        "Parameter Explanation",
        (
            "Whether the roles of the tool's input parameters are explained beyond just their "
            "data types."
        ),
    ),
    (
        "length_completeness",
        "Length & Completeness",
        (
            "Whether the description reaches at least three to four sentences of substantive "
            "detail, rather than being a terse fragment."
        ),
    ),
    (
        "examples",
        "Examples",
        "Whether illustrative examples of correct and effective usage are given.",
    ),
]

LIKERT_SCALE = """5 = Ideal: fully satisfies this component, with no meaningful ambiguity.
4 = Minor ambiguity: mostly satisfies this component; small gaps a reader could resolve unaided.
3 = Minimum viable: meets a bare threshold for this component; noticeably incomplete but usable.
2 = Vague: this aspect is present but too thin or unclear to rely on.
1 = Missing: this aspect is absent from the description entirely."""


def _render_components() -> str:
    return "\n".join(
        f"{i}. {label} ({key}): {definition}"
        for i, (key, label, definition) in enumerate(RUBRIC_COMPONENTS, start=1)
    )


# Identical across both evaluation scenarios (with/without SOURCE_CODE) -- the scenario is
# driven entirely by whether the payload's SOURCE_CODE key is present, not by prompt
# variants. Keeping one prompt text per judge maximizes prompt-cache hit rate (Claude) /
# cached-content reuse (OpenAI, Gemini) across both scenarios.
RUBRIC_SYSTEM_PROMPT = f"""You are an expert evaluator of Model Context Protocol (MCP) tool descriptions, participating in a research study on tool-description quality.

You will be given a JSON object describing one MCP tool, with the following fields:
- "name": the tool's identifier.
- "server_name": the MCP server (repository) the tool belongs to.
- "description": the natural-language description exposed to an LLM agent deciding whether and how to call the tool. This is what you are evaluating.
- "SOURCE_CODE" (optional): source code context for the tool, including its own implementation and up to two levels of the functions it calls. Only present in some evaluations.

Score the "description" against each of the following 6 components, independently, on a 5-point Likert scale. A low score on one component must NOT depress your score on another -- evaluate each on its own terms.

{_render_components()}

Likert scale (apply uniformly to all 6 components):
{LIKERT_SCALE}

Handling "SOURCE_CODE":
- If "SOURCE_CODE" is present in the input, you may use it to inform "limitations" and "parameter_explanation" specifically: does the code reveal constraints, failure modes, or parameter semantics that the description omits?
- If "SOURCE_CODE" is absent, judge the description strictly on its own terms. Do not penalize it for omitting information that only the source code would reveal.

For each of the 6 components, return a Likert score (1-5) and a brief (1-3 sentence) reasoning that cites specific evidence from the description (and from SOURCE_CODE, when present and relevant)."""


def build_user_message(payload: dict) -> str:
    """Mirrors Hasan et al.'s own construction of the LLM-judge input turn: the tool
    payload (from evaluation.payload.build_payload) serialized as indented JSON, nothing
    else added.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False)
