from __future__ import annotations

import json

# Bumping this invalidates every checkpoint key that embeds it (see run_step3.py's
# checkpoint_key()) whenever the rubric text or the RubricScores schema changes materially
# -- old results stay in the output JSONL tagged with their own prompt_version instead of
# being silently skipped or overwritten.
# v2: Changed SOURCE_CODE handling from "may use" (enrich) to "MUST validate" (congruence check)
#     Added explicit penalty guidance for omissions/contradictions found in code
# v3: v2's "don't enrich" instruction only covered limitations/parameter_explanation (strong)
#     and guidelines/examples (soft); purpose and length_completeness had NO SOURCE_CODE
#     guidance at all. Live pilot data (gemini-3.5-flash-lite, N=388) showed with_source
#     scoring higher than description_only on ALL 6 components -- including
#     length_completeness, which grades the description's own prose length and cannot
#     legitimately change with SOURCE_CODE, since "description" is byte-identical between
#     scenarios (see export_for_evaluation.py). That's a context-halo/leniency effect, not
#     genuine congruence checking. v3 replaces the per-component carve-outs with one
#     explicit rule covering all 6: SOURCE_CODE may only ever lower a score (a found
#     contradiction/omission), never raise one -- and requires the reasoning to name the
#     specific issue whenever a score is lowered, so a spot-check can audit compliance.
PROMPT_VERSION = "v3"

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
# variants. Keeping one prompt text per judge maximizes cached-content reuse (Gemini)
# across both scenarios.
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
- If "SOURCE_CODE" is absent, judge the description strictly on its own terms. Do not penalize it for omitting information that only the source code would reveal.
- If "SOURCE_CODE" is present, first judge each component exactly as you would from the description alone. SOURCE_CODE's only function is to check the description for a CONTRADICTION or OMISSION relative to the real implementation -- it is never a source of extra credit.

  THE RULE, for all 6 components without exception: SOURCE_CODE may only LOWER a component's score below what the description text alone would earn. It must NEVER raise a score -- more context available to you, the judge, is not the same as a better description. If you find no contradiction or omission for a component, its score must equal what the description alone would have earned; do not nudge it up because the code happened to clarify or confirm things.

  Component-specific application:
  * "limitations" / "parameter_explanation" (primary intended use of SOURCE_CODE): check whether the code reveals constraints, failure modes, parameter semantics, or edge cases the description does NOT mention. If so, penalize accordingly.
  * "purpose": only lower the score if the code shows the tool does something materially different from what the description claims -- a real contradiction, not just "the code has more detail."
  * "guidelines" / "examples": the code revealing usage patterns or activation criteria the description never mentions is grounds to check for a contradiction (e.g. the description recommends a use the code doesn't support), never grounds to raise the score for detail the description itself lacks.
  * "length_completeness": judged purely on the description's own length and substantiveness. SOURCE_CODE is irrelevant to this component and must not change its score in either direction.

  A vague or terse description remains vague or terse even when the code clarifies or fully justifies the implementation.

For each of the 6 components, return a Likert score (1-5) and a brief (1-3 sentence) reasoning that cites specific evidence from the description (and from SOURCE_CODE, when present and relevant). When SOURCE_CODE is present and you score below what the description alone would earn, your reasoning MUST name the specific contradiction or omission that justifies it; otherwise state that SOURCE_CODE confirmed the description with no issue found."""


def build_user_message(payload: dict) -> str:
    """Mirrors Hasan et al.'s own construction of the LLM-judge input turn: the tool
    payload (from evaluation.payload.build_payload) serialized as indented JSON, nothing
    else added.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False)
