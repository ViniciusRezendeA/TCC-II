from __future__ import annotations

import json

from mcp_pipeline.evaluation.prompts import (
    RUBRIC_COMPONENTS,
    RUBRIC_SYSTEM_PROMPT,
    build_user_message,
)


def test_exactly_six_components_defined():
    assert len(RUBRIC_COMPONENTS) == 6
    assert {key for key, _label, _definition in RUBRIC_COMPONENTS} == {
        "purpose",
        "guidelines",
        "limitations",
        "parameter_explanation",
        "length_completeness",
        "examples",
    }


def test_all_six_components_present_in_system_prompt():
    for key, label, _definition in RUBRIC_COMPONENTS:
        assert key in RUBRIC_SYSTEM_PROMPT
        assert label in RUBRIC_SYSTEM_PROMPT


def test_system_prompt_documents_source_code_scenario_handling():
    assert "SOURCE_CODE" in RUBRIC_SYSTEM_PROMPT


def test_build_user_message_is_valid_json_matching_payload():
    payload = {"name": "get_weather", "server_name": "acme/weather-mcp", "description": "Fetch weather."}

    message = build_user_message(payload)

    assert json.loads(message) == payload
