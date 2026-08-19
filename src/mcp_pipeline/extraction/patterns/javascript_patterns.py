from __future__ import annotations

import functools

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns import ecmascript_common as _common
from mcp_pipeline.extraction.patterns.ecmascript_common import (
    synthetic_handler_name,  # noqa: F401 (re-exported)
)

JS_LANGUAGE = spec_for("JavaScript").ts_language

detect_mcp_tools = functools.partial(_common.detect_mcp_tools, JS_LANGUAGE, "javascript")
extract_definitions = functools.partial(_common.extract_definitions, JS_LANGUAGE)
extract_imports = functools.partial(_common.extract_imports, JS_LANGUAGE)
extract_calls = functools.partial(_common.extract_calls, JS_LANGUAGE)
