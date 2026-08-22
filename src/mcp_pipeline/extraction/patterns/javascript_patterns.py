from __future__ import annotations

import functools

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns import ecmascript_common as _common
from mcp_pipeline.extraction.patterns.ecmascript_common import (  # noqa: F401 (re-exported)
    synthetic_handler_name,
    synthetic_list_tools_handler_name,
)

JS_LANGUAGE = spec_for("JavaScript").ts_language

detect_mcp_tools = functools.partial(_common.detect_mcp_tools, JS_LANGUAGE, "javascript")
detect_fastmcp_npm_addtool = functools.partial(_common.detect_fastmcp_npm_addtool, JS_LANGUAGE, "javascript")
detect_lowlevel_set_request_handler = functools.partial(
    _common.detect_lowlevel_set_request_handler, JS_LANGUAGE, "javascript"
)
extract_definitions = functools.partial(_common.extract_definitions, JS_LANGUAGE)
extract_imports = functools.partial(_common.extract_imports, JS_LANGUAGE)
extract_calls = functools.partial(_common.extract_calls, JS_LANGUAGE)
extract_values = functools.partial(_common.extract_values, JS_LANGUAGE)
