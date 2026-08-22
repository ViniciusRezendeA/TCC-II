from __future__ import annotations

import functools

from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns import ecmascript_common as _common
from mcp_pipeline.extraction.patterns.ecmascript_common import (  # noqa: F401 (re-exported)
    synthetic_handler_name,
    synthetic_list_tools_handler_name,
)

TS_LANGUAGE = spec_for("TypeScript").ts_language

detect_mcp_tools = functools.partial(_common.detect_mcp_tools, TS_LANGUAGE, "typescript")
detect_fastmcp_npm_addtool = functools.partial(_common.detect_fastmcp_npm_addtool, TS_LANGUAGE, "typescript")
detect_lowlevel_set_request_handler = functools.partial(
    _common.detect_lowlevel_set_request_handler, TS_LANGUAGE, "typescript"
)
extract_definitions = functools.partial(_common.extract_definitions, TS_LANGUAGE)
extract_imports = functools.partial(_common.extract_imports, TS_LANGUAGE)
extract_calls = functools.partial(_common.extract_calls, TS_LANGUAGE)
extract_values = functools.partial(_common.extract_values, TS_LANGUAGE)
