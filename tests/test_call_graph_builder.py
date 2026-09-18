from __future__ import annotations

from pathlib import Path

from mcp_pipeline.extraction.call_graph_builder import (
    CallSite,
    build_call_graph,
    call_graph_depth,
    collect_reachable_definitions,
    resolve_call,
)
from mcp_pipeline.extraction.definition_index import (
    DefinitionIndex,
    FunctionDef,
    build_definition_index,
)
from mcp_pipeline.extraction.import_index import ImportedName
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.models import CallGraphNode
from mcp_pipeline.extraction.parser_utils import (
    iter_source_files,
    parse_file,
    relative_path,
)
from mcp_pipeline.extraction.patterns.python_patterns import (
    extract_calls,
    extract_definitions,
    extract_imports,
)

SPEC = spec_for("Python")


def fn(qualified_name, bare_name, file, class_name=None):
    return FunctionDef(
        qualified_name=qualified_name, bare_name=bare_name, file=file,
        start_line=1, end_line=2, body_node=None, class_name=class_name,
    )


# --- resolve_call: each heuristic step in isolation ---------------------


def test_step1_self_receiver_resolves_to_same_class_method():
    defs = DefinitionIndex.build([fn("Weather.get", "get", "server.py", "Weather"), fn("Weather._fetch", "_fetch", "server.py", "Weather")])
    call = CallSite(callee_name="_fetch", receiver="self", raw_text="self._fetch(x)")

    resolved, ambiguous = resolve_call(call, current_file="server.py", current_class="Weather", definitions=defs, imports_by_file={})

    assert resolved.qualified_name == "Weather._fetch"
    assert ambiguous is False


def test_step2_same_file_bare_name_resolves_to_module_helper():
    defs = DefinitionIndex.build([fn("helper", "helper", "server.py"), fn("other.helper", "helper", "other.py", "other")])
    call = CallSite(callee_name="helper", receiver=None, raw_text="helper(x)")

    resolved, ambiguous = resolve_call(call, current_file="server.py", current_class=None, definitions=defs, imports_by_file={})

    assert resolved.file == "server.py"
    assert ambiguous is False


def test_step3_import_alias_resolves_to_imported_submodule_file():
    defs = DefinitionIndex.build(
        [fn("get_cached", "get_cached", "cache_utils.py"), fn("decoy.get_cached", "get_cached", "other/decoy.py", "decoy")]
    )
    imports_by_file = {"server.py": {"cache_utils": ImportedName(module=".", original_name="cache_utils")}}
    call = CallSite(callee_name="get_cached", receiver="cache_utils", raw_text="cache_utils.get_cached(x)")

    resolved, ambiguous = resolve_call(call, current_file="server.py", current_class=None, definitions=defs, imports_by_file=imports_by_file)

    assert resolved.file == "cache_utils.py"
    assert ambiguous is False


def test_step4_unique_repo_wide_name_resolves():
    defs = DefinitionIndex.build([fn("format_output", "format_output", "utils/format.py")])
    call = CallSite(callee_name="format_output", receiver=None, raw_text="format_output(x)")

    resolved, ambiguous = resolve_call(call, current_file="server.py", current_class=None, definitions=defs, imports_by_file={})

    assert resolved.file == "utils/format.py"
    assert ambiguous is False


def test_step4_ambiguous_name_is_flagged_but_still_resolves_a_guess():
    defs = DefinitionIndex.build(
        [fn("a.run", "run", "handlers/a.py", "a"), fn("b.run", "run", "handlers/b.py", "b")]
    )
    call = CallSite(callee_name="run", receiver=None, raw_text="run(x)")

    resolved, ambiguous = resolve_call(call, current_file="handlers/caller.py", current_class=None, definitions=defs, imports_by_file={})

    assert resolved is not None
    assert ambiguous is True


def test_step5_unresolvable_call_returns_none():
    defs = DefinitionIndex.build([fn("helper", "helper", "server.py")])
    call = CallSite(callee_name="requests_get", receiver="requests", raw_text="requests.get(x)")

    resolved, ambiguous = resolve_call(call, current_file="server.py", current_class=None, definitions=defs, imports_by_file={})

    assert resolved is None
    assert ambiguous is False


def test_self_receiver_without_matching_class_method_falls_through():
    """self.foo() where foo isn't defined on the class must not crash — it
    should fall through the remaining steps instead of stopping at step 1."""
    defs = DefinitionIndex.build([fn("helper_foo", "foo", "server.py")])
    call = CallSite(callee_name="foo", receiver="self", raw_text="self.foo(x)")

    resolved, ambiguous = resolve_call(call, current_file="server.py", current_class="Weather", definitions=defs, imports_by_file={})

    assert resolved.qualified_name == "helper_foo"
    assert ambiguous is False


# --- build_call_graph: full integration on real files --------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_build_call_graph_three_levels_with_external_leaf_and_dedup(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        '''
from . import cache_utils
import requests


class WeatherServer:
    def get_weather(self, city):
        a = self._fetch(city)
        b = self._fetch(city)  # duplicate call site, must dedupe to one edge
        return a

    def _fetch(self, city):
        requests.get(city)
        return cache_utils.get_cached(city)
''',
    )
    _write(
        repo / "cache_utils.py",
        """
def get_cached(key):
    return _lookup(key)


def _lookup(key):
    return None
""",
    )

    definitions = build_definition_index(repo, SPEC, extract_definitions)
    imports_by_file = {}
    source_bytes_by_file = {}
    for f in iter_source_files(repo, SPEC):
        tree, source_bytes = parse_file(f, SPEC)
        rel = relative_path(f, repo)
        imports_by_file[rel] = extract_imports(tree.root_node, source_bytes)
        source_bytes_by_file[rel] = source_bytes

    start = definitions.by_qualified_name["WeatherServer.get_weather"]
    graph = build_call_graph(start, definitions, imports_by_file, source_bytes_by_file, extract_calls)

    assert graph.level == 1
    assert graph.qualified_name == "WeatherServer.get_weather"
    assert len(graph.calls) == 1  # deduped from 2 identical call sites to 1 edge

    level2 = graph.calls[0]
    assert level2.qualified_name == "WeatherServer._fetch"
    assert level2.level == 2

    level3_by_name = {c.raw_call_text: c for c in level2.calls}
    assert level3_by_name["requests.get(city)"].external is True
    assert level3_by_name["requests.get(city)"].resolved is False
    assert level3_by_name["cache_utils.get_cached(city)"].resolved is True
    assert level3_by_name["cache_utils.get_cached(city)"].qualified_name == "get_cached"

    # BFS stops at level 3 by construction — get_cached's own call to
    # _lookup must never be scanned/appear as a level-4 node.
    resolved_level3 = level3_by_name["cache_utils.get_cached(city)"]
    assert resolved_level3.calls == []


def test_build_call_graph_handles_recursive_calls_without_infinite_loop(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        """
def get_weather(city):
    return get_weather(city)
""",
    )
    definitions = build_definition_index(repo, SPEC, extract_definitions)
    imports_by_file = {}
    source_bytes_by_file = {}
    for f in iter_source_files(repo, SPEC):
        tree, source_bytes = parse_file(f, SPEC)
        rel = relative_path(f, repo)
        imports_by_file[rel] = extract_imports(tree.root_node, source_bytes)
        source_bytes_by_file[rel] = source_bytes

    start = definitions.by_qualified_name["get_weather"]
    graph = build_call_graph(start, definitions, imports_by_file, source_bytes_by_file, extract_calls)

    assert graph.level == 1
    assert graph.calls[0].qualified_name == "get_weather"
    assert graph.calls[0].level == 2
    assert len(graph.calls[0].calls) == 1
    level3 = graph.calls[0].calls[0]
    assert level3.level == 3
    assert level3.qualified_name == "get_weather"
    assert level3.calls == []  # level-3 nodes are never call-site-scanned, so recursion can't go to level 4


def test_call_graph_depth_reaches_max_level_when_tree_is_full(tmp_path):
    repo = tmp_path / "src"
    _write(
        repo / "server.py",
        '''
from . import cache_utils
import requests


class WeatherServer:
    def get_weather(self, city):
        return self._fetch(city)

    def _fetch(self, city):
        requests.get(city)
        return cache_utils.get_cached(city)
''',
    )
    _write(repo / "cache_utils.py", "def get_cached(key):\n    return None\n")

    definitions = build_definition_index(repo, SPEC, extract_definitions)
    imports_by_file = {}
    source_bytes_by_file = {}
    for f in iter_source_files(repo, SPEC):
        tree, source_bytes = parse_file(f, SPEC)
        rel = relative_path(f, repo)
        imports_by_file[rel] = extract_imports(tree.root_node, source_bytes)
        source_bytes_by_file[rel] = source_bytes

    start = definitions.by_qualified_name["WeatherServer.get_weather"]
    graph = build_call_graph(start, definitions, imports_by_file, source_bytes_by_file, extract_calls)

    assert call_graph_depth(graph) == 3


def test_call_graph_depth_of_leaf_only_node_is_one():
    leaf = CallGraphNode(level=1, resolved=True, external=False, ambiguous=False)
    assert call_graph_depth(leaf) == 1


# --- collect_reachable_definitions: unbounded closure for loc/complexity ----


def _fn_with_marker(name: str) -> FunctionDef:
    # body_node doubles as a lookup key into the fake extract_calls below --
    # real callers pass a tree-sitter Node here, but nothing in
    # collect_reachable_definitions itself inspects body_node's type, only
    # extract_calls(body_node, source_bytes) does, so a plain string marker
    # is enough for a resolver-focused test like this one.
    return FunctionDef(qualified_name=name, bare_name=name, file="server.py", start_line=1, end_line=2, body_node=name)


def _fake_extract_calls(edges: dict[str, list[str]]):
    def extract_calls(body_node, source_bytes):
        return [CallSite(callee_name=callee, receiver=None, raw_text=f"{callee}()") for callee in edges.get(body_node, [])]

    return extract_calls


def test_collect_reachable_definitions_dedupes_diamond_shaped_calls():
    a, b, c, d = (_fn_with_marker(n) for n in "abcd")
    definitions = DefinitionIndex.build([a, b, c, d])
    extract_calls = _fake_extract_calls({"a": ["b", "c"], "b": ["d"], "c": ["d"]})

    reachable = collect_reachable_definitions(
        a, definitions, imports_by_file={}, source_bytes_by_file={"server.py": b""}, extract_calls=extract_calls,
    )

    assert {d.qualified_name for d in reachable} == {"a", "b", "c", "d"}  # d reached via both b and c, counted once


def test_collect_reachable_definitions_terminates_on_mutual_recursion():
    a, b = (_fn_with_marker(n) for n in "ab")
    definitions = DefinitionIndex.build([a, b])
    extract_calls = _fake_extract_calls({"a": ["b"], "b": ["a"]})

    reachable = collect_reachable_definitions(
        a, definitions, imports_by_file={}, source_bytes_by_file={"server.py": b""}, extract_calls=extract_calls,
    )

    assert {d.qualified_name for d in reachable} == {"a", "b"}


def test_collect_reachable_definitions_includes_start_def_alone_when_no_calls():
    a = _fn_with_marker("a")
    definitions = DefinitionIndex.build([a])
    extract_calls = _fake_extract_calls({})

    reachable = collect_reachable_definitions(
        a, definitions, imports_by_file={}, source_bytes_by_file={"server.py": b""}, extract_calls=extract_calls,
    )

    assert reachable == [a]


def _fn_with_marker_in(name: str, file: str) -> FunctionDef:
    return FunctionDef(qualified_name=name, bare_name=name, file=file, start_line=1, end_line=2, body_node=name)


def _fake_extract_calls_with_receiver(edges: dict[str, list[tuple[str, str]]]):
    def extract_calls(body_node, source_bytes):
        return [
            CallSite(callee_name=callee, receiver=receiver, raw_text=f"{receiver}.{callee}()")
            for receiver, callee in edges.get(body_node, [])
        ]

    return extract_calls


def test_collect_reachable_definitions_ignores_repo_wide_fallback_for_receiver_qualified_calls():
    # `obj.helper()` where `obj` isn't self/this and isn't a tracked import
    # alias must NOT be resolved just because `helper` happens to be unique
    # repo-wide -- real case: Accenture/mcp-bench's
    # `client.p.submission.fetch(post_id)` (an external PRAW call) got
    # wrongly matched this way to an unrelated bundled server's own `fetch`
    # tool, exploding the unbounded closure.
    start = _fn_with_marker_in("start", "server.py")
    unrelated = _fn_with_marker_in("helper", "other.py")
    definitions = DefinitionIndex.build([start, unrelated])
    extract_calls = _fake_extract_calls_with_receiver({"start": [("obj", "helper")]})

    reachable = collect_reachable_definitions(
        start, definitions, imports_by_file={}, source_bytes_by_file={"server.py": b"", "other.py": b""},
        extract_calls=extract_calls,
    )

    assert {d.qualified_name for d in reachable} == {"start"}  # helper NOT pulled in


def test_collect_reachable_definitions_still_uses_repo_wide_fallback_for_receiver_less_calls():
    # Unlike the qualified-call case above, a bare `helper()` call still
    # falls back to the repo-wide unique match -- this is the documented,
    # intentional way JS/TS's CommonJS `const { helper } = require(...)`
    # gets resolved at all (extract_imports doesn't track require() bindings).
    start = _fn_with_marker_in("start", "server.py")
    helper = _fn_with_marker_in("helper", "other.py")
    definitions = DefinitionIndex.build([start, helper])
    extract_calls = _fake_extract_calls({"start": ["helper"]})

    reachable = collect_reachable_definitions(
        start, definitions, imports_by_file={}, source_bytes_by_file={"server.py": b"", "other.py": b""},
        extract_calls=extract_calls,
    )

    assert {d.qualified_name for d in reachable} == {"start", "helper"}
