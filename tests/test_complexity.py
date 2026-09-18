from __future__ import annotations

from tree_sitter import Parser

from mcp_pipeline.extraction.complexity import cyclomatic_complexity
from mcp_pipeline.extraction.language_registry import spec_for
from mcp_pipeline.extraction.patterns.csharp_patterns import extract_definitions as csharp_extract_definitions
from mcp_pipeline.extraction.patterns.dart_patterns import extract_definitions as dart_extract_definitions
from mcp_pipeline.extraction.patterns.go_patterns import extract_definitions as go_extract_definitions
from mcp_pipeline.extraction.patterns.java_patterns import extract_definitions as java_extract_definitions
from mcp_pipeline.extraction.patterns.javascript_patterns import extract_definitions as javascript_extract_definitions
from mcp_pipeline.extraction.patterns.kotlin_patterns import extract_definitions as kotlin_extract_definitions
from mcp_pipeline.extraction.patterns.python_patterns import extract_definitions as python_extract_definitions
from mcp_pipeline.extraction.patterns.ruby_patterns import extract_definitions as ruby_extract_definitions
from mcp_pipeline.extraction.patterns.rust_patterns import extract_definitions as rust_extract_definitions
from mcp_pipeline.extraction.patterns.typescript_patterns import extract_definitions as typescript_extract_definitions


def _cc(language: str, extract_definitions, source: str, qualified_name: str | None = None) -> int:
    spec = spec_for(language)
    source_bytes = source.encode("utf-8")
    root = Parser(spec.ts_language).parse(source_bytes).root_node
    defs = extract_definitions(root, source_bytes, f"server.{spec.extensions[0].lstrip('.')}")
    if qualified_name is None:
        assert len(defs) == 1, f"expected exactly 1 function definition, found {len(defs)}"
        target = defs[0]
    else:
        matches = [d for d in defs if d.qualified_name == qualified_name]
        assert len(matches) == 1, f"expected exactly 1 definition named {qualified_name!r}, found {len(matches)}"
        target = matches[0]
    return cyclomatic_complexity(target.body_node, language, source_bytes)


def test_python_counts_if_elif_for_while_except_ternary_and_case_excludes_wildcard():
    source = '''
def f(x):
    if x > 0:
        pass
    elif x < 0:
        pass
    else:
        pass
    for i in range(10):
        pass
    while x:
        pass
    try:
        pass
    except ValueError:
        pass
    except TypeError:
        pass
    y = 1 if x else 2
    match x:
        case 1:
            pass
        case _:
            pass
'''
    # if(1) + elif(1) + for(1) + while(1) + except(2) + ternary(1) + case(1, "case _" excluded) = 8 -> CC 9
    assert _cc("Python", python_extract_definitions, source) == 9


def test_python_does_not_descend_into_nested_function_definitions():
    source = '''
def outer(x):
    if x:
        pass
    def inner(y):
        if y:
            pass
        if y:
            pass
    return inner
'''
    # Only outer's own "if x" counts; inner's 2 ifs are out of scope.
    assert _cc("Python", python_extract_definitions, source, qualified_name="outer") == 2


def test_javascript_counts_if_elseif_for_forin_while_do_catch_switch_ternary():
    source = '''
function f(x) {
  if (x > 0) {
  } else if (x < 0) {
  } else {
  }
  for (let i = 0; i < 10; i++) {}
  for (const k in x) {}
  while (x) {}
  do {} while (x);
  try {} catch (e) {}
  switch (x) {
    case 1: break;
    case 2: break;
    default: break;
  }
  let y = x ? 1 : 2;
}
'''
    # if(2, outer+else-if) + for(1) + for-in(1) + while(1) + do(1) + catch(1)
    # + switch case(2, default excluded) + ternary(1) = 10 -> CC 11
    assert _cc("JavaScript", javascript_extract_definitions, source) == 11


def test_javascript_does_not_descend_into_nested_arrow_function():
    source = '''
function outer(x) {
  if (x) {}
  const inner = (y) => {
    if (y) {}
    if (y) {}
  };
  return inner;
}
'''
    assert _cc("JavaScript", javascript_extract_definitions, source, qualified_name="outer") == 2


def test_typescript_counts_if_elseif_for_while_catch_switch_ternary():
    source = '''
function f(x: number) {
  if (x > 0) {
  } else if (x < 0) {
  } else {
  }
  for (let i = 0; i < 10; i++) {}
  while (x) {}
  try {} catch (e) {}
  switch (x) {
    case 1: break;
    case 2: break;
    default: break;
  }
  let y = x ? 1 : 2;
}
'''
    # if(2) + for(1) + while(1) + catch(1) + switch case(2) + ternary(1) = 8 -> CC 9
    assert _cc("TypeScript", typescript_extract_definitions, source) == 9


def test_java_counts_if_elseif_for_enhancedfor_while_do_catch_switch_ternary():
    source = '''
class A {
  int f(int x) {
    if (x > 0) {
    } else if (x < 0) {
    } else {
    }
    for (int i = 0; i < 10; i++) {}
    for (int k : arr) {}
    while (x > 0) {}
    do {} while (x > 0);
    try {} catch (Exception e) {}
    switch (x) {
      case 1: break;
      case 2: break;
      default: break;
    }
    int y = x > 0 ? 1 : 2;
    return y;
  }
}
'''
    # if(2) + for(1) + enhanced-for(1) + while(1) + do(1) + catch(1)
    # + switch case(2, default excluded) + ternary(1) = 10 -> CC 11
    assert _cc("Java", java_extract_definitions, source) == 11


def test_csharp_counts_if_elseif_for_foreach_while_do_catch_switch_ternary():
    source = '''
class A {
  int F(int x) {
    if (x > 0) {
    } else if (x < 0) {
    } else {
    }
    for (int i = 0; i < 10; i++) {}
    foreach (var k in arr) {}
    while (x > 0) {}
    do {} while (x > 0);
    try {} catch (Exception e) {}
    switch (x) {
      case 1: break;
      case 2: break;
      default: break;
    }
    int y = x > 0 ? 1 : 2;
    return y;
  }
}
'''
    # if(2) + for(1) + foreach(1) + while(1) + do(1) + catch(1)
    # + switch case(2, default excluded) + ternary(1) = 10 -> CC 11
    assert _cc("C#", csharp_extract_definitions, source) == 11


def test_go_counts_if_elseif_for_switchcase_selectcase_excludes_defaults():
    source = '''
package main

func f(x int) int {
    if x > 0 {
    } else if x < 0 {
    } else {
    }
    for i := 0; i < 10; i++ {}
    switch x {
    case 1:
    case 2:
    default:
    }
    select {
    case <-ch:
    default:
    }
    return x
}
'''
    # if(2) + for(1) + expression_case(2, default excluded) + communication_case(1, default excluded) = 6 -> CC 7
    assert _cc("Go", go_extract_definitions, source) == 7


def test_rust_counts_if_elseif_for_while_loop_matcharm_excludes_wildcard():
    source = '''
fn f(x: i32) -> i32 {
    if x > 0 {
    } else if x < 0 {
    } else {
    }
    for i in 0..10 {}
    while x > 0 {}
    loop { break; }
    match x {
        1 => {},
        2 => {},
        _ => {},
    }
    x
}
'''
    # if(2) + for(1) + while(1) + loop(1) + match_arm(2, "_" excluded) = 7 -> CC 8
    assert _cc("Rust", rust_extract_definitions, source) == 8


def test_ruby_counts_if_elsif_for_while_until_unless_when_rescue_ternary():
    source = '''
def f(x)
  if x > 0
  elsif x < 0
  else
  end
  for i in 0..10
  end
  while x
  end
  until x
  end
  unless x
  end
  case x
  when 1
  when 2
  else
  end
  begin
  rescue => e
  end
  y = x ? 1 : 2
end
'''
    # if(1) + elsif(1) + for(1) + while(1) + until(1) + unless(1) + when(2) + rescue(1) + ternary(1) = 10 -> CC 11
    assert _cc("Ruby", ruby_extract_definitions, source) == 11


def test_dart_counts_if_elseif_for_forin_while_do_catch_switch_ternary():
    source = '''
void f(int x) {
  if (x > 0) {
  } else if (x < 0) {
  } else {
  }
  for (int i = 0; i < 10; i++) {}
  for (var k in arr) {}
  while (x > 0) {}
  do {} while (x > 0);
  try {} catch (e) {}
  switch (x) {
    case 1: break;
    case 2: break;
    default: break;
  }
  var y = x > 0 ? 1 : 2;
}
'''
    # if(2) + for(2, classic+for-in) + while(1) + do(1) + catch(1)
    # + switch case(2, default excluded) + ternary(1) = 10 -> CC 11
    assert _cc("Dart", dart_extract_definitions, source) == 11


def test_kotlin_counts_if_elseif_for_while_dowhile_catch_when_excludes_else():
    source = '''
fun f(x: Int): Int {
    if (x > 0) {
    } else if (x < 0) {
    } else {
    }
    for (i in 0..10) {}
    while (x > 0) {}
    do {} while (x > 0)
    try {} catch (e: Exception) {}
    when (x) {
        1 -> {}
        2 -> {}
        else -> {}
    }
    return x
}
'''
    # if(2) + for(1) + while(1) + do-while(1) + catch(1) + when_entry(2, else excluded) = 8 -> CC 9
    assert _cc("Kotlin", kotlin_extract_definitions, source) == 9
