#!/usr/bin/env python
"""Gera um dashboard HTML autocontido a partir de `data/evaluations/{judge_id}.jsonl`.

Recalcula tudo a cada execução (nenhum número fica hardcoded) -- pode rodar a qualquer
momento, inclusive com a Etapa 3 (run_step3.py) ainda em andamento, para ver um snapshot
atualizado dos resultados parciais. Mesmas 4 seções do protótipo original: tiles de status
geral, ranking por componente da rubrica, comparação de cenários e tabela por juiz.

O HTML gerado não depende de nada externo além de fontes do Google Fonts -- pode ser aberto
direto no navegador ou publicado como Artifact.

Uso:
  uv run python scripts/generate_dashboard.py
  uv run python scripts/generate_dashboard.py --output caminho/custom.html
  uv run python scripts/generate_dashboard.py --judge gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from statistics import mean, stdev

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.evaluation.prompts import PROMPT_VERSION, RUBRIC_COMPONENTS
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("generate_dashboard")

# Below this many "ok" evaluations, a judge's mean is flagged as statistically meaningless
# instead of presented at face value -- avoids the dashboard implying a 2- or 3-sample mean
# is comparable to one backed by hundreds of evaluations.
MIN_TRUSTWORTHY_N = 30


def load_records(eval_dir: Path, judge_id: str | None) -> list[dict]:
    records: list[dict] = []
    paths = [eval_dir / f"{judge_id}.jsonl"] if judge_id else sorted(eval_dir.glob("*.jsonl"))
    for path in paths:
        if not path.exists():
            logger.warning("%s não existe, pulando", path)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# Same tiering as scripts/dedupe_evaluations.py -- kept here as its own copy (each script in
# this repo reads the raw jsonl directly rather than importing another script) rather than
# assuming a prior dedupe pass ran: run_step3.py --retry-failed appends a new line without
# removing the old "error" one, so a live pipeline run can reintroduce duplicates between
# dashboard generations, which would otherwise silently inflate every count on this page.
_STATUS_RANK = {"ok": 2, "refused": 2, "error": 1}


def dedupe_records(records: list[dict]) -> list[dict]:
    kept: dict[tuple[str, str, str, str], dict] = {}
    order: list[tuple[str, str, str, str]] = []
    for record in records:
        key = (record["tool_uid"], record["scenario"], record["judge"]["id"], record.get("prompt_version", ""))
        current = kept.get(key)
        if current is None:
            order.append(key)
            kept[key] = record
            continue
        candidate_rank = _STATUS_RANK.get(record.get("status"), 0)
        current_rank = _STATUS_RANK.get(current.get("status"), 0)
        if candidate_rank > current_rank or (
            candidate_rank == current_rank
            and (record.get("evaluated_at") or "") >= (current.get("evaluated_at") or "")
        ):
            kept[key] = record
    return [kept[key] for key in order]


def component_stats(records: list[dict]) -> dict[str, dict]:
    """Mean/stdev/n per rubric component, over `records` (already filtered to status=ok)."""
    stats = {}
    for key, label, _ in RUBRIC_COMPONENTS:
        scores = [r["scores"][key]["score"] for r in records if r.get("scores", {}).get(key)]
        if not scores:
            continue
        stats[key] = {
            "label": label,
            "mean": mean(scores),
            "sd": stdev(scores) if len(scores) > 1 else 0.0,
            "n": len(scores),
        }
    return stats


def compute_breakdown(ok_records: list[dict]) -> dict:
    """Rubric-component ranking + scenario comparison over one slice of ok records --
    shared by the combined ("Todos") view and each per-judge tab, so they read identically.
    """
    components = component_stats(ok_records)
    ranked_keys = sorted(components, key=lambda k: components[k]["mean"], reverse=True)
    rubric_components = [
        {"key": k, "label": components[k]["label"], "mean": components[k]["mean"],
         "sd": components[k]["sd"], "n": components[k]["n"]}
        for k in ranked_keys
    ]

    scenarios = sorted({r["scenario"] for r in ok_records})
    by_scenario = {s: component_stats([r for r in ok_records if r["scenario"] == s]) for s in scenarios}
    scenario_comparison = [
        {"label": components[k]["label"], **{s: by_scenario[s].get(k, {}).get("mean") for s in scenarios}}
        for k in ranked_keys
    ]

    return {"rubric_components": rubric_components, "scenario_comparison": scenario_comparison, "scenario_keys": scenarios}


def build_version_summary(records: list[dict], active_version: str) -> dict:
    """Lightweight per-prompt_version counts (no rubric breakdown) -- checkpoint_key()
    embeds prompt_version, so bumping it (see prompts.py's changelog) makes every old
    result invisible to should_skip() without deleting it: old and new rubric text/schema
    coexist in the same jsonl tagged by their own version. This is what lets a stale
    version be told apart from the one Visão geral/Tools are actually scoped to, and
    surfaces how many rows a purge (scripts/purge_prompt_version.py) would remove.
    """
    versions: dict[str, dict] = {}
    for r in records:
        version = r.get("prompt_version") or "(sem versão)"
        v = versions.setdefault(version, {"version": version, "total": 0, "ok": 0, "error": 0, "refused": 0, "judges": {}})
        v["total"] += 1
        status = r.get("status")
        if status in ("ok", "error", "refused"):
            v[status] += 1
        jid = r["judge"]["id"]
        v["judges"][jid] = v["judges"].get(jid, 0) + 1

    for v in versions.values():
        v["judges"] = [{"id": jid, "n": n} for jid, n in sorted(v["judges"].items(), key=lambda kv: -kv[1])]
        v["is_active"] = v["version"] == active_version

    return dict(sorted(versions.items(), key=lambda kv: kv[0], reverse=True))


def build_dashboard_data(records: list[dict], prompt_version: str | None = None) -> dict:
    available_versions = sorted({r.get("prompt_version") or "(sem versão)" for r in records})
    active_version = prompt_version or available_versions[-1]
    version_summary = build_version_summary(records, active_version)

    scoped = [r for r in records if (r.get("prompt_version") or "(sem versão)") == active_version]
    ok_records = [r for r in scoped if r.get("status") == "ok"]

    overall = {
        "total": len(scoped),
        "ok": len(ok_records),
        "error": sum(1 for r in scoped if r.get("status") == "error"),
        "refused": sum(1 for r in scoped if r.get("status") == "refused"),
    }

    judge_ids = sorted({r["judge"]["id"] for r in scoped}, key=lambda jid: -sum(1 for r in scoped if r["judge"]["id"] == jid))
    judges = []
    breakdowns = {"__all__": compute_breakdown(ok_records)}
    for jid in judge_ids:
        judge_records = [r for r in scoped if r["judge"]["id"] == jid]
        judge_ok = [r for r in judge_records if r.get("status") == "ok"]
        breakdown = compute_breakdown(judge_ok)
        breakdowns[jid] = breakdown
        overall_mean = mean([c["mean"] for c in breakdown["rubric_components"]]) if breakdown["rubric_components"] else None
        judges.append({
            "id": jid,
            "provider": judge_records[0]["judge"]["provider"] if judge_records else "?",
            "n": len(judge_records),
            "ok": len(judge_ok),
            "error": sum(1 for r in judge_records if r.get("status") == "error"),
            "mean": overall_mean,
            "flag": f"N={len(judge_ok)} — amostra insuficiente" if len(judge_ok) < MIN_TRUSTWORTHY_N else None,
        })

    tools_evaluated = len({r["tool_uid"] for r in ok_records})

    return {
        "meta": {
            "generated_at": datetime.datetime.now().astimezone().strftime("%d/%m/%Y, %H:%M (%Z)"),
            "prompt_version": PROMPT_VERSION,
            "active_prompt_version": active_version,
            "tools_evaluated": tools_evaluated,
        },
        "overall": overall,
        "breakdowns": breakdowns,
        "judges": judges,
        "tools": build_tools_data(scoped),
        "prompt_versions": version_summary,
    }


def build_tools_data(records: list[dict]) -> list[dict]:
    """One row per tool_uid, with per-scenario scores averaged across whichever judge(s)
    evaluated it -- most tools right now only have one judge's worth of data, but this
    stays correct once a second judge's results land on the same tool/scenario.

    Technical failures (status=error, e.g. the 429s a judge hit before the rate limiter
    fix) are dropped entirely here rather than surfaced as a row -- a transient API error
    isn't a property of the tool or its description, so listing it next to real scores
    would misrepresent what's actually known about that tool/scenario. A tool with no
    successful evaluation in a scenario just shows as not-yet-evaluated ("—"), identical to
    one nobody has attempted yet -- both cases mean "no signal", not "known bad".
    """
    by_tool: dict[str, dict] = {}
    for r in records:
        if r.get("status") != "ok" or not r.get("scores"):
            continue
        entry = by_tool.setdefault(r["tool_uid"], {
            "tool_uid": r["tool_uid"],
            "name": r["tool"]["name"],
            "qualified_name": r["tool"]["qualified_name"],
            "repo": r["repo"]["name_with_owner"],
            "language": r["repo"].get("primary_language") or "—",
            "by_scenario": {},
        })
        component_scores = {k: r["scores"][k]["score"] for k, _, _ in RUBRIC_COMPONENTS if r["scores"].get(k)}
        if not component_scores:
            continue
        entry["by_scenario"].setdefault(r["scenario"], []).append({
            "judge_id": r["judge"]["id"],
            "mean": mean(component_scores.values()),
            "components": component_scores,
        })

    tools = []
    for entry in by_tool.values():
        scenario_summary = {}
        scenario_means = []
        judges_present: set[str] = set()
        for scenario, judge_rows in entry["by_scenario"].items():
            scenario_mean = mean(jr["mean"] for jr in judge_rows)
            scenario_means.append(scenario_mean)
            scenario_summary[scenario] = {"mean": scenario_mean, "judges": judge_rows}
            judges_present.update(jr["judge_id"] for jr in judge_rows)
        tools.append({
            "tool_uid": entry["tool_uid"],
            "name": entry["name"],
            "qualified_name": entry["qualified_name"],
            "repo": entry["repo"],
            "language": entry["language"],
            "scenarios": scenario_summary,
            "judges_present": sorted(judges_present),
            "overall_mean": mean(scenario_means) if scenario_means else None,
        })

    tools.sort(key=lambda t: (t["overall_mean"] is None, -(t["overall_mean"] or 0)))
    return tools


HTML_TEMPLATE = """<title>Rubrica MCP</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    color-scheme: light;
    --surface-0: #fcfcfb;
    --surface-1: #f4f2ec;
    --surface-2: #ebe8df;
    --line: #dcd8cc;
    --text-primary: #17160f;
    --text-secondary: #55523f;
    --text-muted: #8a866e;
    --accent-1: #2a78d6;
    --accent-2: #eb6834;
    --status-good: #0ca30c;
    --status-good-soft: #e2f5df;
    --status-critical: #d03b3b;
    --status-critical-soft: #fbe4e1;
    --status-warning: #b8790a;
    --shadow: 0 1px 2px rgba(23,22,15,.06), 0 8px 24px -12px rgba(23,22,15,.16);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #17160f; --surface-1: #201f16; --surface-2: #2a2919; --line: #3a3826;
      --text-primary: #f4f2ea; --text-secondary: #c7c3ac; --text-muted: #8f8b71;
      --accent-1: #3987e5; --accent-2: #d95926;
      --status-good: #29c229; --status-good-soft: #163318;
      --status-critical: #e66767; --status-critical-soft: #3a1c1c; --status-warning: #d99a2b;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #17160f; --surface-1: #201f16; --surface-2: #2a2919; --line: #3a3826;
    --text-primary: #f4f2ea; --text-secondary: #c7c3ac; --text-muted: #8f8b71;
    --accent-1: #3987e5; --accent-2: #d95926;
    --status-good: #29c229; --status-good-soft: #163318;
    --status-critical: #e66767; --status-critical-soft: #3a1c1c; --status-warning: #d99a2b;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
  }
  * { box-sizing: border-box; }
  body { background: var(--surface-0); color: var(--text-primary); font-family: "IBM Plex Sans", system-ui, sans-serif; line-height: 1.5; }
  .wrap { max-width: 920px; margin: 0 auto; padding: 48px 24px 96px; }
  h1, h2 { font-family: "Newsreader", Georgia, serif; text-wrap: balance; margin: 0; }
  .eyebrow { font-family: "IBM Plex Mono", monospace; font-size: 12px; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted); }
  .tabular { font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", monospace; }
  header { margin-bottom: 40px; }
  header .eyebrow { margin-bottom: 10px; }
  h1 { font-size: clamp(30px, 4vw, 40px); font-weight: 600; letter-spacing: -.01em; }
  header p.lede { margin: 14px 0 0; max-width: 62ch; color: var(--text-secondary); font-size: 15.5px; }
  .run-meta { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 20px; font-size: 13px; color: var(--text-muted); }
  .run-meta span b { color: var(--text-secondary); font-weight: 500; }
  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px 3px 8px; border-radius: 100px; background: var(--status-warning); color: var(--surface-0); font-family: "IBM Plex Mono", monospace; font-size: 11.5px; letter-spacing: .02em; font-weight: 500; }
  .badge::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: pulse 1.8s ease-in-out infinite; }
  @media (prefers-reduced-motion: reduce) { .badge::before { animation: none; } }
  @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(184,121,10,.5); } 70% { box-shadow: 0 0 0 6px rgba(184,121,10,0); } 100% { box-shadow: 0 0 0 0 rgba(184,121,10,0); } }
  .tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; margin-bottom: 48px; }
  .tile { background: var(--surface-1); padding: 18px 20px; }
  .tile .label { font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--text-muted); }
  .tile .value { display: block; margin-top: 8px; font-family: "IBM Plex Mono", monospace; font-size: 28px; font-weight: 500; letter-spacing: -.01em; }
  .tile .sub { display: block; margin-top: 3px; font-size: 12.5px; color: var(--text-muted); }
  .tile.ok .value { color: var(--status-good); }
  .tile.error .value { color: var(--status-critical); }
  @media (max-width: 640px) { .tiles { grid-template-columns: repeat(2, 1fr); } }
  section { margin-bottom: 52px; }
  section > h2 { font-size: 22px; font-weight: 600; margin-bottom: 4px; }
  section > .section-note { color: var(--text-secondary); font-size: 14px; max-width: 68ch; margin: 0 0 22px; }
  .legend { display: flex; gap: 18px; margin-bottom: 16px; font-size: 13px; color: var(--text-secondary); }
  .legend .key { display: inline-flex; align-items: center; gap: 7px; }
  .legend .swatch { width: 10px; height: 10px; border-radius: 3px; }
  .chart { display: flex; flex-direction: column; gap: 14px; }
  .bar-row { display: grid; grid-template-columns: 168px 1fr; align-items: center; gap: 14px; }
  .bar-row .row-label { font-size: 13.5px; color: var(--text-secondary); text-align: right; }
  .bar-track { position: relative; height: 22px; background: var(--surface-2); border-radius: 5px; }
  .bar-fill { position: absolute; top: 2px; left: 2px; bottom: 2px; border-radius: 4px; background: var(--accent-1); transition: filter .15s ease; }
  .bar-fill:hover, .bar-fill:focus-visible { filter: brightness(1.08); outline: none; }
  .bar-fill .val { position: absolute; right: -34px; top: 50%; transform: translateY(-50%); font-family: "IBM Plex Mono", monospace; font-size: 12.5px; font-weight: 500; color: var(--text-primary); }
  .bar-track .grid-tick { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--line); }
  .bar-track .grid-tick.major { background: var(--text-muted); opacity: .35; }
  .bar-row.grouped .bar-track { display: flex; flex-direction: column; gap: 3px; background: none; height: auto; }
  .bar-row.grouped .bar-track .sub-track { position: relative; height: 15px; background: var(--surface-2); border-radius: 4px; }
  .bar-row.grouped .bar-fill { top: 1px; bottom: 1px; }
  #tooltip { position: fixed; pointer-events: none; z-index: 50; background: var(--text-primary); color: var(--surface-0); font-family: "IBM Plex Mono", monospace; font-size: 12px; line-height: 1.5; padding: 7px 10px; border-radius: 7px; box-shadow: var(--shadow); opacity: 0; transform: translate(-50%, -100%); transition: opacity .1s ease; white-space: nowrap; }
  #tooltip.show { opacity: 1; }
  #tooltip b { font-weight: 600; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  thead th { text-align: left; font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: .04em; text-transform: uppercase; color: var(--text-muted); font-weight: 500; padding: 0 12px 10px; border-bottom: 1px solid var(--line); }
  thead th.num, tbody td.num { text-align: right; }
  tbody td { padding: 14px 12px; border-bottom: 1px solid var(--line); vertical-align: middle; }
  tbody tr:last-child td { border-bottom: none; }
  tbody td.judge-id { font-weight: 500; }
  tbody td.judge-id .provider { display: block; font-size: 12px; color: var(--text-muted); font-weight: 400; margin-top: 1px; }
  .pill { display: inline-flex; align-items: center; gap: 5px; padding: 2px 8px; border-radius: 100px; font-family: "IBM Plex Mono", monospace; font-size: 12px; font-weight: 500; }
  .pill.ok { background: var(--status-good-soft); color: var(--status-good); }
  .pill.error { background: var(--status-critical-soft); color: var(--status-critical); }
  .n-flag { font-size: 11.5px; color: var(--status-warning); font-family: "IBM Plex Mono", monospace; }
  .overflow-x { overflow-x: auto; }
  footer { border-top: 1px solid var(--line); padding-top: 24px; font-size: 13px; color: var(--text-muted); }
  footer p { max-width: 68ch; margin: 0 0 10px; }
  footer code { font-family: "IBM Plex Mono", monospace; background: var(--surface-1); padding: 1px 5px; border-radius: 4px; font-size: 12px; color: var(--text-secondary); }

  /* ---------- tabs ---------- */
  .tabs { display: flex; gap: 4px; margin-top: 28px; border-bottom: 1px solid var(--line); }
  .tab-btn {
    font: inherit; font-family: "IBM Plex Sans", sans-serif; font-size: 14.5px; font-weight: 500;
    background: none; border: none; cursor: pointer;
    color: var(--text-muted); padding: 10px 4px; margin-right: 20px;
    border-bottom: 2px solid transparent; transform: translateY(1px);
  }
  .tab-btn:hover { color: var(--text-primary); }
  .tab-btn[aria-selected="true"] { color: var(--text-primary); border-bottom-color: var(--accent-1); }
  .tab-btn:focus-visible { outline: 2px solid var(--accent-1); outline-offset: 2px; }
  [role="tabpanel"] { padding-top: 40px; }

  /* ---------- AI sub-tabs (inside Visão geral) ---------- */
  .ai-tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 40px; }
  .ai-tab-btn {
    font: inherit; font-family: "IBM Plex Mono", monospace; font-size: 12.5px; font-weight: 500;
    cursor: pointer; padding: 7px 14px; border-radius: 100px;
    border: 1px solid var(--line); background: var(--surface-1); color: var(--text-secondary);
  }
  .ai-tab-btn:hover { border-color: var(--accent-1); color: var(--text-primary); }
  .ai-tab-btn[aria-selected="true"] { background: var(--accent-1); border-color: var(--accent-1); color: var(--surface-0); }
  .ai-tab-btn:focus-visible { outline: 2px solid var(--accent-1); outline-offset: 2px; }

  /* ---------- tools tab ---------- */
  .tools-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 16px; }
  .tools-search {
    flex: 1; min-width: 180px; max-width: 320px; font: inherit; font-family: "IBM Plex Sans", sans-serif; font-size: 13.5px;
    padding: 9px 12px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--surface-1); color: var(--text-primary);
  }
  .tools-search:focus-visible { outline: 2px solid var(--accent-1); outline-offset: 1px; }
  .tools-filter {
    font: inherit; font-family: "IBM Plex Sans", sans-serif; font-size: 13px;
    padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line);
    background: var(--surface-1); color: var(--text-secondary);
  }
  .tools-filter:focus-visible { outline: 2px solid var(--accent-1); outline-offset: 1px; }
  .tools-count { margin-left: auto; font-size: 12.5px; color: var(--text-muted); font-family: "IBM Plex Mono", monospace; white-space: nowrap; }
  th.sortable { cursor: pointer; user-select: none; }
  th.sortable:hover { color: var(--text-secondary); }
  th.sortable .arrow { display: inline-block; width: 1em; opacity: .5; }
  th.sortable[aria-sort="ascending"] .arrow, th.sortable[aria-sort="descending"] .arrow { opacity: 1; color: var(--accent-1); }
  tr.tool-row { cursor: pointer; }
  tr.tool-row:hover td { background: var(--surface-1); }
  tr.tool-row .name { font-weight: 500; }
  tr.tool-row .qualified { display: block; font-size: 11.5px; color: var(--text-muted); font-family: "IBM Plex Mono", monospace; margin-top: 1px; }
  .score-cell { font-family: "IBM Plex Mono", monospace; font-weight: 500; }
  .score-cell.missing { color: var(--text-muted); font-weight: 400; }
  .score-cell.err { color: var(--status-critical); }
  .expand-icon { display: inline-block; width: 14px; color: var(--text-muted); transition: transform .12s ease; }
  tr.tool-row[aria-expanded="true"] .expand-icon { transform: rotate(90deg); }
  tr.tool-detail td { padding: 0 12px 20px; border-bottom: 1px solid var(--line); background: var(--surface-1); }
  tr.tool-detail .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; padding-top: 4px; }
  .detail-scenario h4 { margin: 0 0 8px; font-family: "IBM Plex Sans", sans-serif; font-size: 12.5px; font-weight: 600; color: var(--text-secondary); }
  .detail-scenario .comp-row { display: flex; justify-content: space-between; font-size: 12.5px; padding: 3px 0; color: var(--text-secondary); }
  .detail-scenario .comp-row b { color: var(--text-primary); font-family: "IBM Plex Mono", monospace; font-weight: 500; }
  .detail-scenario .err-msg { font-size: 12px; color: var(--status-critical); font-family: "IBM Plex Mono", monospace; word-break: break-word; }
  .empty-state { padding: 40px 0; text-align: center; color: var(--text-muted); font-size: 14px; }

  /* ---------- versions tab ---------- */
  .version-badge { display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px; border-radius: 100px; font-family: "IBM Plex Mono", monospace; font-size: 12px; font-weight: 500; background: var(--status-good-soft); color: var(--status-good); }
  .version-badge.stale { background: var(--surface-2); color: var(--text-muted); }
  .purge-cmd { display: block; margin-top: 6px; font-family: "IBM Plex Mono", monospace; font-size: 11.5px; background: var(--surface-2); color: var(--text-secondary); padding: 6px 9px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; }
  .judges-inline { font-size: 12.5px; color: var(--text-secondary); }
  .judges-inline b { font-family: "IBM Plex Mono", monospace; color: var(--text-primary); font-weight: 500; }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">Etapa 3 · LLM-as-Judge · Hasan et al. (2026), rubrica de 6 componentes</div>
    <h1>Rubrica MCP</h1>
    <p class="lede">Como as descrições de ferramentas MCP avaliadas até agora se saem nos seis componentes da rubrica, em escala Likert de 1 a 5.</p>
    <div class="run-meta">
      <span class="badge">snapshot gerado localmente</span>
      <span>Gerado em: <b id="meta-generated-at">—</b></span>
      <span>Tools avaliadas: <b id="meta-tools">—</b></span>
      <span>Prompt version exibida: <b id="meta-prompt-version">—</b></span>
    </div>
    <div class="tabs" role="tablist">
      <button class="tab-btn" role="tab" id="tab-btn-overview" aria-controls="tab-overview" aria-selected="true">Visão geral</button>
      <button class="tab-btn" role="tab" id="tab-btn-tools" aria-controls="tab-tools" aria-selected="false">Tools</button>
      <button class="tab-btn" role="tab" id="tab-btn-versions" aria-controls="tab-versions" aria-selected="false">Versões do prompt</button>
    </div>
  </header>

  <div id="tab-overview" role="tabpanel" aria-labelledby="tab-btn-overview">
    <div class="tiles">
      <div class="tile"><span class="label">Avaliações</span><span class="value tabular" id="tile-total">—</span><span class="sub">tool × cenário × juiz</span></div>
      <div class="tile ok"><span class="label">Sucesso</span><span class="value tabular" id="tile-ok">—</span><span class="sub" id="tile-ok-pct">—</span></div>
      <div class="tile error"><span class="label">Erro</span><span class="value tabular" id="tile-error">—</span><span class="sub" id="tile-error-pct">—</span></div>
      <div class="tile"><span class="label">Recusas</span><span class="value tabular" id="tile-refused">—</span><span class="sub">segurança</span></div>
    </div>

    <div class="ai-tabs" id="ai-tabs" role="tablist" aria-label="Filtrar rubrica por juiz"></div>

    <section>
      <h2>Por componente da rubrica</h2>
      <p class="section-note" id="components-note">Média de todas as avaliações concluídas com sucesso, nos cenários combinados. Ordenado do melhor para o pior.</p>
      <div class="chart" id="chart-components"></div>
    </section>

    <section id="section-scenarios">
      <h2>Comparação de cenários</h2>
      <p class="section-note" id="scenario-note"></p>
      <div class="legend" id="scenario-legend"></div>
      <div class="chart" id="chart-scenarios"></div>
    </section>

    <section>
      <h2>Por juiz</h2>
      <p class="section-note">Juízes com poucas avaliações concluídas têm a média sinalizada como não confiável.</p>
      <div class="overflow-x">
        <table>
          <thead><tr><th>Juiz</th><th class="num">Avaliações</th><th>Status</th><th class="num">Média geral</th></tr></thead>
          <tbody id="judge-rows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <div id="tab-tools" role="tabpanel" aria-labelledby="tab-btn-tools" hidden>
    <section style="margin-bottom: 0;">
      <h2>Todas as tools</h2>
      <p class="section-note">Uma linha por tool avaliada com sucesso (chave <code>tool_uid</code>), com a média por cenário e, se houver mais de um juiz na mesma tool/cenário, a média entre eles. Avaliações com erro técnico não aparecem aqui -- contam só no resumo geral. Clique numa linha para ver o detalhe por componente.</p>
      <div class="tools-toolbar">
        <input type="text" id="tools-search" class="tools-search" placeholder="Buscar por nome ou repositório…" autocomplete="off">
        <select id="tools-filter-language" class="tools-filter"><option value="">Toda linguagem</option></select>
        <select id="tools-filter-judge" class="tools-filter"><option value="">Todo juiz</option></select>
        <select id="tools-filter-coverage" class="tools-filter">
          <option value="">Todo cenário</option>
          <option value="both">Ambos os cenários</option>
          <option value="description_only">Só description_only</option>
          <option value="with_source">Só with_source</option>
        </select>
        <span class="tools-count" id="tools-count"></span>
      </div>
      <div class="overflow-x">
        <table>
          <thead>
            <tr>
              <th class="sortable" data-sort="name">Tool <span class="arrow">↕</span></th>
              <th class="sortable" data-sort="repo">Repositório <span class="arrow">↕</span></th>
              <th>Linguagem</th>
              <th class="num sortable" data-sort="description_only">description_only <span class="arrow">↕</span></th>
              <th class="num sortable" data-sort="with_source">with_source <span class="arrow">↕</span></th>
              <th class="num sortable" data-sort="overall_mean">Média geral <span class="arrow">↕</span></th>
            </tr>
          </thead>
          <tbody id="tools-rows"></tbody>
        </table>
      </div>
      <div class="empty-state" id="tools-empty" hidden>Nenhuma tool encontrada para essa busca.</div>
    </section>
  </div>

  <div id="tab-versions" role="tabpanel" aria-labelledby="tab-btn-versions" hidden>
    <section>
      <h2>Versões do prompt</h2>
      <p class="section-note">Cada linha do jsonl carrega o <code>prompt_version</code> vigente quando foi gerada (ver <code>PROMPT_VERSION</code> em <code>prompts.py</code>) -- bumpar a versão não apaga resultados antigos, só os torna invisíveis para o checkpoint (uma nova versão nunca reaproveita avaliações da anterior). "Visão geral" e "Tools" mostram só a versão mais recente presente nos dados, para não misturar rubricas diferentes na mesma média.</p>
      <div class="overflow-x">
        <table>
          <thead>
            <tr>
              <th>Versão</th>
              <th class="num">Avaliações</th>
              <th>Status</th>
              <th>Juízes</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="version-rows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <footer>
    <p>Gerado por <code>scripts/generate_dashboard.py</code> a partir de <code>data/evaluations/*.jsonl</code> -- recalculado do zero a cada execução, sem números hardcoded. Deduplicação de retries (<code>--retry-failed</code>) já embutida na leitura, mantendo só a avaliação mais recente/bem-sucedida por tool × cenário × juiz.</p>
  </footer>
</div>

<div id="tooltip"></div>

<script>
  const DATA = __DASHBOARD_DATA__;
  const SCALE_MAX = 5;
  const SERIES_COLORS = ["var(--accent-1)", "var(--accent-2)"];
  const scenarioKeys = DATA.breakdowns.__all__.scenario_keys; // used by the Tools tab (detail panel, filters)

  const tooltip = document.getElementById("tooltip");
  function showTooltip(el, html) {
    const r = el.getBoundingClientRect();
    tooltip.innerHTML = html;
    tooltip.style.left = (r.left + r.width / 2) + "px";
    tooltip.style.top = (r.top - 8) + "px";
    tooltip.classList.add("show");
  }
  function hideTooltip() { tooltip.classList.remove("show"); }
  function wireTooltip(el, html) {
    el.addEventListener("mouseenter", () => showTooltip(el, html));
    el.addEventListener("focus", () => showTooltip(el, html));
    el.addEventListener("mouseleave", hideTooltip);
    el.addEventListener("blur", hideTooltip);
  }

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.setAttribute("aria-selected", "false"));
      document.querySelectorAll('[role="tabpanel"]').forEach(p => { p.hidden = true; });
      btn.setAttribute("aria-selected", "true");
      document.getElementById(btn.getAttribute("aria-controls")).hidden = false;
    });
  });

  document.getElementById("meta-generated-at").textContent = DATA.meta.generated_at;
  document.getElementById("meta-tools").textContent = DATA.meta.tools_evaluated;
  document.getElementById("meta-prompt-version").textContent = DATA.meta.active_prompt_version;

  document.getElementById("tile-total").textContent = DATA.overall.total;
  document.getElementById("tile-ok").textContent = DATA.overall.ok;
  document.getElementById("tile-ok-pct").textContent = (100 * DATA.overall.ok / DATA.overall.total).toFixed(1) + "%";
  document.getElementById("tile-error").textContent = DATA.overall.error;
  document.getElementById("tile-error-pct").textContent = (100 * DATA.overall.error / DATA.overall.total).toFixed(1) + "%";
  document.getElementById("tile-refused").textContent = DATA.overall.refused;

  // ---- rubric-by-component / scenario-comparison charts, parameterized by which
  // breakdown (combined "__all__" or a single judge_id) is currently selected ----
  function renderComponentsChart(breakdown) {
    const componentsEl = document.getElementById("chart-components");
    componentsEl.innerHTML = "";
    breakdown.rubric_components.forEach(c => {
      const row = document.createElement("div");
      row.className = "bar-row";
      const pct = (c.mean / SCALE_MAX) * 100;
      row.innerHTML = `
        <div class="row-label">${c.label}</div>
        <div class="bar-track">
          <div class="grid-tick" style="left:20%"></div>
          <div class="grid-tick major" style="left:40%"></div>
          <div class="grid-tick" style="left:60%"></div>
          <div class="grid-tick" style="left:80%"></div>
          <div class="bar-fill" tabindex="0" style="width:${pct}%">
            <span class="val tabular">${c.mean.toFixed(2)}</span>
          </div>
        </div>`;
      wireTooltip(row.querySelector(".bar-fill"), `<b>${c.label}</b><br>média ${c.mean.toFixed(2)} · dp ${c.sd.toFixed(2)} · N=${c.n}`);
      componentsEl.appendChild(row);
    });
  }

  function renderScenariosChart(breakdown) {
    const scenarioKeys = breakdown.scenario_keys;
    const scenarioSection = document.getElementById("section-scenarios");
    if (scenarioKeys.length < 2) {
      scenarioSection.style.display = "none";
      return;
    }
    scenarioSection.style.display = "";
    document.getElementById("scenario-note").textContent =
      `Comparação entre ${scenarioKeys.join(" e ")}, média por componente.`;
    const legendEl = document.getElementById("scenario-legend");
    legendEl.innerHTML = "";
    scenarioKeys.forEach((s, i) => {
      const span = document.createElement("span");
      span.className = "key";
      span.innerHTML = `<span class="swatch" style="background:${SERIES_COLORS[i]}"></span>${s}`;
      legendEl.appendChild(span);
    });
    const scenariosEl = document.getElementById("chart-scenarios");
    scenariosEl.innerHTML = "";
    breakdown.scenario_comparison.forEach(c => {
      const row = document.createElement("div");
      row.className = "bar-row grouped";
      const subTracks = scenarioKeys.map((s, i) => {
        const val = c[s];
        const pct = (val / SCALE_MAX) * 100;
        return `<div class="sub-track"><div class="bar-fill" tabindex="0" style="width:${pct}%; background:${SERIES_COLORS[i]}"><span class="val tabular">${val.toFixed(2)}</span></div></div>`;
      }).join("");
      row.innerHTML = `<div class="row-label">${c.label}</div><div class="bar-track">${subTracks}</div>`;
      row.querySelectorAll(".bar-fill").forEach((fill, i) => {
        wireTooltip(fill, `<b>${c.label}</b><br>${scenarioKeys[i]}: ${c[scenarioKeys[i]].toFixed(2)}`);
      });
      scenariosEl.appendChild(row);
    });
  }

  function selectAiTab(key, label) {
    document.querySelectorAll(".ai-tab-btn").forEach(b => b.setAttribute("aria-selected", String(b.dataset.key === key)));
    document.getElementById("components-note").textContent = key === "__all__"
      ? "Média de todas as avaliações concluídas com sucesso, nos cenários combinados. Ordenado do melhor para o pior."
      : `Média das avaliações de ${label}, nos cenários combinados. Ordenado do melhor para o pior.`;
    renderComponentsChart(DATA.breakdowns[key]);
    renderScenariosChart(DATA.breakdowns[key]);
  }

  const aiTabsEl = document.getElementById("ai-tabs");
  const aiTabDefs = [{ key: "__all__", label: "Todos" }, ...DATA.judges.map(j => ({ key: j.id, label: j.id }))];
  aiTabDefs.forEach(({ key, label }) => {
    const btn = document.createElement("button");
    btn.className = "ai-tab-btn";
    btn.type = "button";
    btn.dataset.key = key;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", key === "__all__" ? "true" : "false");
    btn.textContent = label;
    btn.addEventListener("click", () => selectAiTab(key, label));
    aiTabsEl.appendChild(btn);
  });
  selectAiTab("__all__", "Todos");

  const tbody = document.getElementById("judge-rows");
  DATA.judges.forEach(j => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="judge-id">${j.id}<span class="provider">${j.provider}</span></td>
      <td class="num tabular">${j.n}</td>
      <td>
        <span class="pill ok">${j.ok} ok</span>
        ${j.error > 0 ? `<span class="pill error">${j.error} erro</span>` : ""}
      </td>
      <td class="num tabular">${j.mean !== null ? j.mean.toFixed(2) : "—"}${j.flag ? `<br><span class="n-flag">${j.flag}</span>` : ""}</td>`;
    tbody.appendChild(tr);
  });

  // ---- tools tab ----
  // No error branch here on purpose: build_tools_data() (Python) already drops every
  // status=error record before this data reaches the page, so a scenario is either a real
  // mean or simply absent -- "missing" covers both "not evaluated yet" and "only attempt
  // errored", which is the point (a transient API failure isn't a fact about the tool).
  function scoreCellHTML(scenarioData) {
    if (!scenarioData) return `<span class="score-cell missing">—</span>`;
    return `<span class="score-cell tabular">${scenarioData.mean.toFixed(2)}</span>`;
  }

  function renderToolDetail(t) {
    const allComponents = DATA.breakdowns.__all__.rubric_components;
    const blocks = scenarioKeys.map(s => {
      const sc = t.scenarios[s];
      if (!sc) return `<div class="detail-scenario"><h4>${s}</h4><div class="comp-row"><span>sem avaliação com sucesso</span></div></div>`;
      const judgeBlocks = sc.judges.map(j => {
        const compRows = Object.entries(j.components).map(([k, v]) => {
          const found = allComponents.find(c => c.key === k);
          const label = found ? found.label : k;
          return `<div class="comp-row"><span>${label}</span><b>${v}</b></div>`;
        }).join("");
        return `<div class="comp-row" style="font-weight:600"><span>${j.judge_id}</span><b>${j.mean.toFixed(2)}</b></div>${compRows}`;
      }).join("");
      return `<div class="detail-scenario"><h4>${s}</h4>${judgeBlocks}</div>`;
    }).join("");
    return `<div class="detail-grid">${blocks}</div>`;
  }

  // ---- filter dropdowns, populated once from the data actually present ----
  const languageSelect = document.getElementById("tools-filter-language");
  const judgeSelect = document.getElementById("tools-filter-judge");
  const coverageSelect = document.getElementById("tools-filter-coverage");

  [...new Set(DATA.tools.map(t => t.language))].sort().forEach(lang => {
    const opt = document.createElement("option");
    opt.value = lang;
    opt.textContent = lang;
    languageSelect.appendChild(opt);
  });
  DATA.judges.forEach(j => {
    const opt = document.createElement("option");
    opt.value = j.id;
    opt.textContent = j.id;
    judgeSelect.appendChild(opt);
  });

  let toolsSort = { field: "overall_mean", dir: "desc" };
  let toolsFilter = "";
  let activeLanguage = "";
  let activeJudge = "";
  let activeCoverage = "";

  function toolSortValue(t, field) {
    if (field === "name" || field === "repo") return (t[field] || "").toLowerCase();
    if (field === "overall_mean") return t.overall_mean === null ? -Infinity : t.overall_mean;
    const sc = t.scenarios[field];
    return sc && sc.mean !== null && sc.mean !== undefined ? sc.mean : -Infinity;
  }

  function matchesCoverage(t, coverage) {
    if (!coverage) return true;
    const has = s => !!t.scenarios[s];
    if (coverage === "both") return scenarioKeys.every(has);
    return has(coverage) && !scenarioKeys.filter(s => s !== coverage).some(has);
  }

  function renderToolsTable() {
    const tbody = document.getElementById("tools-rows");
    const emptyEl = document.getElementById("tools-empty");
    const countEl = document.getElementById("tools-count");
    tbody.innerHTML = "";

    const q = toolsFilter.trim().toLowerCase();
    let rows = DATA.tools.filter(t =>
      (!q || t.name.toLowerCase().includes(q) || t.repo.toLowerCase().includes(q) || t.qualified_name.toLowerCase().includes(q)) &&
      (!activeLanguage || t.language === activeLanguage) &&
      (!activeJudge || t.judges_present.includes(activeJudge)) &&
      matchesCoverage(t, activeCoverage)
    );

    const dirMul = toolsSort.dir === "asc" ? 1 : -1;
    rows = rows.slice().sort((a, b) => {
      const av = toolSortValue(a, toolsSort.field), bv = toolSortValue(b, toolsSort.field);
      if (av < bv) return -1 * dirMul;
      if (av > bv) return 1 * dirMul;
      return 0;
    });

    countEl.textContent = `${rows.length} de ${DATA.tools.length} tools`;
    emptyEl.hidden = rows.length > 0;

    rows.forEach(t => {
      const tr = document.createElement("tr");
      tr.className = "tool-row";
      tr.tabIndex = 0;
      tr.setAttribute("aria-expanded", "false");
      tr.innerHTML = `
        <td><span class="expand-icon">▸</span><span class="name">${t.name}</span><span class="qualified">${t.qualified_name}</span></td>
        <td>${t.repo}</td>
        <td>${t.language}</td>
        <td class="num">${scoreCellHTML(t.scenarios["description_only"])}</td>
        <td class="num">${scoreCellHTML(t.scenarios["with_source"])}</td>
        <td class="num score-cell tabular">${t.overall_mean !== null ? t.overall_mean.toFixed(2) : "—"}</td>`;

      const detailTr = document.createElement("tr");
      detailTr.className = "tool-detail";
      detailTr.hidden = true;
      const detailCell = document.createElement("td");
      detailCell.colSpan = 6;
      detailCell.innerHTML = renderToolDetail(t);
      detailTr.appendChild(detailCell);

      const toggle = () => {
        const expanded = tr.getAttribute("aria-expanded") === "true";
        tr.setAttribute("aria-expanded", String(!expanded));
        detailTr.hidden = expanded;
      };
      tr.addEventListener("click", toggle);
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      });

      tbody.appendChild(tr);
      tbody.appendChild(detailTr);
    });
  }

  document.getElementById("tools-search").addEventListener("input", (e) => {
    toolsFilter = e.target.value;
    renderToolsTable();
  });
  languageSelect.addEventListener("change", (e) => { activeLanguage = e.target.value; renderToolsTable(); });
  judgeSelect.addEventListener("change", (e) => { activeJudge = e.target.value; renderToolsTable(); });
  coverageSelect.addEventListener("change", (e) => { activeCoverage = e.target.value; renderToolsTable(); });

  document.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const field = th.dataset.sort;
      if (toolsSort.field === field) {
        toolsSort.dir = toolsSort.dir === "asc" ? "desc" : "asc";
      } else {
        toolsSort = { field, dir: (field === "name" || field === "repo") ? "asc" : "desc" };
      }
      document.querySelectorAll("th.sortable").forEach(h => h.removeAttribute("aria-sort"));
      th.setAttribute("aria-sort", toolsSort.dir === "asc" ? "ascending" : "descending");
      renderToolsTable();
    });
  });

  renderToolsTable();

  // ---- versions tab ----
  function renderVersionsTable() {
    const tbody = document.getElementById("version-rows");
    tbody.innerHTML = "";
    Object.values(DATA.prompt_versions).forEach(v => {
      const tr = document.createElement("tr");
      const judgesHTML = v.judges.map(j => `<span class="judges-inline">${j.id} <b>${j.n}</b></span>`).join(" · ");
      const purgeHTML = v.is_active
        ? `<span class="judges-inline">em uso -- nada a limpar</span>`
        : `<span class="judges-inline">rode para descartar:</span><code class="purge-cmd">uv run python scripts/purge_prompt_version.py --version ${v.version}</code>`;
      tr.innerHTML = `
        <td><span class="version-badge ${v.is_active ? "" : "stale"}">${v.version}${v.is_active ? " · ativa" : ""}</span></td>
        <td class="num tabular">${v.total}</td>
        <td>
          <span class="pill ok">${v.ok} ok</span>
          ${v.error > 0 ? `<span class="pill error">${v.error} erro</span>` : ""}
          ${v.refused > 0 ? `<span class="pill error">${v.refused} recusada</span>` : ""}
        </td>
        <td>${judgesHTML}</td>
        <td>${purgeHTML}</td>`;
      tbody.appendChild(tr);
    });
  }
  renderVersionsTable();
</script>
"""


def render_html(data: dict) -> str:
    return HTML_TEMPLATE.replace("__DASHBOARD_DATA__", json.dumps(data, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dashboard HTML dos resultados da Etapa 3.")
    parser.add_argument("--judge", type=str, default=None, help="judge_id único (default: todos em data/evaluations/).")
    parser.add_argument("--output", type=Path, default=None, help="Caminho de saída (default: data/analysis/dashboard.html).")
    parser.add_argument(
        "--prompt-version", type=str, default=None,
        help="Restringe Visão geral/Tools a este prompt_version (default: o mais recente presente nos dados -- ver aba Versões).",
    )
    args = parser.parse_args()

    eval_dir = DATA_DIR / "evaluations"
    records = load_records(eval_dir, args.judge)
    if not records:
        logger.error("nenhuma avaliação encontrada em %s", eval_dir)
        raise SystemExit(1)

    deduped = dedupe_records(records)
    if len(deduped) != len(records):
        logger.info("removidas %s avaliações duplicadas (retries via --retry-failed)", len(records) - len(deduped))
    records = deduped

    data = build_dashboard_data(records, prompt_version=args.prompt_version)
    html = render_html(data)

    output_path = args.output or (DATA_DIR / "analysis" / "dashboard.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    logger.info(
        "dashboard gerado em %s (prompt_version=%s: %s avaliações, %s tools, %s juízes)",
        output_path, data["meta"]["active_prompt_version"], data["overall"]["total"],
        data["meta"]["tools_evaluated"], len(data["judges"]),
    )


if __name__ == "__main__":
    main()
