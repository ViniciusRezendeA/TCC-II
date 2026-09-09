#!/usr/bin/env python
"""Análise dos resultados dos juízes - Estilo Hassan et al.

Lê avaliações de `data/evaluations/{judge_id}.jsonl` e gera:
1. Estatísticas por componente da rubrica (6 componentes, escala 1-5 Likert)
2. Comparação entre cenários (description_only vs with_source)
3. Distribuição por linguagem, servidor, padrão SDK
4. Gráficos e tabelas em data/analysis/

Baseado na rubrica de 6 componentes de Hasan et al. (2026).

Uso:
  uv run python scripts/analyze_judges_results.py [--judge judge_id] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

from mcp_pipeline.config import DATA_DIR, ensure_dirs
from mcp_pipeline.evaluation.prompts import RUBRIC_COMPONENTS
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("analyze_judges")

# Nomes dos 6 componentes da rubrica
RUBRIC_KEYS = [key for key, _, _ in RUBRIC_COMPONENTS]
RUBRIC_LABELS = {key: label for key, label, _ in RUBRIC_COMPONENTS}


def load_evaluations(eval_dir: Path, judge_id: str | None = None) -> list[dict]:
    """Carrega todos os resultados de avaliação."""
    results = []

    if judge_id:
        # Carrega apenas um juiz específico
        eval_file = eval_dir / f"{judge_id}.jsonl"
        if not eval_file.exists():
            logger.warning(f"Arquivo não encontrado: {eval_file}")
            return []

        with open(eval_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    else:
        # Carrega todos os juízes
        for eval_file in sorted(eval_dir.glob("*.jsonl")):
            logger.info(f"Carregando {eval_file.name}...")
            with open(eval_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))

    logger.info(f"Total de avaliações carregadas: {len(results)}")
    return results


def extract_scores(evaluation: dict) -> dict[str, int | None]:
    """Extrai scores dos 6 componentes."""
    scores = {}
    if evaluation.get("scores"):
        for key in RUBRIC_KEYS:
            component = evaluation["scores"].get(key, {})
            if isinstance(component, dict):
                scores[key] = component.get("score")
            else:
                scores[key] = None
    return scores


def analyze_by_component(results: list[dict]) -> dict:
    """Análise estatística por componente da rubrica."""
    stats = {}

    for component in RUBRIC_KEYS:
        scores = []
        for result in results:
            if result.get("status") == "ok":
                result_scores = extract_scores(result)
                score = result_scores.get(component)
                if score is not None:
                    scores.append(score)

        if scores:
            stats[component] = {
                "count": len(scores),
                "mean": mean(scores),
                "stdev": stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
                "median": sorted(scores)[len(scores) // 2],
                "distribution": {
                    1: scores.count(1),
                    2: scores.count(2),
                    3: scores.count(3),
                    4: scores.count(4),
                    5: scores.count(5),
                },
            }

    return stats


def analyze_by_scenario(results: list[dict]) -> dict:
    """Análise comparativa entre cenários (description_only vs with_source)."""
    scenarios = defaultdict(list)

    for result in results:
        if result.get("status") == "ok":
            scenario = result.get("scenario", "unknown")
            scenarios[scenario].append(result)

    comparison = {}
    for scenario, scenario_results in scenarios.items():
        comparison[scenario] = analyze_by_component(scenario_results)

    return comparison


def analyze_by_language(results: list[dict]) -> dict:
    """Análise por linguagem de programação."""
    by_language = defaultdict(list)

    for result in results:
        if result.get("status") == "ok":
            language = result.get("repo", {}).get("primary_language", "Unknown")
            by_language[language].append(result)

    stats = {}
    for language, language_results in sorted(by_language.items()):
        stats[language] = {
            "count": len(language_results),
            "components": analyze_by_component(language_results),
        }

    return stats


def analyze_by_judge(results: list[dict]) -> dict:
    """Análise separada por juiz."""
    by_judge = defaultdict(list)

    for result in results:
        if result.get("status") == "ok":
            judge_id = result.get("judge", {}).get("id", "unknown")
            by_judge[judge_id].append(result)

    stats = {}
    for judge_id, judge_results in sorted(by_judge.items()):
        stats[judge_id] = {
            "count": len(judge_results),
            "components": analyze_by_component(judge_results),
        }

    return stats


def generate_summary_table(stats: dict) -> pd.DataFrame:
    """Gera tabela resumida com as 6 componentes."""
    rows = []

    for component in RUBRIC_KEYS:
        if component in stats:
            comp_stat = stats[component]
            rows.append({
                "Componente": RUBRIC_LABELS.get(component, component),
                "Média": f"{comp_stat['mean']:.2f}",
                "Desvio Padrão": f"{comp_stat['stdev']:.2f}",
                "Min": comp_stat["min"],
                "Max": comp_stat["max"],
                "N": comp_stat["count"],
            })

    return pd.DataFrame(rows)


def print_summary(results: list[dict]):
    """Imprime sumário executivo."""
    total = len(results)
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    refused_count = sum(1 for r in results if r.get("status") == "refused")
    error_count = sum(1 for r in results if r.get("status") == "error")

    print("\n" + "=" * 80)
    print("ANÁLISE DE RESULTADOS DOS JUÍZES - ESTILO HASSAN ET AL.")
    print("=" * 80)

    print(f"\n📊 RESUMO GERAL")
    print(f"  Total de avaliações: {total}")
    print(f"  ✅ OK: {ok_count} ({100*ok_count/total:.1f}%)")
    print(f"  ⚠️  Recusadas (segurança): {refused_count} ({100*refused_count/total:.1f}%)")
    print(f"  ❌ Erros: {error_count} ({100*error_count/total:.1f}%)")

    # Análise por componente
    component_stats = analyze_by_component(results)

    print(f"\n📈 ANÁLISE POR COMPONENTE DA RUBRICA")
    print(f"  (Escala Likert: 1=Ausente, 2=Vago, 3=Mínimo, 4=Menor ambiguidade, 5=Ideal)")
    print()

    df = generate_summary_table(component_stats)
    print(df.to_string(index=False))

    # Ranking dos componentes
    print(f"\n🏆 RANKING (por média)")
    ranked = sorted(component_stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    for i, (comp_key, comp_stat) in enumerate(ranked, 1):
        print(f"  {i}. {RUBRIC_LABELS.get(comp_key):25} {comp_stat['mean']:5.2f}")

    # Análise por cenário
    scenario_stats = analyze_by_scenario(results)
    if len(scenario_stats) > 1:
        print(f"\n🔄 COMPARAÇÃO: CENÁRIOS")
        for scenario, stats in scenario_stats.items():
            avg_score = mean([s["mean"] for s in stats.values()]) if stats else 0
            print(f"  {scenario:20} Média geral: {avg_score:.2f}")

    # Análise por linguagem
    lang_stats = analyze_by_language(results)
    if len(lang_stats) > 1:
        print(f"\n🗣️  ANÁLISE POR LINGUAGEM")
        for language, stats in sorted(lang_stats.items(), key=lambda x: -x[1]["count"])[:5]:
            avg_score = mean([s["mean"] for s in stats["components"].values()]) if stats["components"] else 0
            print(f"  {language:20} N={stats['count']:4}  Média: {avg_score:.2f}")

    # Análise por juiz (se houver múltiplos)
    judge_stats = analyze_by_judge(results)
    if len(judge_stats) > 1:
        print(f"\n⚖️  ANÁLISE POR JUIZ")
        for judge_id, stats in sorted(judge_stats.items(), key=lambda x: -x[1]["count"]):
            avg_score = mean([s["mean"] for s in stats["components"].values()]) if stats["components"] else 0
            print(f"  {judge_id:30} N={stats['count']:5}  Média: {avg_score:.2f}")

    print("\n" + "=" * 80)


def save_detailed_results(results: list[dict], output_dir: Path):
    """Salva resultados detalhados em CSV e JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Análise por componente
    component_stats = analyze_by_component(results)
    df = generate_summary_table(component_stats)
    df.to_csv(output_dir / "rubric_components_summary.csv", index=False)
    logger.info(f"✓ Salvou: {output_dir / 'rubric_components_summary.csv'}")

    # Análise por cenário
    scenario_stats = analyze_by_scenario(results)
    if len(scenario_stats) > 1:
        scenario_data = []
        for scenario, stats in scenario_stats.items():
            for component, comp_stat in stats.items():
                scenario_data.append({
                    "Scenario": scenario,
                    "Component": RUBRIC_LABELS.get(component, component),
                    "Mean": comp_stat["mean"],
                    "Stdev": comp_stat["stdev"],
                })
        df_scenario = pd.DataFrame(scenario_data)
        df_scenario.to_csv(output_dir / "scenario_comparison.csv", index=False)
        logger.info(f"✓ Salvou: {output_dir / 'scenario_comparison.csv'}")

    # Análise por linguagem
    lang_stats = analyze_by_language(results)
    lang_data = []
    for language, stats in lang_stats.items():
        for component, comp_stat in stats["components"].items():
            lang_data.append({
                "Language": language,
                "Tools": stats["count"],
                "Component": RUBRIC_LABELS.get(component, component),
                "Mean": comp_stat["mean"],
            })
    if lang_data:
        df_lang = pd.DataFrame(lang_data)
        df_lang.to_csv(output_dir / "by_language.csv", index=False)
        logger.info(f"✓ Salvou: {output_dir / 'by_language.csv'}")

    # Resultado em JSON para processamento futuro
    json_output = {
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r.get("status") == "ok"),
            "refused": sum(1 for r in results if r.get("status") == "refused"),
            "error": sum(1 for r in results if r.get("status") == "error"),
        },
        "rubric_components": component_stats,
        "by_scenario": {k: {k2: v2 for k2, v2 in v.items()} for k, v in scenario_stats.items()},
        "by_language": {k: {"count": v["count"]} for k, v in lang_stats.items()},
        "by_judge": {k: {"count": v["count"]} for k, v in analyze_by_judge(results).items()},
    }

    with open(output_dir / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, default=str)
    logger.info(f"✓ Salvou: {output_dir / 'analysis_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Analisa resultados dos juízes (style Hassan et al.)")
    parser.add_argument(
        "--judge",
        help="Judge ID específico para analisar (ex: qwen2.5-14b-instruct)",
    )
    parser.add_argument("--limit", type=int, help="Limitar número de resultados")
    args = parser.parse_args()

    ensure_dirs()
    eval_dir = DATA_DIR / "evaluations"

    if not eval_dir.exists():
        logger.error(f"Diretório de avaliações não encontrado: {eval_dir}")
        logger.error("Execute a Etapa 3 primeiro: uv run python -m mcp_pipeline.pipeline.run_step3")
        sys.exit(1)

    # Carrega avaliações
    results = load_evaluations(eval_dir, judge_id=args.judge)

    if args.limit:
        results = results[: args.limit]

    if not results:
        logger.error("Nenhuma avaliação encontrada")
        sys.exit(1)

    # Imprime sumário
    print_summary(results)

    # Salva resultados detalhados
    output_dir = DATA_DIR / "analysis" / "judges_results"
    save_detailed_results(results, output_dir)
    logger.info(f"\n✓ Análise completa salva em: {output_dir}")


if __name__ == "__main__":
    main()
