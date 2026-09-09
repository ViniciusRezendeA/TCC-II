#!/usr/bin/env python
"""Teste piloto dos juízes locais (Qwen + Llama).

Roda avaliação em ~20 tools aleatórios de dataset.jsonl contra os juízes locais,
registrando latências, erros e validação de estrutura das respostas.

Uso:
  uv run python scripts/test_local_judges.py [--sample-size N] [--judge qwen|llama|both]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from mcp_pipeline.config import DATA_DIR, ensure_dirs
from mcp_pipeline.evaluation.judges.base import JudgeRefusal
from mcp_pipeline.evaluation.judges.registry import load_judges
from mcp_pipeline.evaluation.payload import build_payload, repo_src_root_for
from mcp_pipeline.extraction.models import CallGraphNode, ToolRecord
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("test_local_judges")


def load_sample_tools(dataset_path: Path, sample_size: int = 20) -> list[dict]:
    """Carrega todos os tools do dataset e retorna uma amostra aleatória."""
    all_tools = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_tools.append(json.loads(line))

    if len(all_tools) < sample_size:
        logger.warning(f"Dataset tem apenas {len(all_tools)} tools, usando todos")
        return all_tools

    return random.sample(all_tools, sample_size)


def test_judge(judge, payload: dict, tool_info: str) -> dict:
    """Testa um juiz individual contra um payload."""
    result = {
        "tool": tool_info,
        "judge_id": judge.judge_id,
        "status": None,
        "error": None,
        "latency_ms": None,
        "tokens": None,
    }

    try:
        evaluation = judge.evaluate(payload)
        result["status"] = "ok"
        result["latency_ms"] = evaluation.latency_ms
        result["tokens"] = {
            "input": evaluation.input_tokens,
            "output": evaluation.output_tokens,
        }
        # Valida que scores foi populado
        if evaluation.scores:
            result["scores_validated"] = True
        logger.info(
            f"✓ {judge.judge_id:20} {tool_info:50} "
            f"latency={evaluation.latency_ms:.0f}ms tokens={evaluation.input_tokens}→{evaluation.output_tokens}"
        )
    except JudgeRefusal as e:
        result["status"] = "refused"
        result["error"] = f"Recusado: {e.category}"
        logger.warning(f"⚠ {judge.judge_id:20} {tool_info:50} RECUSADO: {e.category}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"✗ {judge.judge_id:20} {tool_info:50} ERRO: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Teste piloto dos juízes locais")
    parser.add_argument("--sample-size", type=int, default=20, help="Número de tools para testar")
    parser.add_argument(
        "--judge",
        choices=["qwen", "llama", "both"],
        default="both",
        help="Qual juiz testar",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade")
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dirs()

    dataset_path = DATA_DIR / "dataset.jsonl"
    if not dataset_path.exists():
        logger.error(f"Dataset não encontrado: {dataset_path}")
        sys.exit(1)

    logger.info(f"Carregando {args.sample_size} tools aleatórios de {dataset_path}...")
    tools = load_sample_tools(dataset_path, args.sample_size)
    logger.info(f"Carregado {len(tools)} tools")

    judge_filter = {args.judge} if args.judge != "both" else {"qwen", "llama"}
    judges = [j for j in load_judges() if j.provider in judge_filter or j.judge_id in judge_filter]

    if not judges:
        logger.error(f"Nenhum juiz encontrado para {judge_filter}")
        sys.exit(1)

    logger.info(f"Testando com juízes: {[j.judge_id for j in judges]}")
    logger.info("=" * 100)

    all_results = []
    for i, tool_row in enumerate(tools, start=1):
        logger.info(f"\n[{i}/{len(tools)}] Testando tool: {tool_row['tool']['qualified_name']}")

        try:
            tool = ToolRecord.from_dict(tool_row["tool"])
            call_graph = CallGraphNode.from_dict(tool_row["call_graph"])
            name_with_owner = tool_row["repo"]["name_with_owner"]
            payload = build_payload(
                tool,
                call_graph,
                repo_src_root_for(name_with_owner),
                name_with_owner,
                include_source=False,  # Testa apenas description_only no piloto
            )
        except Exception as e:
            logger.error(f"Erro ao construir payload: {e}")
            continue

        tool_info = f"{name_with_owner}::{tool_row['tool']['qualified_name']}"

        for judge in judges:
            result = test_judge(judge, payload, tool_info)
            all_results.append(result)

    logger.info("\n" + "=" * 100)
    logger.info("RESUMO DO TESTE PILOTO")
    logger.info("=" * 100)

    for judge in judges:
        results_for_judge = [r for r in all_results if r["judge_id"] == judge.judge_id]
        ok_count = sum(1 for r in results_for_judge if r["status"] == "ok")
        refused_count = sum(1 for r in results_for_judge if r["status"] == "refused")
        error_count = sum(1 for r in results_for_judge if r["status"] == "error")

        avg_latency = None
        if ok_count > 0:
            latencies = [r["latency_ms"] for r in results_for_judge if r["latency_ms"]]
            if latencies:
                avg_latency = sum(latencies) / len(latencies)

        logger.info(f"\n{judge.judge_id} ({judge.model_id}):")
        logger.info(f"  OK: {ok_count}/{len(results_for_judge)}")
        logger.info(f"  Recusado: {refused_count}/{len(results_for_judge)}")
        logger.info(f"  Erro: {error_count}/{len(results_for_judge)}")
        if avg_latency:
            logger.info(f"  Latência média: {avg_latency:.0f}ms")

        if error_count > 0:
            logger.warning(f"  Primeiros erros:")
            for r in results_for_judge[:3]:
                if r["status"] == "error":
                    logger.warning(f"    - {r['tool']}: {r['error']}")

    # Salva resultados
    results_file = DATA_DIR / "test_local_judges_results.jsonl"
    with open(results_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    logger.info(f"\nResultados salvos em: {results_file}")


if __name__ == "__main__":
    main()
