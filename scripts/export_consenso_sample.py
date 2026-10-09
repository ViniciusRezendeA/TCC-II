#!/usr/bin/env python
"""Exporta as tools com consenso total de mudança entre juízes, uma linha por tool.

Aplica os mesmos 3 filtros de scripts/analysis_evaluation_report.py -- (1) tools avaliadas
pelos juízes, (2) avaliações cuja nota mudou entre description_only e with_source, (3) só os
casos em que TODOS os juízes que avaliaram aquela tool mudaram a nota no mesmo componente (ver
consenso_divergencia_por_tool(), que já encapsula os 3 passos) -- uma tool entra se atingiu
esse consenso em PELO MENOS um componente da rubrica.

Cada linha do JSONL de saída é uma tool (não mais uma tool x componente -- uma tool com
consenso em vários componentes aparece 1 vez só). `avaliacoes` traz, por juiz que avaliou os
dois cenários, só os componentes cujo score mudou entre description_only e with_source (ver
_trim_avaliacoes()): um juiz que não mudou nota em nenhum componente desta tool é omitido
inteiramente (não participou do consenso que trouxe a tool pra este arquivo). Cada entrada tem
só `{scenario, provider, scores}` -- nada do resto do registro bruto (schema_version,
prompt_version, repo, tool, judge completo, status, usage, latency_ms, evaluated_at: ver
AI_CONTEXT.md §10.2 pro schema bruto original, em data/evaluations/{judge_id}.jsonl).

O campo `source_code` traz o MESMO texto que foi enviado ao juiz no cenário with_source --
construído via evaluation/payload.py::build_payload() (que chama
schema/render_source_view.py), o caminho de código real da Etapa 3, não uma releitura
simplificada do arquivo. Requer data/dataset.jsonl (rode `uv run python -m
mcp_pipeline.schema.assemble_dataset` antes, se não existir -- é só concatenação dos
tools.jsonl por repo já extraídos, não refaz a Etapa 2) e os repositórios já clonados em
data/repos/. `source_code_sha256` recomputado é comparado contra o hash já gravado numa
avaliação with_source da própria tool (gravado pela Etapa 3 quando a avaliação rodou) -- uma
divergência (logada como aviso) indica que o conteúdo do repo ou a extração mudaram desde a
rodada original.

Lê data/evaluations/{judge_id}.jsonl, data/dataset.jsonl e os repositórios clonados em
data/repos/ -- não faz nenhuma chamada de API.

Uso:
  uv run python -m scripts.export_consenso_sample
  uv run python -m scripts.export_consenso_sample --output data/analysis/consenso_amostra.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.evaluation.payload import build_payload, repo_src_root_for
from mcp_pipeline.extraction.models import CallGraphNode, ToolRecord
from mcp_pipeline.logging_setup import setup_logging
from scripts.analysis_evaluation_report import (
    consenso_divergencia_por_tool,
    load_evaluations,
    registros_versao_ativa,
    scores_long,
    tool_key_for,
)
from scripts.dedupe_evaluations import dedupe_records

logger = setup_logging("export_consenso_sample")


def _avaliacoes_por_tool(records: list[dict]) -> dict[str, list[dict]]:
    """Para cada tool_uid (tool_key_for()), a lista dos registros BRUTOS (status="ok") dos dois
    cenários e de todos os juízes que avaliaram aquela tool -- os mesmos objetos de
    data/evaluations/{judge_id}.jsonl, sem extrair nem achatar nada (ver schema em
    AI_CONTEXT.md §10.2). Ordenado por (juiz, cenário) para leitura manual previsível: as duas
    linhas de um mesmo juiz ficam juntas.

    Uma tool com N juízes cobrindo os dois cenários tem exatamente 2*N registros aqui -- não
    filtra por componente porque uma única chamada ao juiz já pontua os 6 componentes de uma
    vez (ver evaluate() em evaluation/judges/base.py), então "os registros desta tool" é
    independente de qual componente disparou o consenso.
    """
    por_tool: dict[str, list[dict]] = {}
    for r in records:
        if r.get("status") != "ok" or not r.get("scores"):
            continue
        por_tool.setdefault(tool_key_for(r), []).append(r)
    for avaliacoes in por_tool.values():
        avaliacoes.sort(key=lambda r: (r["judge"]["id"], r["scenario"]))
    return por_tool


def _trim_avaliacoes(avaliacoes_brutas: list[dict]) -> list[dict]:
    """Reduz os registros brutos de uma tool (ver _avaliacoes_por_tool()) a, por juiz, só os
    componentes cujo score mudou entre description_only e with_source -- mesmo conjunto de
    chaves nas duas entradas (scenario, provider, scores) daquele juiz, já que "mudou" é uma
    propriedade do par, não de um lado isolado.

    Um juiz com só 1 dos 2 cenários (cobertura parcial) é omitido -- sem o par, não dá pra
    calcular "mudou". Um juiz cuja nota não mudou em NENHUM componente (não participou de
    nenhum consenso desta tool -- "espectador") também é omitido inteiramente: incluí-lo com
    `scores: {}` nos dois cenários não agregaria nada.

    `provider` vem de judge["provider"] (ex: "deepseek", "google", "ollama"), não de
    judge["id"] -- com os juízes atuais de config/judges.yaml cada provider mapeia pra
    exatamente 1 juiz, então não há ambiguidade hoje; se isso deixar de valer (dois juízes do
    mesmo provider), este campo por si só não distingue mais os dois.
    """
    por_juiz: dict[str, dict[str, dict]] = {}
    provider_por_juiz: dict[str, str] = {}
    for av in avaliacoes_brutas:
        jid = av["judge"]["id"]
        por_juiz.setdefault(jid, {})[av["scenario"]] = av
        provider_por_juiz[jid] = av["judge"]["provider"]

    trimmed: list[dict] = []
    for jid, por_cenario in por_juiz.items():
        desc = por_cenario.get("description_only")
        src = por_cenario.get("with_source")
        if desc is None or src is None:
            continue
        mudaram = [
            componente
            for componente, valor in desc["scores"].items()
            if valor and src["scores"].get(componente) and valor["score"] != src["scores"][componente]["score"]
        ]
        if not mudaram:
            continue
        provider = provider_por_juiz[jid]
        for cenario, record in (("description_only", desc), ("with_source", src)):
            trimmed.append({"scenario": cenario, "provider": provider, "scores": {c: record["scores"][c] for c in mudaram}})
    return trimmed


def consenso_detalhado(records: list[dict], avaliacoes_por_tool: dict[str, list[dict]]) -> list[dict]:
    """Uma entrada por tool que atingiu consenso total de mudança em PELO MENOS um componente
    (consenso_divergencia_por_tool() ainda decide quem qualifica, via os 3 filtros de
    analysis_evaluation_report.py -- só muda que uma tool com consenso em vários componentes
    agora gera 1 entrada, não 1 por componente). `avaliacoes` já vem reduzida por
    _trim_avaliacoes().
    """
    long_df = scores_long(records)
    consenso = consenso_divergencia_por_tool(long_df)
    tool_uids_qualificadas = set(consenso["tool_uid"])

    resultado: list[dict] = []
    for tool_uid in sorted(tool_uids_qualificadas):
        avaliacoes_brutas = avaliacoes_por_tool.get(tool_uid)
        if not avaliacoes_brutas:
            continue
        trimmed = _trim_avaliacoes(avaliacoes_brutas)
        if not trimmed:
            continue
        resultado.append(
            {
                "tool_uid": tool_uid,
                "tool_name": avaliacoes_brutas[0]["tool"]["name"],
                "avaliacoes": trimmed,
            }
        )
    return resultado


# Mesma lógica de pipeline/run_step3.py::tool_uid_for(), duplicada aqui (não importada) de
# propósito: scripts/ são consumidores read-only que nunca importam de evaluation/judges/*
# (ver AI_CONTEXT.md §8) -- importar run_step3.py pra reusar essa função de 5 linhas traria
# load_judges() e os SDKs de todo provedor (Gemini, DeepSeek, etc.) de carona, sem necessidade.
_LOWLEVEL_SHARED_LOCATION_PATTERNS = frozenset(
    {"python.list_tools_lowlevel", "typescript.set_request_handler_lowlevel", "javascript.set_request_handler_lowlevel"}
)


def _tool_uid_for(row: dict) -> str:
    loc = row["tool"]["source_location"]
    base = f"{row['repo']['name_with_owner']}::{row['tool']['qualified_name']}::{loc['file']}:{loc['start_line']}"
    if row["tool"].get("sdk_pattern") in _LOWLEVEL_SHARED_LOCATION_PATTERNS:
        return f"{base}::{row['tool']['name']}"
    return base


def _dataset_por_tool_uid(dataset_path: Path) -> dict[str, dict]:
    """Índice tool_uid -> linha de dataset.jsonl, pela mesma chave usada pelas avaliações já
    coletadas -- NÃO é o tool_uid_for() bruto (_tool_uid_for() acima, que só sufixa ::{name}
    nos padrões "lowlevel"): tool_key_for() (analysis_evaluation_report.py, já importada)
    sufixa ::{name} incondicionalmente, em TODO tool_uid, não só nos lowlevel (ver docstring
    de tool_key_for() -- "aplicado incondicionalmente"). Passar _tool_uid_for(row) direto como
    chave (sem passar por tool_key_for()) foi tentado primeiro e errou 531/600 lookups -- só os
    padrões lowlevel (onde as duas regras coincidem) bateram.
    """
    index: dict[str, dict] = {}
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_uid = _tool_uid_for(row)
            key = tool_key_for({"tool_uid": raw_uid, "tool": {"name": row["tool"]["name"]}})
            index[key] = row
    return index


def anexar_codigo_fonte(amostra: list[dict], dataset_path: Path, avaliacoes_por_tool: dict[str, list[dict]]) -> None:
    """Modifica `amostra` in-place, adicionando `source_code`/`source_code_sha256` a cada
    tool -- construído pelo MESMO caminho de código usado pela Etapa 3 pra montar o payload do
    cenário with_source (evaluation/payload.py::build_payload(), que por sua vez chama
    schema/render_source_view.py -- nada reimplementado aqui), incluindo o mesmo cap de
    MAX_SOURCE_CODE_CHARS. Uma tool sem linha correspondente em dataset.jsonl (repo
    reprocessado, ou abaixo de min_tools) ou cujo arquivo-fonte não existe mais em disco (ex:
    repo re-clonado/alterado desde a avaliação original) recebe `source_code: null` -- aviso
    logado, não erro, pra não abortar o resto da amostra (ver docstring do módulo).

    `avaliacoes_por_tool` (os registros BRUTOS, não os já reduzidos de `tool["avaliacoes"]` --
    ver _trim_avaliacoes()) é quem fornece o `source_code_sha256` original para a verificação
    de integridade abaixo: o formato reduzido não carrega mais esse campo. Uma divergência
    entre o hash recomputado e o gravado pela Etapa 3 quando a avaliação rodou (ver
    _base_record() em pipeline/run_step3.py) é avisada (conteúdo do repo ou a extração mudaram
    desde então), mas o código recomputado é anexado do mesmo jeito: é o melhor disponível
    agora, mesmo que não seja mais bit-a-bit idêntico ao que o juiz viu.
    """
    dataset_index = _dataset_por_tool_uid(dataset_path)
    n_ausentes = 0
    n_erro_leitura = 0
    n_hash_diferente = 0

    for tool in amostra:
        row = dataset_index.get(tool["tool_uid"])
        if row is None:
            tool["source_code"] = None
            tool["source_code_sha256"] = None
            n_ausentes += 1
            continue

        name_with_owner = row["repo"]["name_with_owner"]
        try:
            tool_record = ToolRecord.from_dict(row["tool"])
            call_graph = CallGraphNode.from_dict(row["call_graph"])
            payload = build_payload(
                tool_record, call_graph, repo_src_root_for(name_with_owner), name_with_owner, include_source=True,
            )
            source_code = payload.get("SOURCE_CODE") or ""
        except OSError as exc:
            logger.warning("Falha lendo código-fonte de %s: %s", tool["tool_uid"], exc)
            tool["source_code"] = None
            tool["source_code_sha256"] = None
            n_erro_leitura += 1
            continue

        source_code_sha256 = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
        tool["source_code"] = source_code
        tool["source_code_sha256"] = source_code_sha256

        hash_registrado = next(
            (
                av["source_code_sha256"]
                for av in avaliacoes_por_tool.get(tool["tool_uid"], [])
                if av["scenario"] == "with_source" and av.get("source_code_sha256")
            ),
            None,
        )
        if hash_registrado and hash_registrado != source_code_sha256:
            n_hash_diferente += 1

    if n_ausentes:
        logger.warning("%s tool(s) sem linha correspondente em dataset.jsonl -- source_code ficou null", n_ausentes)
    if n_erro_leitura:
        logger.warning("%s tool(s) com erro de leitura do código-fonte em disco -- source_code ficou null", n_erro_leitura)
    if n_hash_diferente:
        logger.warning(
            "%s tool(s) com source_code_sha256 recomputado diferente do registrado na avaliação original "
            "(repo ou extração podem ter mudado desde então)",
            n_hash_diferente,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta as tools com consenso total de mudança entre juízes, uma linha por tool, para validação manual."
    )
    parser.add_argument("--evaluations-dir", type=Path, default=None, help="Diretório com {judge_id}.jsonl (default: data/evaluations).")
    parser.add_argument("--dataset", type=Path, default=None, help="Caminho de dataset.jsonl, pra anexar o código-fonte (default: data/dataset.jsonl).")
    parser.add_argument("--output", type=Path, default=None, help="Caminho do JSONL de saída (default: data/analysis/consenso_amostra.jsonl).")
    args = parser.parse_args()

    evaluations_dir = args.evaluations_dir or (DATA_DIR / "evaluations")
    output_path = args.output or (DATA_DIR / "analysis" / "consenso_amostra.jsonl")

    if not evaluations_dir.exists() or not any(evaluations_dir.glob("*.jsonl")):
        logger.error(
            "Nenhum {judge_id}.jsonl encontrado em %s -- rode a Etapa 3 (pipeline/run_step3.py) primeiro.",
            evaluations_dir,
        )
        sys.exit(1)

    records = load_evaluations(evaluations_dir)
    logger.info("Carregadas %s avaliações de %s", len(records), evaluations_dir)

    deduped = dedupe_records(records)
    if len(deduped) != len(records):
        logger.info("Removidas %s avaliações duplicadas (retries via --retry-failed)", len(records) - len(deduped))
    records = deduped

    scoped = registros_versao_ativa(records)
    avaliacoes_por_tool = _avaliacoes_por_tool(scoped)

    amostra = consenso_detalhado(scoped, avaliacoes_por_tool)
    logger.info("%s tools com consenso total de mudança em pelo menos um componente", len(amostra))

    dataset_path = args.dataset or (DATA_DIR / "dataset.jsonl")
    if not dataset_path.exists():
        logger.error(
            "%s não encontrado -- rode `uv run python -m mcp_pipeline.schema.assemble_dataset` "
            "primeiro (só concatena os tools.jsonl por repo já extraídos, não refaz a Etapa 2).",
            dataset_path,
        )
        sys.exit(1)
    anexar_codigo_fonte(amostra, dataset_path, avaliacoes_por_tool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in amostra)

    logger.info("%s tools salvas em %s", len(amostra), output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Exportação da amostra de consenso falhou")
        sys.exit(1)
