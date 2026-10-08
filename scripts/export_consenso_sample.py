#!/usr/bin/env python
"""Exporta uma amostra de tools com consenso total de mudança entre juízes, por componente da
rubrica, para validação manual.

Aplica os mesmos 3 filtros de scripts/analysis_evaluation_report.py -- (1) tools avaliadas
pelos juízes, (2) avaliações cuja nota mudou entre description_only e with_source, (3) só os
casos em que TODOS os juízes que avaliaram aquela tool mudaram a nota no mesmo componente (ver
consenso_divergencia_por_tool(), que já encapsula os 3 passos) -- e, para cada componente,
sorteia uma amostra de N tools (default 100) dentre as que passaram pelos 3 filtros, com no
máximo --max-por-repo tools do mesmo repositório (default 6 -- ver MAX_POR_REPO_DEFAULT: a
população de consenso é extremamente concentrada por repositório, 2 repositórios somam mais de
50% dela, e um sorteio sem esse teto reproduz essa concentração na amostra). Mesma ordem de
grandeza da amostra de validação manual já usada no TCC (ver
overleaf/sectionsTCCII/05_Resultados_Parciais.tex, "100 pares") e o mesmo padrão de amostra
reprodutível via --seed de scripts/test_local_judges.py::load_sample_tools().

Cada linha do JSONL de saída é uma tool x componente selecionada; o campo `avaliacoes` traz os
registros BRUTOS dos juízes (mesmo schema de data/evaluations/{judge_id}.jsonl, ver
AI_CONTEXT.md §10.2: scores dos 6 componentes, judge, usage, latency_ms, etc. -- nada extraído
ou achatado), um por (juiz, cenário), para a tool inteira -- não só o componente que disparou o
consenso, já que cada chamada ao juiz pontua os 6 componentes de uma vez e o contexto completo
importa para a leitura manual. Uma tool com consenso em mais de um componente pode aparecer em
mais de uma linha (uma por componente em que foi sorteada), repetindo os mesmos `avaliacoes`.

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
  uv run python -m scripts.export_consenso_sample --n-por-componente 50 --seed 7
  uv run python -m scripts.export_consenso_sample --output data/analysis/consenso_amostra.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.evaluation.payload import build_payload, repo_src_root_for
from mcp_pipeline.evaluation.prompts import RUBRIC_COMPONENTS
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

# Mesma ordem de grandeza da amostra de validação manual já usada no TCC (ver docstring do
# módulo) -- default, sobrescrevível via --n-por-componente.
N_POR_COMPONENTE_DEFAULT = 100

# A população de tools com consenso é extremamente concentrada por repositório (medido em
# 2026-10-03: de 4473 pares tool x componente com consenso, codespar/mcp-dev-latam sozinho é
# 30.9% e google/mcp-security é 20.4% -- 51.3% só desses dois, de 157 repositórios distintos).
# random.sample() sem limite herda essa concentração (confirmado: um sorteio sem teto deixou
# esses 2 repos com 46.5% das 600 linhas) -- categorias/padrões extraídos manualmente de uma
# amostra assim tendem a refletir a convenção de documentação de 2 projetos, não do dataset.
# 6 é o menor teto por repositório que ainda garante >= N_POR_COMPONENTE_DEFAULT tools
# disponíveis no componente mais escasso (examples: 155 na população, 38 repositórios,
# 116 tools possíveis com teto 6 vs. só 104 com teto 5 -- margem baixa demais pra mudanças
# futuras no dataset).
MAX_POR_REPO_DEFAULT = 6


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


def consenso_detalhado_por_componente(records: list[dict]) -> dict[str, list[dict]]:
    """Uma lista por componente da rubrica (chave: componente), cada item uma tool com
    consenso total de mudança naquele componente (consenso_divergencia_por_tool() já aplica os
    passos 1 e 2 -- tools avaliadas pelos juízes e avaliações com mudança de nota entre
    cenários -- antes de exigir unanimidade), com os registros brutos dos juízes anexados (ver
    _avaliacoes_por_tool()) para leitura manual sem reabrir data/evaluations/*.jsonl.
    """
    long_df = scores_long(records)
    consenso = consenso_divergencia_por_tool(long_df)
    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    avaliacoes_por_tool = _avaliacoes_por_tool(records)

    por_componente: dict[str, list[dict]] = {key: [] for key, _, _ in RUBRIC_COMPONENTS}
    for (tool_uid, componente), grupo in consenso.groupby(["tool_uid", "componente"]):
        avaliacoes = avaliacoes_por_tool.get(tool_uid)
        if not avaliacoes:
            continue
        primeira_tool = avaliacoes[0]
        primeira_consenso = grupo.iloc[0]
        por_componente[componente].append(
            {
                "tool_uid": tool_uid,
                "tool_name": primeira_tool["tool"]["name"],
                "qualified_name": primeira_tool["tool"]["qualified_name"],
                "repo": primeira_tool["repo"]["name_with_owner"],
                "language": primeira_tool["repo"].get("primary_language") or "—",
                "componente": componente,
                "componente_label": labels.get(componente, componente),
                "n_juizes": int(primeira_consenso["n_juizes"]),
                "mesma_direcao": bool(primeira_consenso["mesma_direcao"]),
                "avaliacoes": avaliacoes,
            }
        )
    return por_componente


def amostrar_por_componente(por_componente: dict[str, list[dict]], n: int, max_por_repo: int) -> list[dict]:
    """Sorteia até `n` tools de cada componente, com no máximo `max_por_repo` tools do mesmo
    repositório (ver MAX_POR_REPO_DEFAULT: a população de consenso é extremamente concentrada
    -- 2 repositórios somam mais de 50% dela -- e random.sample() sem esse teto reproduz essa
    concentração na amostra, enviesando qualquer categorização manual feita em cima dela para a
    convenção de documentação de poucos projetos). Chame random.seed() antes, no caller, para
    reprodutibilidade (mesmo padrão de scripts/test_local_judges.py::load_sample_tools()).

    Implementação: embaralha o pool do componente (random.shuffle) e percorre uma única vez,
    pulando qualquer tool cujo repositório já atingiu o teto -- não corta o laço ao atingir `n`
    só porque um pool maior ainda pode ter repositórios abaixo do teto mais adiante; corta
    quando `n` é atingido OU o pool inteiro foi percorrido. Se mesmo assim sobrar menos que
    `n` (teto baixo demais para a diversidade de repositórios daquele componente), loga aviso
    em vez de erro e entrega o que deu.
    """
    amostra = []
    for componente, tools in por_componente.items():
        if not tools:
            continue
        pool = list(tools)
        random.shuffle(pool)

        por_repo: dict[str, int] = {}
        selecionadas = []
        for tool in pool:
            if len(selecionadas) >= n:
                break
            repo = tool["repo"]
            if por_repo.get(repo, 0) >= max_por_repo:
                continue
            selecionadas.append(tool)
            por_repo[repo] = por_repo.get(repo, 0) + 1

        if len(selecionadas) < n:
            logger.warning(
                "Componente %s: só %s/%s tools sorteadas (teto de %s por repositório esgotou "
                "a diversidade disponível; população tinha %s tools no total)",
                componente, len(selecionadas), n, max_por_repo, len(tools),
            )
        amostra.extend(selecionadas)
    return amostra


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


def anexar_codigo_fonte(amostra: list[dict], dataset_path: Path) -> None:
    """Modifica `amostra` in-place, adicionando `source_code`/`source_code_sha256` a cada
    tool -- construído pelo MESMO caminho de código usado pela Etapa 3 pra montar o payload do
    cenário with_source (evaluation/payload.py::build_payload(), que por sua vez chama
    schema/render_source_view.py -- nada reimplementado aqui), incluindo o mesmo cap de
    MAX_SOURCE_CODE_CHARS. Uma tool sem linha correspondente em dataset.jsonl (repo
    reprocessado, ou abaixo de min_tools) ou cujo arquivo-fonte não existe mais em disco (ex:
    repo re-clonado/alterado desde a avaliação original) recebe `source_code: null` -- aviso
    logado, não erro, pra não abortar o resto da amostra (ver docstring do módulo).

    Compara o hash recomputado contra o `source_code_sha256` já gravado numa avaliação
    with_source da própria tool (gravado pela Etapa 3 quando ela rodou, ver _base_record() em
    pipeline/run_step3.py) -- qualquer divergência é avisada (conteúdo do repo ou a extração
    mudaram desde então), mas o código recomputado é anexado do mesmo jeito: é o melhor
    disponível agora, mesmo que não seja mais bit-a-bit idêntico ao que o juiz viu.
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

        try:
            tool_record = ToolRecord.from_dict(row["tool"])
            call_graph = CallGraphNode.from_dict(row["call_graph"])
            payload = build_payload(
                tool_record, call_graph, repo_src_root_for(tool["repo"]), tool["repo"], include_source=True,
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
                for av in tool["avaliacoes"]
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
        description="Exporta uma amostra de tools com consenso total de mudança entre juízes, por componente, para validação manual."
    )
    parser.add_argument("--evaluations-dir", type=Path, default=None, help="Diretório com {judge_id}.jsonl (default: data/evaluations).")
    parser.add_argument("--dataset", type=Path, default=None, help="Caminho de dataset.jsonl, pra anexar o código-fonte (default: data/dataset.jsonl).")
    parser.add_argument("--output", type=Path, default=None, help="Caminho do JSONL de saída (default: data/analysis/consenso_amostra.jsonl).")
    parser.add_argument(
        "--n-por-componente", type=int, default=N_POR_COMPONENTE_DEFAULT,
        help=f"Quantas tools sortear por componente (default: {N_POR_COMPONENTE_DEFAULT}).",
    )
    parser.add_argument(
        "--max-por-repo", type=int, default=MAX_POR_REPO_DEFAULT,
        help=f"Máximo de tools do mesmo repositório por componente, pra evitar viés de poucos repositórios dominarem a amostra (default: {MAX_POR_REPO_DEFAULT}).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para reprodutibilidade da amostra (default: 42).")
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

    por_componente = consenso_detalhado_por_componente(scoped)
    for componente, tools in por_componente.items():
        logger.info("Componente %s: %s tools com consenso total de mudança", componente, len(tools))

    random.seed(args.seed)
    amostra = amostrar_por_componente(por_componente, args.n_por_componente, args.max_por_repo)

    dataset_path = args.dataset or (DATA_DIR / "dataset.jsonl")
    if not dataset_path.exists():
        logger.error(
            "%s não encontrado -- rode `uv run python -m mcp_pipeline.schema.assemble_dataset` "
            "primeiro (só concatena os tools.jsonl por repo já extraídos, não refaz a Etapa 2).",
            dataset_path,
        )
        sys.exit(1)
    anexar_codigo_fonte(amostra, dataset_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in amostra)

    logger.info("%s tools (de %s componentes) salvas em %s", len(amostra), len(por_componente), output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Exportação da amostra de consenso falhou")
        sys.exit(1)
