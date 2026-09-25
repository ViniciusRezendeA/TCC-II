#!/usr/bin/env python
"""Gera um dashboard HTML autocontido a partir de `data/evaluations/{judge_id}.jsonl`.

Recalcula tudo a cada execução (nenhum número fica hardcoded) -- pode rodar a qualquer
momento, inclusive com a Etapa 3 (run_step3.py) ainda em andamento, para ver um snapshot
atualizado dos resultados parciais. Mesmas 4 seções do protótipo original: tiles de status
geral, ranking por componente da rubrica, comparação de cenários e tabela por juiz.

O HTML gerado não depende de nada externo além de fontes do Google Fonts -- pode ser aberto
direto no navegador ou publicado como Artifact.

Uso (via -m: importa de scripts.analysis_evaluation_report/scripts.dedupe_evaluations, então
precisa da raiz do projeto no sys.path -- python scripts/generate_dashboard.py direto não
resolve isso):
  uv run python -m scripts.generate_dashboard
  uv run python -m scripts.generate_dashboard --output caminho/custom.html
  uv run python -m scripts.generate_dashboard --judge gemini-3.5-flash-lite
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from statistics import mean

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.evaluation.prompts import PROMPT_VERSION, RUBRIC_COMPONENTS
from mcp_pipeline.logging_setup import setup_logging
from scripts.analysis_evaluation_report import (
    MIN_JUIZES_CONSENSO,
    MOTIVO_FALLBACK,
    MUDANCA_MINIMA_QUARTIS,
    classificar_motivos,
    consenso_divergencia_por_tool,
    custo_real_por_juiz,
    migracao_quartil_por_tool,
    motivos_por_divergencia,
    resumo_consenso_por_componente,
    resumo_motivos_por_componente,
    scores_long,
    tool_key_for,
    veredito_custo_beneficio,
    wilcoxon_por_componente,
)
from scripts.analysis_report import (
    call_graph_resolucao,
    description_literal_rate,
    distribuicao_complexidade_ciclomatica,
    distribuicao_estrelas,
    distribuicao_loc,
    distribuicao_sdk_pattern,
    load_jsonl,
    media_tools_por_server,
    profundidade_call_graph,
    repos_por_linguagem,
    taxa_cobertura_por_linguagem,
    top_repos_por_tools,
    tools_por_linguagem,
)
from scripts.dedupe_evaluations import dedupe_records

logger = setup_logging("generate_dashboard")

# Below this many "ok" evaluations, a judge's mean is flagged as statistically meaningless
# instead of presented at face value -- avoids the dashboard implying a 2- or 3-sample mean
# is comparable to one backed by hundreds of evaluations.
MIN_TRUSTWORTHY_N = 30

# Rótulos de exibição para as chaves de MOTIVO_KEYWORDS/MOTIVO_FALLBACK (analysis_evaluation_
# report.py) -- mantidos aqui, não lá, porque são só apresentação (a chave em si é o que os
# CSVs/testes usam).
MOTIVO_LABELS: dict[str, str] = {
    "tool_sem_parametro": "Tool sem parâmetro",
    "omissao_parametros": "Omissão de parâmetros",
    "omissao_funcionalidades": "Omissão de funcionalidades",
    "omissao_saidas": "Omissão de saídas",
    "omissao_limitacoes": "Omissão de limitações",
    "contradicao": "Contradição",
    MOTIVO_FALLBACK: "Sem achado no código",
}


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


def build_dataset_data() -> dict | None:
    """Etapas 1-2 (coleta + extração) na mesma página interativa da Etapa 3 -- reaproveita as
    métricas de scripts/analysis_report.py (o companion PNG/CSV desta Etapa) em vez de
    recalculá-las de novo, mesmo princípio de reaproveitamento de
    scripts.analysis_evaluation_report usado no resto deste arquivo.

    Opcional por completo: sem selected_repos.jsonl/dataset.jsonl no disco (repo clonado só
    para rodar Etapa 3 a partir de jsonl de avaliação já prontos, sem os artefatos brutos da
    extração), a aba "Dataset" fica oculta -- não é um erro, mesmo tratamento de
    load_narrative_analysis() para o cache da análise Gemini ausente.
    """
    selected_repos_path = DATA_DIR / "selected_repos.jsonl"
    dataset_path = DATA_DIR / "dataset.jsonl"
    if not selected_repos_path.exists() or not dataset_path.exists():
        logger.info(
            "%s e/ou %s não encontrados -- dashboard sem aba Dataset (rode Etapas 1-2 para gerá-los).",
            selected_repos_path, dataset_path,
        )
        return None

    selected_repos = load_jsonl(selected_repos_path)
    dataset = load_jsonl(dataset_path)
    complexidade = distribuicao_complexidade_ciclomatica(dataset)

    return {
        "repos_total": len(selected_repos),
        "tools_total": len(dataset),
        "repos_por_linguagem": repos_por_linguagem(selected_repos).to_dict("records"),
        "tools_por_linguagem": tools_por_linguagem(dataset).to_dict("records"),
        "media_tools_por_server": media_tools_por_server(selected_repos, dataset).to_dict("records"),
        "distribuicao_sdk_pattern": distribuicao_sdk_pattern(dataset).to_dict("records"),
        "taxa_cobertura_por_linguagem": taxa_cobertura_por_linguagem(selected_repos, dataset).to_dict("records"),
        "call_graph_resolucao": call_graph_resolucao(dataset).to_dict("records"),
        "distribuicao_loc": distribuicao_loc(dataset).to_dict("records"),
        "distribuicao_complexidade_ciclomatica": complexidade.to_dict("records"),
        "profundidade_call_graph": profundidade_call_graph(dataset).to_dict("records"),
        "distribuicao_estrelas": distribuicao_estrelas(selected_repos).to_dict("records"),
        "description_literal_rate": description_literal_rate(dataset).to_dict("records"),
        "top_repos_por_tools": top_repos_por_tools(dataset, n=15).to_dict("records"),
    }


def compute_breakdown(ok_records: list[dict]) -> dict:
    """Rubric-component ranking, comparação de cenários e (só quando `ok_records` vem de um
    único juiz) teste de Wilcoxon por componente -- sobre uma fatia de registros ok,
    compartilhada pela visão combinada ("Todos") e por cada aba de juiz, para lerem de forma
    idêntica.

    Construído sobre scores_long()/wilcoxon_por_componente() de
    scripts/analysis_evaluation_report.py -- mesma identidade de tool (tool_key_for, que já
    corrige a colisão dos padrões de SDK "lowlevel") e mesmo teste de significância usados na
    planilha resumo_etapa_3.xlsx, reaproveitados aqui em vez de reimplementados.

    Wilcoxon não é computado na visão combinada ("Todos"): misturar juízes diferentes num só
    teste pareado não tem o significado que a metodologia do TCC pede (o teste é aplicado
    separadamente por modelo, ver Seção "Análise Comparativa dos Experimentos") -- só
    devolvido quando `ok_records` já está restrito a um único judge_id.
    """
    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    long_df = scores_long(ok_records)
    if long_df.empty:
        return {"rubric_components": [], "scenario_comparison": [], "scenario_keys": [], "wilcoxon": [], "boxplot": []}

    overall = long_df.groupby("componente")["nota"].agg(["mean", "std", "count"])
    ranked_keys = overall["mean"].sort_values(ascending=False).index.tolist()
    rubric_components = [
        {
            "key": k,
            "label": labels.get(k, k),
            "mean": float(overall.loc[k, "mean"]),
            "sd": float(overall.loc[k, "std"]) if overall.loc[k, "count"] > 1 and overall.loc[k, "std"] == overall.loc[k, "std"] else 0.0,
            "n": int(overall.loc[k, "count"]),
        }
        for k in ranked_keys
    ]

    scenarios = sorted(long_df["cenario"].unique().tolist())
    scenario_comparison: list[dict] = []
    wilcoxon: list[dict] = []
    if len(scenarios) >= 2:
        by_scenario_mean = long_df.groupby(["componente", "cenario"])["nota"].mean()
        scenario_comparison = [
            {
                "label": labels.get(k, k),
                **{s: (float(by_scenario_mean[(k, s)]) if (k, s) in by_scenario_mean.index else None) for s in scenarios},
            }
            for k in ranked_keys
        ]

        if long_df["juiz"].nunique() == 1:
            def _or_none(value):  # NaN (teste não aplicável, ex: componente todo empatado) -> null no JSON
                return None if value != value else float(value)

            wilcoxon = [
                {
                    "key": row["componente"],
                    "label": labels.get(row["componente"], row["componente"]),
                    "n_pareado": row["n_pareado"],
                    "proporcao_empates_percentual": _or_none(row["proporcao_empates_percentual"]),
                    "p_valor": _or_none(row["p_valor"]),
                    "p_valor_bh": _or_none(row["p_valor_bh"]),
                    "significativo": bool(row["significativo_bh_0.05"]),
                }
                for row in wilcoxon_por_componente(long_df).sort_values("componente").to_dict("records")
            ]

    return {
        "rubric_components": rubric_components,
        "scenario_comparison": scenario_comparison,
        "scenario_keys": scenarios,
        "wilcoxon": wilcoxon,
        # Mesma fatia de ok_records desta breakdown (combinada ou de um único juiz) alimentando
        # build_boxplot_data() -- dá à aba Divergências a mesma segmentação "Todos"/por juiz que
        # o resto da Visão geral já tem, sem duplicar a lógica de quartil/quantil em outro lugar.
        "boxplot": build_boxplot_data(ok_records),
    }


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
            "refused": sum(1 for r in judge_records if r.get("status") == "refused"),
            "mean": overall_mean,
            "flag": f"N={len(judge_ok)} — amostra insuficiente" if len(judge_ok) < MIN_TRUSTWORTHY_N else None,
        })

    tools_evaluated = len({tool_key_for(r) for r in ok_records})

    return {
        "meta": {
            "generated_at": datetime.datetime.now().astimezone().strftime("%d/%m/%Y, %H:%M (%Z)"),
            "prompt_version": PROMPT_VERSION,
            "active_prompt_version": active_version,
            "tools_evaluated": tools_evaluated,
            "divergence_method": f"Migração de pelo menos {MUDANCA_MINIMA_QUARTIS} faixas de quartil da nota entre cenários, por componente e juiz, após descarte de empates",
            "divergence_min_quartile_change": MUDANCA_MINIMA_QUARTIS,
            "motivo_labels": MOTIVO_LABELS,
            "motivo_fallback": MOTIVO_FALLBACK,
            "min_juizes_consenso": MIN_JUIZES_CONSENSO,
        },
        "overall": overall,
        "breakdowns": breakdowns,
        "judges": judges,
        "tools": build_tools_data(scoped),
        "divergences": build_divergences_data(scoped),
        "motivos_summary": build_motivos_summary_data(scoped),
        "consenso_juizes": build_consenso_data(scoped),
        "consenso_summary": build_consenso_summary_data(scoped),
        "tradeoff": build_tradeoff_data(scoped),
        "custo_real": build_custo_real_data(records),
        "prompt_versions": version_summary,
    }


def build_divergences_data(records: list[dict]) -> list[dict]:
    """Pares (tool, componente, juiz) cuja nota migrou pelo menos MUDANCA_MINIMA_QUARTIS
    faixas de quartil entre description_only e with_source (ver migracao_quartil_por_tool()
    em analysis_evaluation_report.py: quartil calculado por rank percentual dentro da
    distribuição de notas daquele componente/juiz/cenário, não pela nota bruta) -- os casos
    mais úteis para inspeção manual: ou o código revelou uma omissão/contradição real que a
    descrição escondia (o efeito que a rubrica pretende capturar), ou o juiz está reagindo à
    mera presença/tamanho do código em vez de validar a descrição contra ele (halo effect --
    ver changelog do PROMPT_VERSION em evaluation/prompts.py). O reasoning de cada cenário vem
    lado a lado para permitir essa leitura sem reabrir o jsonl bruto.

    Pares empatados (mesma nota, logo mesmo quartil, nos dois cenários) nunca aparecem aqui --
    não migraram por definição. Migrações de só 1 faixa (Q1->Q2) também ficam de fora: o
    critério exige um salto maior para valer inspeção manual.

    Chave por (tool_key_for(r), judge_id, componente) -- mesma identidade corrigida usada no
    resto do dashboard (ver build_tools_data()), não a tool_uid bruta do registro.
    """
    migracao = migracao_quartil_por_tool(scores_long(records))
    migracao_por_chave = {
        (row["juiz"], row["componente"], row["tool_uid"]): row
        for row in migracao.to_dict("records")
    }

    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    by_key: dict[tuple[str, str, str], dict] = {}
    for r in records:
        if r.get("status") != "ok" or not r.get("scores"):
            continue
        entry_key = (tool_key_for(r), r["judge"]["id"], "")
        for component_key, component in r["scores"].items():
            if not component:
                continue
            key = (entry_key[0], entry_key[1], component_key)
            entry = by_key.setdefault(key, {
                "tool_name": r["tool"]["name"],
                "qualified_name": r["tool"]["qualified_name"],
                "repo": r["repo"]["name_with_owner"],
                "language": r["repo"].get("primary_language") or "—",
                "judge_id": r["judge"]["id"],
                "componente": component_key,
                "componente_label": labels.get(component_key, component_key),
                "scenarios": {},
            })
            entry["scenarios"][r["scenario"]] = {
                "score": component["score"],
                "reasoning": component.get("reasoning") or "",
            }

    rows = []
    for (tool_key, judge_id, componente_key), entry in by_key.items():
        desc = entry["scenarios"].get("description_only")
        src = entry["scenarios"].get("with_source")
        if desc is None or src is None:
            continue
        migracao_tool = migracao_por_chave.get((judge_id, componente_key, tool_key))
        if migracao_tool is None or not migracao_tool["diverge"]:
            continue
        rows.append({
            "tool_name": entry["tool_name"],
            "qualified_name": entry["qualified_name"],
            "repo": entry["repo"],
            "language": entry["language"],
            "judge_id": entry["judge_id"],
            "componente": entry["componente"],
            "componente_label": entry["componente_label"],
            "description_only": desc,
            "with_source": src,
            "diff": src["score"] - desc["score"],
            "quartil_description_only": migracao_tool["quartil_description_only"],
            "quartil_with_source": migracao_tool["quartil_with_source"],
            "diff_quartil": migracao_tool["diff_quartil"],
            # Junção por palavra-chave (classificar_motivos()) sobre a reasoning do with_source
            # já buscada acima -- ver MOTIVO_KEYWORDS em analysis_evaluation_report.py.
            "motivos": classificar_motivos(src["reasoning"]),
        })

    rows.sort(key=lambda row: (abs(row["diff_quartil"]), abs(row["diff"])), reverse=True)
    return rows


def build_boxplot_data(records: list[dict]) -> list[dict]:
    """Sumário de cinco números (mínimo, Q1, mediana, Q3, máximo) por componente da rubrica x
    cenário, somando os dois juízes -- alimenta o boxplot da aba Divergências, que mostra a
    forma completa da distribuição de notas em description_only vs. with_source (diferente da
    tabela de build_divergences_data(), que já vem filtrada só nas migrações de tool
    individuais >= 2 faixas de quartil).

    Quantis calculados por pd.Series.quantile() (interpolação linear, o default do pandas),
    sobre a mesma base long (scores_long()) usada pelo resto do arquivo, para não introduzir
    um segundo critério de quantil dentro do mesmo dashboard (ver migracao_quartil_por_tool(),
    que usa rank percentual -- métodos diferentes por servirem perguntas diferentes: aqui é a
    distribuição agregada, lá é o quartil de uma tool individual dentro dela).
    """
    long_df = scores_long(records)
    if long_df.empty:
        return []

    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    order = [key for key, _, _ in RUBRIC_COMPONENTS]
    result = []
    for componente in order:
        notas_por_cenario = long_df[long_df["componente"] == componente].groupby("cenario")["nota"]
        if notas_por_cenario.ngroups == 0:
            continue
        scenarios = {
            cenario: {
                "min": float(notas.min()),
                "q1": float(notas.quantile(0.25)),
                "median": float(notas.quantile(0.5)),
                "q3": float(notas.quantile(0.75)),
                "max": float(notas.max()),
                "mean": float(notas.mean()),
                "n": int(notas.count()),
            }
            for cenario, notas in notas_por_cenario
        }
        result.append({
            "componente": componente,
            "componente_label": labels.get(componente, componente),
            "scenarios": scenarios,
        })
    return result


def build_motivos_summary_data(records: list[dict]) -> list[dict]:
    """Envelopa resumo_motivos_por_componente(motivos_por_divergencia()) em JSON -- alimenta a
    seção "Motivos de mudança" da aba Divergências: responde "quais são os motivos" agregando
    entre juízes (a granularidade por juiz completa, usada por veredito_custo_beneficio(), fica
    nos CSVs gerados por analysis_evaluation_report.py).
    """
    resumo = resumo_motivos_por_componente(motivos_por_divergencia(records))
    if resumo.empty:
        return []

    agregado = (
        resumo.groupby(["componente", "motivo"])
        .agg(ocorrencias=("ocorrencias", "sum"), n_subiu=("n_subiu", "sum"), n_desceu=("n_desceu", "sum"))
        .reset_index()
    )
    agregado["pct_subiu"] = (agregado["n_subiu"] / agregado["ocorrencias"] * 100).round(1)

    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    agregado["componente_label"] = agregado["componente"].map(lambda k: labels.get(k, k))
    agregado["motivo_label"] = agregado["motivo"].map(lambda k: MOTIVO_LABELS.get(k, k))
    return agregado.sort_values(["componente", "ocorrencias"], ascending=[True, False]).to_dict("records")


def build_tradeoff_data(records: list[dict]) -> list[dict]:
    """Envelopa veredito_custo_beneficio() em JSON -- alimenta a seção "Custo-benefício: vale a
    pena o código?" da aba Divergências. `records` não precisa vir pré-filtrado por
    registros_versao_ativa(): veredito_custo_beneficio() já filtra internamente (ver seu
    docstring), então passar `scoped` (já filtrado em build_dashboard_data()) é redundante mas
    inofensivo -- mantido por consistência com o resto das chamadas desta função.
    """
    df = veredito_custo_beneficio(records)
    if df.empty:
        return []

    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    df = df.copy()
    df["componente_label"] = df["componente"].map(lambda k: labels.get(k, k))
    # Todas essas colunas passam por pd.DataFrame(rows) dentro de veredito_custo_beneficio() e
    # de wilcoxon_por_componente() -- qualquer None de origem (ex: sem par suficiente para o
    # teste, ou juiz sem custo_latencia_por_juiz_e_cenario nos dois cenários) já virou NaN
    # nesse ponto, porque uma coluna float64 do pandas não distingue None de NaN. Sem essa
    # normalização, o NaN cru vazaria como o literal JS NaN no HTML gerado, em vez de null --
    # mesmo cuidado que compute_breakdown()::_or_none() já toma para wilcoxon_por_componente()
    # no resto do dashboard.
    campos_nullable = [
        "mediana_diferenca_efetiva", "pct_sem_motivo", "delta_input_tokens", "delta_latencia_ms",
        "custo_percentual_extra", "custo_extra_usd",
    ]
    rows = df.sort_values(["componente", "juiz"]).to_dict("records")
    for row in rows:
        for campo in campos_nullable:
            if row[campo] != row[campo]:
                row[campo] = None
    return rows


def build_consenso_data(records: list[dict]) -> list[dict]:
    """Envelopa consenso_divergencia_por_tool() em JSON -- uma linha por (tool, componente) em
    que TODOS os juízes que avaliaram aquela tool nos dois cenários (universo dinâmico, ver
    cobertura_juizes_por_tool() em analysis_evaluation_report.py; mínimo MIN_JUIZES_CONSENSO
    juízes) mudaram a nota daquele componente entre description_only e with_source. Diferente
    de consenso_divergencia_por_tool(), que tem uma linha por juiz, aqui os diffs de cada juiz
    já vêm agrupados em `juizes` -- alimenta a tabela "Tools com consenso total de mudança" da
    aba Divergências.

    Metadados (nome, repo, linguagem) vêm de um lookup por tool_key_for() sobre os records
    brutos -- mesma técnica de build_divergences_data(), porque consenso_divergencia_por_tool()
    só carrega tool_uid (a chave), não os metadados de exibição.
    """
    long_df = scores_long(records)
    consenso = consenso_divergencia_por_tool(long_df)
    if consenso.empty:
        return []

    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    metadata: dict[str, dict] = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        metadata.setdefault(
            tool_key_for(r),
            {
                "tool_name": r["tool"]["name"],
                "qualified_name": r["tool"]["qualified_name"],
                "repo": r["repo"]["name_with_owner"],
                "language": r["repo"].get("primary_language") or "—",
            },
        )

    rows = []
    for (tool_uid, componente), grupo in consenso.groupby(["tool_uid", "componente"]):
        meta = metadata.get(tool_uid)
        if meta is None:
            continue
        primeira = grupo.iloc[0]
        rows.append({
            **meta,
            "componente": componente,
            "componente_label": labels.get(componente, componente),
            "n_juizes": int(primeira["n_juizes"]),
            "mesma_direcao": bool(primeira["mesma_direcao"]),
            "juizes": [
                {"judge_id": row["juiz"], "diff": row["diff_nota"], "subiu": bool(row["subiu"])}
                for row in grupo.to_dict("records")
            ],
        })
    rows.sort(key=lambda row: (row["n_juizes"], sum(abs(j["diff"]) for j in row["juizes"])), reverse=True)
    return rows


def build_consenso_summary_data(records: list[dict]) -> list[dict]:
    """Envelopa resumo_consenso_por_componente() em JSON -- alimenta a tabela "Consenso entre
    juízes: componentes mais afetados" da aba Divergências: a resposta mais forte possível para
    RQ2 (quais componentes são mais afetados pela adição do código), porque só conta quando
    TODOS os juízes concordaram, não maioria nem "pelo menos um" (ver build_motivos_summary_data
    para a versão mais ampla, sem exigir unanimidade)."""
    resumo = resumo_consenso_por_componente(consenso_divergencia_por_tool(scores_long(records)))
    if resumo.empty:
        return []
    labels = {key: label for key, label, _ in RUBRIC_COMPONENTS}
    resumo = resumo.copy()
    resumo["componente_label"] = resumo["componente"].map(lambda k: labels.get(k, k))
    return resumo.to_dict("records")


def build_custo_real_data(records: list[dict]) -> list[dict]:
    """Envelopa custo_real_por_juiz() em JSON -- alimenta os tiles de "custo real" no topo da
    aba Divergências. Usa `records` sem passar por registros_versao_ativa(): dinheiro já gasto
    em avaliações de uma versão anterior do prompt continua tendo sido gasto de verdade (mesmo
    raciocínio do docstring de custo_real_por_juiz()), diferente das demais funções desta aba.
    """
    df = custo_real_por_juiz(records)
    if df.empty:
        return []
    return df.to_dict("records")


def build_tools_data(records: list[dict]) -> list[dict]:
    """One row per tool (keyed by tool_key_for(), not the raw tool_uid), with per-scenario
    scores averaged across whichever judge(s) evaluated it -- most tools right now only have
    one judge's worth of data, but this stays correct once a second judge's results land on
    the same tool/scenario.

    tool_key_for() (scripts/analysis_evaluation_report.py) instead of raw tool_uid: for the
    "lowlevel" SDK patterns, several distinct tools (different tool.name/description) share
    one tool_uid because they all inherit the location of the handler that registers them --
    grouping by the raw tool_uid here would silently merge those distinct tools into one row,
    displayed under whichever one's name happened to be inserted first.

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
        entry = by_tool.setdefault(tool_key_for(r), {
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


HTML_TEMPLATE = """<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rubrica MCP</title>
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
    --status-warning-soft: #faf0dd;
    --shadow: 0 1px 2px rgba(23,22,15,.06), 0 8px 24px -12px rgba(23,22,15,.16);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --surface-0: #17160f; --surface-1: #201f16; --surface-2: #2a2919; --line: #3a3826;
      --text-primary: #f4f2ea; --text-secondary: #c7c3ac; --text-muted: #8f8b71;
      --accent-1: #3987e5; --accent-2: #d95926;
      --status-good: #29c229; --status-good-soft: #163318;
      --status-critical: #e66767; --status-critical-soft: #3a1c1c; --status-warning: #d99a2b; --status-warning-soft: #3a2c12;
      --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --surface-0: #17160f; --surface-1: #201f16; --surface-2: #2a2919; --line: #3a3826;
    --text-primary: #f4f2ea; --text-secondary: #c7c3ac; --text-muted: #8f8b71;
    --accent-1: #3987e5; --accent-2: #d95926;
    --status-good: #29c229; --status-good-soft: #163318;
    --status-critical: #e66767; --status-critical-soft: #3a1c1c; --status-warning: #d99a2b; --status-warning-soft: #3a2c12;
    --shadow: 0 1px 2px rgba(0,0,0,.3), 0 8px 24px -12px rgba(0,0,0,.5);
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body { background: var(--surface-0); color: var(--text-primary); font-family: "IBM Plex Sans", system-ui, sans-serif; line-height: 1.5; overflow-x: hidden; }
  .wrap { max-width: 920px; margin: 0 auto; padding: 48px 24px 96px; }
  @media (max-width: 640px) { .wrap { padding: 32px 16px 64px; } }
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
  .narrative-summary { background: var(--surface-1); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; margin-bottom: 40px; font-size: 14.5px; color: var(--text-primary); line-height: 1.6; }
  .narrative-summary .eyebrow { display: block; margin-bottom: 6px; }
  .narrative-text { border-left: 2px solid var(--accent-1); padding: 2px 0 2px 14px; margin: 14px 0 0; font-size: 14px; color: var(--text-primary); line-height: 1.6; }
  .narrative-text:empty { display: none; }
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
  @media (max-width: 640px) {
    .bar-row { grid-template-columns: 96px 1fr; gap: 10px; }
    .bar-row .row-label { font-size: 12.5px; }
    .bar-track, .bar-row.grouped .bar-track { margin-right: 34px; }
    .bar-fill .val { font-size: 11.5px; }
  }
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
  .pill.warn { background: var(--status-warning-soft); color: var(--status-warning); }
  .motivo-badges { display: flex; flex-wrap: wrap; gap: 4px; }
  .n-flag { font-size: 11.5px; color: var(--status-warning); font-family: "IBM Plex Mono", monospace; }
  /* width:100% on `table` above is fine for narrow tables, but a wide one (many columns,
     long tool/repo names) would rather shrink every column to fit than trigger this
     wrapper's scroll -- min-width:max-content lets it grow past 100% when content actually
     needs the room, so overflow-x:auto here has something real to scroll instead of
     silently crushing/hiding the later columns on narrow screens. */
  .overflow-x { overflow-x: auto; }
  .overflow-x table { width: max-content; min-width: 100%; }
  .overflow-x th, .overflow-x td { white-space: nowrap; }
  .boxplot-row { display: grid; grid-template-columns: 168px 1fr; align-items: center; gap: 14px; margin-bottom: 16px; }
  .boxplot-row .row-label { font-size: 13.5px; color: var(--text-secondary); text-align: right; }
  .boxplot-tracks { display: flex; flex-direction: column; gap: 5px; }
  .boxplot-track { position: relative; height: 20px; }
  .boxplot-track .grid-tick { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--line); }
  .boxplot-track .grid-tick.major { background: var(--text-muted); opacity: .35; }
  .boxplot-whisker { position: absolute; top: 50%; height: 1px; background: currentColor; opacity: .55; }
  .boxplot-whisker-cap { position: absolute; top: 25%; bottom: 25%; width: 1.5px; background: currentColor; opacity: .65; }
  .boxplot-box { position: absolute; top: 2px; bottom: 2px; border-radius: 4px; background: currentColor; opacity: .25; border: 1.5px solid currentColor; }
  .boxplot-median { position: absolute; top: 0; bottom: 0; width: 2px; background: currentColor; }
  @media (max-width: 640px) { .boxplot-row { grid-template-columns: 100px 1fr; } }
  footer { border-top: 1px solid var(--line); padding-top: 24px; font-size: 13px; color: var(--text-muted); }
  footer p { max-width: 68ch; margin: 0 0 10px; }
  footer code { font-family: "IBM Plex Mono", monospace; background: var(--surface-1); padding: 1px 5px; border-radius: 4px; font-size: 12px; color: var(--text-secondary); }

  /* ---------- tabs ---------- */
  .tabs { display: flex; gap: 4px; margin-top: 28px; border-bottom: 1px solid var(--line); overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .tab-btn {
    font: inherit; font-family: "IBM Plex Sans", sans-serif; font-size: 14.5px; font-weight: 500;
    background: none; border: none; cursor: pointer; flex: 0 0 auto;
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
      <button class="tab-btn" role="tab" id="tab-btn-dataset" aria-controls="tab-dataset" aria-selected="false">Dataset</button>
      <button class="tab-btn" role="tab" id="tab-btn-tools" aria-controls="tab-tools" aria-selected="false">Tools</button>
      <button class="tab-btn" role="tab" id="tab-btn-divergences" aria-controls="tab-divergences" aria-selected="false">Divergências</button>
      <button class="tab-btn" role="tab" id="tab-btn-versions" aria-controls="tab-versions" aria-selected="false">Versões do prompt</button>
    </div>
  </header>

  <div id="tab-overview" role="tabpanel" aria-labelledby="tab-btn-overview">
    <div class="tiles">
      <div class="tile"><span class="label">Avaliações</span><span class="value tabular" id="tile-total">—</span><span class="sub" id="tile-total-sub">tool × cenário × juiz</span></div>
      <div class="tile ok"><span class="label">Sucesso</span><span class="value tabular" id="tile-ok">—</span><span class="sub" id="tile-ok-pct">—</span></div>
      <div class="tile error"><span class="label">Erro</span><span class="value tabular" id="tile-error">—</span><span class="sub" id="tile-error-pct">—</span></div>
      <div class="tile"><span class="label">Recusas</span><span class="value tabular" id="tile-refused">—</span><span class="sub">segurança</span></div>
    </div>

    <div class="narrative-summary" id="narrative-summary" hidden>
      <span class="eyebrow">Análise (Gemini) · gerada a partir dos mesmos números desta página, ver <code id="narrative-model">—</code></span>
      <span id="narrative-overall-text"></span>
    </div>

    <div class="ai-tabs" id="ai-tabs" role="tablist" aria-label="Filtrar rubrica por juiz"></div>

    <section>
      <h2>Por componente da rubrica</h2>
      <p class="section-note" id="components-note">Média de todas as avaliações concluídas com sucesso, nos cenários combinados. Ordenado do melhor para o pior.</p>
      <div class="chart" id="chart-components"></div>
      <p class="narrative-text" id="narrative-rubric-components"></p>
    </section>

    <section id="section-scenarios">
      <h2>Comparação de cenários</h2>
      <p class="section-note" id="scenario-note"></p>
      <div class="legend" id="scenario-legend"></div>
      <div class="chart" id="chart-scenarios"></div>
      <p class="narrative-text" id="narrative-scenario-comparison"></p>
    </section>

    <section id="section-wilcoxon">
      <h2>Significância (Teste de Wilcoxon)</h2>
      <p class="section-note">Diferença pareada with_source − description_only por tool, componente a componente, só para este juiz. p-valor corrigido por Benjamini-Hochberg entre os 6 componentes (não disponível na visão "Todos", que mistura juízes -- o teste é sempre por modelo).</p>
      <div class="overflow-x">
        <table>
          <thead><tr><th>Componente</th><th class="num">N pareado</th><th class="num">% empates</th><th class="num">p-valor (BH)</th><th>Significativo (α=0,05)</th></tr></thead>
          <tbody id="wilcoxon-rows"></tbody>
        </table>
      </div>
      <p class="narrative-text" id="narrative-wilcoxon-significance"></p>
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
      <p class="narrative-text" id="narrative-judges"></p>
    </section>
  </div>

  <div id="tab-dataset" role="tabpanel" aria-labelledby="tab-btn-dataset" hidden>
    <div class="tiles" style="grid-template-columns: repeat(2, 1fr); margin-bottom: 40px;">
      <div class="tile"><span class="label">Repositórios selecionados</span><span class="value tabular" id="dataset-tile-repos">—</span></div>
      <div class="tile"><span class="label">Tools extraídas</span><span class="value tabular" id="dataset-tile-tools">—</span></div>
    </div>

    <section>
      <h2>Repositórios por linguagem</h2>
      <div class="chart" id="chart-repos-linguagem"></div>
    </section>

    <section>
      <h2>Tools por linguagem</h2>
      <div class="chart" id="chart-tools-linguagem"></div>
    </section>

    <section id="section-media-tools">
      <h2>Média de tools por servidor, por linguagem</h2>
      <p class="section-note">Duas médias por propósito diferente: por repositório selecionado na Etapa 1 (inclui repos com 0 tools) vs. só entre os que confirmaram pelo menos 1 tool na Etapa 2.</p>
      <div class="legend" id="media-tools-legend"></div>
      <div class="chart" id="chart-media-tools"></div>
    </section>

    <section>
      <h2>Cobertura da Etapa 2 por linguagem</h2>
      <p class="section-note">% dos repositórios selecionados (Etapa 1), por linguagem, que confirmaram pelo menos 1 tool detectável (Etapa 2).</p>
      <div class="chart" id="chart-cobertura-linguagem"></div>
    </section>

    <section>
      <h2>Padrão de SDK</h2>
      <p class="section-note">Como cada tool declara sua ferramenta ao SDK do protocolo MCP (decorator de alto nível vs. handler de baixo nível, por linguagem).</p>
      <div class="chart" id="chart-sdk-pattern"></div>
    </section>

    <section>
      <h2>Profundidade do call graph</h2>
      <p class="section-note">Nível máximo alcançado na árvore de chamadas de cada tool, limitada a 3 por construção.</p>
      <div class="chart" id="chart-profundidade"></div>
    </section>

    <section>
      <h2>Resolução do call graph</h2>
      <p class="section-note">Dos nós de nível 2-3 da árvore de chamadas: quantos foram resolvidos sem ambiguidade, resolvidos por desempate, ou permaneceram externos/dinâmicos.</p>
      <div class="chart" id="chart-call-graph-resolucao"></div>
    </section>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 32px;">
      <section style="margin-bottom: 0;">
        <h2>Distribuição de estrelas</h2>
        <div class="overflow-x">
          <table>
            <thead><tr><th>Estatística</th><th class="num">Estrelas</th></tr></thead>
            <tbody id="stats-estrelas-rows"></tbody>
          </table>
        </div>
      </section>

      <section style="margin-bottom: 0;">
        <h2>LOC por tool</h2>
        <p class="section-note">Linhas de código da função que implementa a tool (nível 1 do call graph).</p>
        <div class="overflow-x">
          <table>
            <thead><tr><th>Estatística</th><th class="num">LOC</th></tr></thead>
            <tbody id="stats-loc-rows"></tbody>
          </table>
        </div>
      </section>

      <section id="section-complexidade" style="margin-bottom: 0;" hidden>
        <h2>Complexidade ciclomática</h2>
        <p class="section-note">McCabe, da mesma função. Só disponível para tools extraídas depois que este campo passou a existir.</p>
        <div class="overflow-x">
          <table>
            <thead><tr><th>Estatística</th><th class="num">Complexidade</th></tr></thead>
            <tbody id="stats-complexidade-rows"></tbody>
          </table>
        </div>
      </section>
    </div>

    <section>
      <h2>Taxa de descrição literal, por linguagem</h2>
      <p class="section-note">% de descrições que são literais de string simples (vs. resolvidas a partir de constante/variável) -- afeta a confiabilidade da extração da descrição em si.</p>
      <div class="overflow-x">
        <table>
          <thead><tr><th>Linguagem</th><th class="num">Descrições literais</th><th class="num">Tools</th><th class="num">% literal</th></tr></thead>
          <tbody id="literal-rate-rows"></tbody>
        </table>
      </div>
    </section>

    <section style="margin-bottom: 0;">
      <h2>Top repositórios por número de tools</h2>
      <div class="overflow-x">
        <table>
          <thead><tr><th>Repositório</th><th class="num">Tools</th><th>Linguagem</th><th class="num">Estrelas</th><th class="num">% do total</th></tr></thead>
          <tbody id="top-repos-rows"></tbody>
        </table>
      </div>
    </section>
  </div>

  <div id="tab-tools" role="tabpanel" aria-labelledby="tab-btn-tools" hidden>
    <section style="margin-bottom: 0;">
      <h2>Todas as tools</h2>
      <p class="section-note">Uma linha por tool avaliada com sucesso, com a média por cenário e, se houver mais de um juiz na mesma tool/cenário, a média entre eles. Avaliações com erro técnico não aparecem aqui -- contam só no resumo geral. Clique numa linha para ver o detalhe por componente.</p>
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

  <div id="tab-divergences" role="tabpanel" aria-labelledby="tab-btn-divergences" hidden>
    <div class="ai-tabs" id="divergences-ai-tabs" role="tablist" aria-label="Filtrar divergências por juiz"></div>

    <section>
      <h2>Custo real gasto</h2>
      <p class="section-note">Preço público do provedor de cada juiz no momento de cada chamada (ver <code>DEEPSEEK_FLASH_PRICING_USD_POR_1M</code> em <code>analysis_evaluation_report.py</code>), somando os dois cenários e todas as versões de prompt já rodadas -- dinheiro já gasto não depende de qual versão do prompt está ativa hoje. Juízes locais (llama.cpp) ou no free tier do Google não têm custo direto.</p>
      <div class="tiles" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 0;" id="cost-tiles"></div>
    </section>

    <section>
      <h2>Custo-benefício: vale a pena o código?</h2>
      <p class="section-note">Por juiz e componente da rubrica: direção e significância do efeito de mandar <code>with_source</code> (Wilcoxon), % das divergências sem justificativa específica na reasoning e o custo extra (tokens e USD reais) de mandar o código-fonte. A direção ("sobe"/"desce") é só descritiva -- o texto ativo do prompt não restringe para qual lado a nota pode mudar, só diz que o ajuste deve vir de uma inconsistência identificada. "Vale a pena" quando há um efeito estatisticamente significativo (em qualquer direção) E menos da metade das divergências ficam sem justificativa específica -- ver docstring de <code>veredito_custo_beneficio()</code> para a regra completa.</p>
      <div class="overflow-x">
        <table>
          <thead>
            <tr>
              <th>Juiz</th>
              <th>Componente</th>
              <th>Direção</th>
              <th>Significativo</th>
              <th class="num">% sem motivo</th>
              <th class="num">Custo extra (tokens)</th>
              <th class="num">Custo extra (%)</th>
              <th class="num">Custo extra (US$/1000 aval.)</th>
              <th>Vale a pena?</th>
            </tr>
          </thead>
          <tbody id="tradeoff-rows"></tbody>
        </table>
      </div>
      <div class="empty-state" id="tradeoff-empty" hidden>Sem dados suficientes para um veredito.</div>
    </section>

    <section>
      <h2>Motivos de mudança</h2>
      <p class="section-note">Junção por palavra-chave (ver <code>MOTIVO_KEYWORDS</code> em <code>analysis_evaluation_report.py</code>) sobre a justificativa do <code>with_source</code> de cada divergência (tabela "Maiores divergências" abaixo): quantas vezes cada motivo aparece, e se acompanhou subida ou descida de quartil. "Sem justificativa específica" marca divergências cuja reasoning não nomeia nenhum problema concreto encontrado no código, independente da nota ter subido ou descido.</p>
      <div class="overflow-x">
        <table>
          <thead>
            <tr>
              <th>Componente</th>
              <th>Motivo</th>
              <th class="num">Ocorrências</th>
              <th class="num">Subiu</th>
              <th class="num">Desceu</th>
            </tr>
          </thead>
          <tbody id="motivos-summary-rows"></tbody>
        </table>
      </div>
      <div class="empty-state" id="motivos-summary-empty" hidden>Nenhuma divergência para classificar.</div>
    </section>

    <section>
      <h2>Consenso entre juízes: componentes mais afetados</h2>
      <p class="section-note">Critério mais forte de mudança de nota: só conta quando <b>todos</b> os juízes que avaliaram aquela tool nos dois cenários mudaram a nota do mesmo componente (não maioria, não "pelo menos um") -- exige pelo menos <b id="consenso-min-juizes" class="tabular"></b> juízes cobrindo a tool (ver <code>MIN_JUIZES_CONSENSO</code> em <code>analysis_evaluation_report.py</code>). "Mesma direção" conta só os casos em que, além de todos mudarem, todos mudaram para o mesmo lado (todos subiram ou todos desceram).</p>
      <div class="overflow-x">
        <table>
          <thead>
            <tr>
              <th>Componente</th>
              <th class="num">Tools com consenso</th>
              <th class="num">Mesma direção</th>
              <th class="num">Média de juízes</th>
            </tr>
          </thead>
          <tbody id="consenso-summary-rows"></tbody>
        </table>
      </div>
      <div class="empty-state" id="consenso-summary-empty" hidden>Nenhum consenso de mudança entre juízes encontrado.</div>
    </section>

    <section>
      <h2>Tools com consenso total de mudança</h2>
      <p class="section-note">Uma linha por tool × componente em que todo o universo de juízes daquela tool mudou a nota entre <code>description_only</code> e <code>with_source</code>. O diff de cada juiz aparece entre parênteses; "Direção" indica se todos mudaram para o mesmo lado.</p>
      <div class="tools-toolbar">
        <input type="text" id="consenso-search" class="tools-search" placeholder="Buscar por nome ou repositório…" autocomplete="off">
        <select id="consenso-filter-componente" class="tools-filter"><option value="">Todo componente</option></select>
        <select id="consenso-filter-direction" class="tools-filter">
          <option value="">Toda direção</option>
          <option value="mesma">Mesma direção</option>
          <option value="mista">Direção mista</option>
        </select>
        <span class="tools-count" id="consenso-count"></span>
      </div>
      <div class="overflow-x">
        <table>
          <thead>
            <tr>
              <th>Componente</th>
              <th>Tool</th>
              <th>Repositório</th>
              <th class="num">Nº juízes</th>
              <th>Diffs por juiz</th>
              <th>Direção</th>
            </tr>
          </thead>
          <tbody id="consenso-rows"></tbody>
        </table>
      </div>
      <div class="empty-state" id="consenso-empty" hidden>Nenhuma tool com consenso total de mudança encontrada.</div>
    </section>

    <section>
      <h2>Distribuição das notas por cenário</h2>
      <p class="section-note" id="boxplot-note">Para cada componente da rubrica, boxplot (mínimo, Q1, mediana, Q3, máximo) da nota em <code>description_only</code> contra <code>with_source</code>, somando os dois juízes -- a forma completa da distribuição em cada cenário, não só a média.</p>
      <div class="legend" id="boxplot-legend"></div>
      <div class="chart" id="chart-divergences-boxplot"></div>
    </section>

    <section style="margin-bottom: 0;">
      <h2>Maiores divergências entre cenários</h2>
      <p class="section-note">Pares tool × componente × juiz cuja nota migrou pelo menos <b id="divergence-min-change" class="tabular"></b> faixas de quartil entre <code>description_only</code> e <code>with_source</code> (ex: Q1 → Q3), após descartar empates -- candidatos a inspeção manual: ou o código revelou algo que a descrição escondia, ou o juiz reagiu à presença do código em vez de validar a descrição contra ele. Clique numa linha para ver a justificativa dos dois cenários lado a lado.</p>
      <p class="narrative-text" id="narrative-divergences"></p>
      <div class="tools-toolbar">
        <input type="text" id="divergences-search" class="tools-search" placeholder="Buscar por nome ou repositório…" autocomplete="off">
        <select id="divergences-filter-language" class="tools-filter"><option value="">Toda linguagem</option></select>
        <select id="divergences-filter-componente" class="tools-filter"><option value="">Todo componente</option></select>
        <select id="divergences-filter-direction" class="tools-filter">
          <option value="">Toda direção</option>
          <option value="subiu">Nota subiu</option>
          <option value="desceu">Nota desceu</option>
        </select>
        <span class="tools-count" id="divergences-count"></span>
      </div>
      <div class="overflow-x">
        <table>
          <thead>
            <tr>
              <th>Componente</th>
              <th>Tool</th>
              <th>Repositório</th>
              <th>Juiz</th>
              <th class="num">description_only</th>
              <th class="num">with_source</th>
              <th class="num">Diferença</th>
              <th class="num">Quartil</th>
              <th>Motivo(s)</th>
            </tr>
          </thead>
          <tbody id="divergences-rows"></tbody>
        </table>
      </div>
      <div class="empty-state" id="divergences-empty" hidden>Nenhuma tool migrou quartil o suficiente para aparecer aqui.</div>
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

  // ---- dataset tab (Etapas 1-2) -- rendered once at load, not tab-segmented by juiz (this
  // data predates Etapa 3 entirely). Generic bar helper, unlike renderComponentsChart's
  // (fixed 1-5 Likert scale with reference gridlines): these are arbitrary-scale counts,
  // percentuais and category labels, so it just scales to each chart's own max.
  function renderBarRows(containerId, rows, labelKey, valueKey, opts) {
    opts = opts || {};
    const el = document.getElementById(containerId);
    el.innerHTML = "";
    const maxValue = opts.max || Math.max(1, ...rows.map(r => r[valueKey]));
    const fmt = opts.format || (v => v);
    rows.forEach(r => {
      const row = document.createElement("div");
      row.className = "bar-row";
      const pct = (r[valueKey] / maxValue) * 100;
      row.innerHTML = `
        <div class="row-label">${escapeHtml(String(r[labelKey]))}</div>
        <div class="bar-track">
          <div class="bar-fill" tabindex="0" style="width:${pct}%">
            <span class="val tabular">${fmt(r[valueKey])}</span>
          </div>
        </div>`;
      el.appendChild(row);
    });
  }

  function renderStatsTable(bodyId, rows, valueKey) {
    const tbody = document.getElementById(bodyId);
    tbody.innerHTML = "";
    rows.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHtml(r.estatistica)}</td><td class="num tabular">${r[valueKey]}</td>`;
      tbody.appendChild(tr);
    });
  }

  function renderDatasetTab() {
    const d = DATA.dataset;
    if (!d) {
      document.getElementById("tab-btn-dataset").hidden = true;
      return;
    }
    document.getElementById("dataset-tile-repos").textContent = d.repos_total;
    document.getElementById("dataset-tile-tools").textContent = d.tools_total;

    renderBarRows("chart-repos-linguagem", d.repos_por_linguagem, "linguagem", "repositorios");
    renderBarRows("chart-tools-linguagem", d.tools_por_linguagem, "linguagem", "tools");

    const mediaTools = d.media_tools_por_server.filter(r => r.linguagem !== "Total");
    const mediaToolsKeys = ["media_tools_por_repo_selecionado", "media_tools_por_repo_com_tools"];
    const mediaToolsLabels = ["Por repo selecionado (Etapa 1)", "Por repo com ≥1 tool"];
    document.getElementById("media-tools-legend").innerHTML = mediaToolsLabels
      .map((label, i) => `<span class="key"><span class="swatch" style="background:${SERIES_COLORS[i]}"></span>${label}</span>`)
      .join("");
    const mediaToolsMax = Math.max(1, ...mediaTools.flatMap(r => mediaToolsKeys.map(k => r[k])));
    const mediaToolsEl = document.getElementById("chart-media-tools");
    mediaToolsEl.innerHTML = "";
    mediaTools.forEach(r => {
      const row = document.createElement("div");
      row.className = "bar-row grouped";
      const subTracks = mediaToolsKeys.map((k, i) =>
        `<div class="sub-track"><div class="bar-fill" tabindex="0" style="width:${(r[k] / mediaToolsMax) * 100}%; background:${SERIES_COLORS[i]}"><span class="val tabular">${r[k].toFixed(2)}</span></div></div>`
      ).join("");
      row.innerHTML = `<div class="row-label">${escapeHtml(r.linguagem)}</div><div class="bar-track">${subTracks}</div>`;
      mediaToolsEl.appendChild(row);
    });

    renderBarRows("chart-cobertura-linguagem", d.taxa_cobertura_por_linguagem, "linguagem", "taxa_cobertura_percentual", { max: 100, format: v => v.toFixed(1) + "%" });
    renderBarRows("chart-sdk-pattern", d.distribuicao_sdk_pattern, "sdk_pattern", "tools");
    renderBarRows("chart-profundidade", d.profundidade_call_graph, "profundidade", "tools");
    renderBarRows("chart-call-graph-resolucao", d.call_graph_resolucao, "categoria", "percentual", { max: 100, format: v => v.toFixed(1) + "%" });

    renderStatsTable("stats-estrelas-rows", d.distribuicao_estrelas, "estrelas");
    renderStatsTable("stats-loc-rows", d.distribuicao_loc, "loc");

    if (d.distribuicao_complexidade_ciclomatica.length > 0) {
      document.getElementById("section-complexidade").hidden = false;
      renderStatsTable("stats-complexidade-rows", d.distribuicao_complexidade_ciclomatica, "complexidade_ciclomatica");
    }

    const literalTbody = document.getElementById("literal-rate-rows");
    literalTbody.innerHTML = "";
    d.description_literal_rate.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHtml(r.linguagem)}</td><td class="num tabular">${r.descricoes_literais}</td><td class="num tabular">${r.tools_total}</td><td class="num tabular">${r.taxa_literal_percentual.toFixed(1)}%</td>`;
      literalTbody.appendChild(tr);
    });

    const topReposTbody = document.getElementById("top-repos-rows");
    topReposTbody.innerHTML = "";
    d.top_repos_por_tools.forEach(r => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${escapeHtml(r.repositorio)}</td><td class="num tabular">${r.tools}</td><td>${escapeHtml(r.linguagem)}</td><td class="num tabular">${r.estrelas}</td><td class="num tabular">${r.percentual_do_total.toFixed(1)}%</td>`;
      topReposTbody.appendChild(tr);
    });
  }
  renderDatasetTab();

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

  // ---- top tiles: re-rendered per selected AI (judge) tab by selectAiTab() below, so
  // "Avaliações"/"Sucesso"/"Erro"/"Recusas" reflect whichever judge is currently in view
  // instead of always the combined total. ----
  function renderTiles(key) {
    const judge = key === "__all__" ? null : DATA.judges.find(j => j.id === key);
    const stats = judge ? { total: judge.n, ok: judge.ok, error: judge.error, refused: judge.refused } : DATA.overall;
    document.getElementById("tile-total").textContent = stats.total;
    document.getElementById("tile-total-sub").textContent = judge ? "tool × cenário" : "tool × cenário × juiz";
    document.getElementById("tile-ok").textContent = stats.ok;
    document.getElementById("tile-ok-pct").textContent = stats.total ? (100 * stats.ok / stats.total).toFixed(1) + "%" : "—";
    document.getElementById("tile-error").textContent = stats.error;
    document.getElementById("tile-error-pct").textContent = stats.total ? (100 * stats.error / stats.total).toFixed(1) + "%" : "—";
    document.getElementById("tile-refused").textContent = stats.refused;
  }

  // ---- narrative analysis (Gemini), only rendered when the cache file existed at
  // generation time (scripts/generate_narrative_analysis.py) -- every narrative-* element
  // stays empty/hidden otherwise, this is never an error state. Placed right after each
  // section it analyzes (not one block at the top) and, for the 3 fields that vary by
  // breakdown, re-rendered on every ai-tab switch by renderNarrativeForBreakdown() below --
  // same "Todos"/per-judge segmentation the charts already use, not a separate mechanism.
  function setNarrativeText(id, text) {
    const el = document.getElementById(id);
    el.textContent = text || "";
  }

  function renderNarrativeForBreakdown(key) {
    if (!DATA.narrative) return;
    const entry = (DATA.narrative.by_breakdown || []).find(b => b.key === key);
    setNarrativeText("narrative-rubric-components", entry && entry.rubric_components);
    setNarrativeText("narrative-scenario-comparison", entry && entry.scenario_comparison);
    setNarrativeText("narrative-wilcoxon-significance", entry && entry.wilcoxon_significance);
  }

  if (DATA.narrative) {
    const n = DATA.narrative;
    document.getElementById("narrative-summary").hidden = false;
    document.getElementById("narrative-model").textContent = n.model;
    document.getElementById("narrative-overall-text").textContent = n.overall_summary;
    setNarrativeText("narrative-judges", n.judges_analysis);
    setNarrativeText("narrative-divergences", n.divergences_analysis);
  }

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

  function renderWilcoxonSection(breakdown) {
    const section = document.getElementById("section-wilcoxon");
    const wilcoxon = breakdown.wilcoxon || [];
    if (wilcoxon.length === 0) {
      section.style.display = "none";
      return;
    }
    section.style.display = "";
    const tbody = document.getElementById("wilcoxon-rows");
    tbody.innerHTML = "";
    wilcoxon.forEach(w => {
      const tr = document.createElement("tr");
      const pValorHTML = w.p_valor_bh === null ? "—" : w.p_valor_bh.toFixed(4);
      const empatesHTML = w.proporcao_empates_percentual === null ? "—" : w.proporcao_empates_percentual.toFixed(1) + "%";
      tr.innerHTML = `
        <td>${w.label}</td>
        <td class="num tabular">${w.n_pareado}</td>
        <td class="num tabular">${empatesHTML}</td>
        <td class="num tabular">${pValorHTML}</td>
        <td>${w.significativo ? `<span class="pill ok">sim</span>` : `<span class="pill">não</span>`}</td>`;
      tbody.appendChild(tr);
    });
  }

  function selectAiTab(key, label) {
    document.querySelectorAll(".ai-tab-btn").forEach(b => b.setAttribute("aria-selected", String(b.dataset.key === key)));
    document.getElementById("components-note").textContent = key === "__all__"
      ? "Média de todas as avaliações concluídas com sucesso, nos cenários combinados. Ordenado do melhor para o pior."
      : `Média das avaliações de ${label}, nos cenários combinados. Ordenado do melhor para o pior.`;
    renderTiles(key);
    renderComponentsChart(DATA.breakdowns[key]);
    renderScenariosChart(DATA.breakdowns[key]);
    renderWilcoxonSection(DATA.breakdowns[key]);
    renderNarrativeForBreakdown(key);
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

  // ---- divergences tab ----
  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---- divergences table filters (mirrors the Tools tab toolbar) + AI (judge)
  // segmentation (mirrors Visão geral's Todos/por-juiz tabs) -- activeDivJudge is shared
  // with the boxplot and the cost-benefit table below it, so all three move together. ----
  let divergencesFilter = "";
  let activeDivLanguage = "";
  let activeDivComponente = "";
  let activeDivDirection = "";
  let activeDivJudge = "__all__";

  const divLanguageSelect = document.getElementById("divergences-filter-language");
  const divComponenteSelect = document.getElementById("divergences-filter-componente");
  const divDirectionSelect = document.getElementById("divergences-filter-direction");

  [...new Set(DATA.divergences.map(d => d.language))].sort().forEach(lang => {
    const opt = document.createElement("option");
    opt.value = lang;
    opt.textContent = lang;
    divLanguageSelect.appendChild(opt);
  });
  [...new Set(DATA.divergences.map(d => d.componente_label))].sort().forEach(label => {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    divComponenteSelect.appendChild(opt);
  });

  function renderDivergencesTable() {
    document.getElementById("divergence-min-change").textContent = DATA.meta.divergence_min_quartile_change;
    const tbody = document.getElementById("divergences-rows");
    const emptyEl = document.getElementById("divergences-empty");
    const countEl = document.getElementById("divergences-count");
    tbody.innerHTML = "";

    const q = divergencesFilter.trim().toLowerCase();
    const rows = DATA.divergences.filter(d =>
      (activeDivJudge === "__all__" || d.judge_id === activeDivJudge) &&
      (!q || d.tool_name.toLowerCase().includes(q) || d.repo.toLowerCase().includes(q) || d.qualified_name.toLowerCase().includes(q)) &&
      (!activeDivLanguage || d.language === activeDivLanguage) &&
      (!activeDivComponente || d.componente_label === activeDivComponente) &&
      (!activeDivDirection || (activeDivDirection === "subiu" ? d.diff > 0 : d.diff < 0))
    );
    countEl.textContent = `${rows.length} de ${DATA.divergences.length} divergência(s)`;
    emptyEl.hidden = rows.length > 0;

    rows.forEach(d => {
      const tr = document.createElement("tr");
      tr.className = "tool-row";
      tr.tabIndex = 0;
      tr.setAttribute("aria-expanded", "false");
      const diffClass = d.diff > 0 ? "" : "err";
      const motivosHTML = d.motivos.map(m =>
        `<span class="pill ${m === DATA.meta.motivo_fallback ? "warn" : ""}">${escapeHtml(DATA.meta.motivo_labels[m] || m)}</span>`
      ).join("");
      tr.innerHTML = `
        <td><span class="expand-icon">▸</span>${escapeHtml(d.componente_label)}</td>
        <td><span class="name">${escapeHtml(d.tool_name)}</span><span class="qualified">${escapeHtml(d.qualified_name)}</span></td>
        <td>${escapeHtml(d.repo)}</td>
        <td>${escapeHtml(d.judge_id)}</td>
        <td class="num score-cell tabular">${d.description_only.score}</td>
        <td class="num score-cell tabular">${d.with_source.score}</td>
        <td class="num score-cell tabular ${diffClass}">${d.diff > 0 ? "+" : ""}${d.diff}</td>
        <td class="num score-cell tabular">Q${d.quartil_description_only} → Q${d.quartil_with_source}</td>
        <td><div class="motivo-badges">${motivosHTML}</div></td>`;

      const detailTr = document.createElement("tr");
      detailTr.className = "tool-detail";
      detailTr.hidden = true;
      const detailCell = document.createElement("td");
      detailCell.colSpan = 9;
      detailCell.innerHTML = `
        <div class="detail-grid">
          <div class="detail-scenario">
            <h4>description_only (nota ${d.description_only.score}, Q${d.quartil_description_only})</h4>
            <div class="comp-row" style="display:block; white-space:pre-wrap;">${escapeHtml(d.description_only.reasoning) || "<em>sem justificativa registrada</em>"}</div>
          </div>
          <div class="detail-scenario">
            <h4>with_source (nota ${d.with_source.score}, Q${d.quartil_with_source})</h4>
            <div class="comp-row" style="display:block; white-space:pre-wrap;">${escapeHtml(d.with_source.reasoning) || "<em>sem justificativa registrada</em>"}</div>
          </div>
        </div>`;
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

  // ---- divergences boxplot: score distribution per component, description_only vs
  // with_source, mapped to a fixed 1-5 domain (the Likert scale itself, not 0-5 like the
  // mean bars above -- a boxplot's whole point is showing spread within the scale, so
  // anchoring at 1 instead of 0 uses the space that actually holds data). ----
  function boxplotTrackHTML(stats, color) {
    if (!stats) return `<div class="boxplot-track"></div>`;
    const x = v => ((v - 1) / 4) * 100;
    const whiskerLeft = x(stats.min), whiskerWidth = x(stats.max) - x(stats.min);
    const boxLeft = x(stats.q1), boxWidth = x(stats.q3) - x(stats.q1);
    return `
      <div class="boxplot-track" tabindex="0" style="color:${color}">
        <div class="grid-tick" style="left:25%"></div>
        <div class="grid-tick major" style="left:50%"></div>
        <div class="grid-tick" style="left:75%"></div>
        <div class="boxplot-whisker" style="left:${whiskerLeft}%; width:${whiskerWidth}%"></div>
        <div class="boxplot-whisker-cap" style="left:${x(stats.min)}%"></div>
        <div class="boxplot-whisker-cap" style="left:${x(stats.max)}%"></div>
        <div class="boxplot-box" style="left:${boxLeft}%; width:${boxWidth}%"></div>
        <div class="boxplot-median" style="left:${x(stats.median)}%"></div>
      </div>`;
  }

  function renderDivergencesBoxplot(key) {
    const legendEl = document.getElementById("boxplot-legend");
    legendEl.innerHTML = scenarioKeys.map((s, i) =>
      `<span class="key"><span class="swatch" style="background:${SERIES_COLORS[i]}"></span>${s}</span>`
    ).join("");
    document.getElementById("boxplot-note").textContent = key === "__all__"
      ? "Para cada componente da rubrica, boxplot (mínimo, Q1, mediana, Q3, máximo) da nota em description_only contra with_source, somando os dois juízes -- a forma completa da distribuição em cada cenário, não só a média."
      : `O mesmo boxplot, só para as avaliações de ${key}.`;

    const container = document.getElementById("chart-divergences-boxplot");
    container.innerHTML = "";
    (DATA.breakdowns[key].boxplot || []).forEach(c => {
      const row = document.createElement("div");
      row.className = "boxplot-row";
      const tracks = scenarioKeys.map((s, i) => boxplotTrackHTML(c.scenarios[s], SERIES_COLORS[i])).join("");
      row.innerHTML = `<div class="row-label">${escapeHtml(c.componente_label)}</div><div class="boxplot-tracks">${tracks}</div>`;
      row.querySelectorAll(".boxplot-track").forEach((track, i) => {
        const stats = c.scenarios[scenarioKeys[i]];
        if (!stats) return;
        wireTooltip(track, `<b>${escapeHtml(c.componente_label)}</b> · ${scenarioKeys[i]}<br>mín ${stats.min} · Q1 ${stats.q1.toFixed(2)} · mediana ${stats.median.toFixed(2)} · Q3 ${stats.q3.toFixed(2)} · máx ${stats.max} · N=${stats.n}`);
      });
      container.appendChild(row);
    });
  }

  // ---- motivos de mudança (junção por palavra-chave) -- combinado entre juízes só; a
  // granularidade por juiz completa fica nos CSVs de analysis_evaluation_report.py (ver
  // build_motivos_summary_data()), então esta seção não participa da segmentação por AI tab
  // abaixo. ----
  function renderMotivosSummary() {
    const tbody = document.getElementById("motivos-summary-rows");
    const emptyEl = document.getElementById("motivos-summary-empty");
    tbody.innerHTML = "";
    const rows = DATA.motivos_summary;
    emptyEl.hidden = rows.length > 0;
    rows.forEach(m => {
      const tr = document.createElement("tr");
      const motivoClass = m.motivo === DATA.meta.motivo_fallback ? "warn" : "";
      tr.innerHTML = `
        <td>${escapeHtml(m.componente_label)}</td>
        <td><span class="pill ${motivoClass}">${escapeHtml(m.motivo_label)}</span></td>
        <td class="num tabular">${m.ocorrencias}</td>
        <td class="num tabular">${m.n_subiu}</td>
        <td class="num tabular">${m.n_desceu}</td>`;
      tbody.appendChild(tr);
    });
  }
  renderMotivosSummary();

  // ---- consenso entre juízes: resumo por componente (mais forte que motivos: exige
  // unanimidade, não maioria) ----
  function renderConsensoSummary() {
    const tbody = document.getElementById("consenso-summary-rows");
    const emptyEl = document.getElementById("consenso-summary-empty");
    tbody.innerHTML = "";
    const rows = DATA.consenso_summary;
    emptyEl.hidden = rows.length > 0;
    rows.forEach(c => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(c.componente_label)}</td>
        <td class="num tabular">${c.n_tools_consenso}</td>
        <td class="num tabular">${c.n_tools_mesma_direcao}</td>
        <td class="num tabular">${c.media_juizes_por_consenso.toFixed(2)}</td>`;
      tbody.appendChild(tr);
    });
  }
  renderConsensoSummary();
  document.getElementById("consenso-min-juizes").textContent = DATA.meta.min_juizes_consenso;

  // ---- consenso entre juízes: detalhe por tool × componente (filtros mirram a tabela de
  // divergências acima, sem segmentação por juiz porque o universo de juízes já varia por
  // linha) ----
  let consensoFilter = "";
  let activeConsensoComponente = "";
  let activeConsensoDirection = "";

  const consensoComponenteSelect = document.getElementById("consenso-filter-componente");
  const consensoDirectionSelect = document.getElementById("consenso-filter-direction");
  [...new Set(DATA.consenso_juizes.map(c => c.componente_label))].sort().forEach(label => {
    const opt = document.createElement("option");
    opt.value = label;
    opt.textContent = label;
    consensoComponenteSelect.appendChild(opt);
  });

  function renderConsensoTable() {
    const tbody = document.getElementById("consenso-rows");
    const emptyEl = document.getElementById("consenso-empty");
    const countEl = document.getElementById("consenso-count");
    tbody.innerHTML = "";

    const q = consensoFilter.trim().toLowerCase();
    const rows = DATA.consenso_juizes.filter(c =>
      (!q || c.tool_name.toLowerCase().includes(q) || c.repo.toLowerCase().includes(q) || c.qualified_name.toLowerCase().includes(q)) &&
      (!activeConsensoComponente || c.componente_label === activeConsensoComponente) &&
      (!activeConsensoDirection || (activeConsensoDirection === "mesma" ? c.mesma_direcao : !c.mesma_direcao))
    );
    countEl.textContent = `${rows.length} de ${DATA.consenso_juizes.length} tool(s) × componente`;
    emptyEl.hidden = rows.length > 0;

    rows.forEach(c => {
      const tr = document.createElement("tr");
      const diffsHTML = c.juizes.map(j => `${escapeHtml(j.judge_id)} (${j.diff > 0 ? "+" : ""}${j.diff})`).join(" · ");
      tr.innerHTML = `
        <td>${escapeHtml(c.componente_label)}</td>
        <td><span class="name">${escapeHtml(c.tool_name)}</span><span class="qualified">${escapeHtml(c.qualified_name)}</span></td>
        <td>${escapeHtml(c.repo)}</td>
        <td class="num tabular">${c.n_juizes}</td>
        <td class="tabular">${diffsHTML}</td>
        <td><span class="pill ${c.mesma_direcao ? "ok" : ""}">${c.mesma_direcao ? "mesma" : "mista"}</span></td>`;
      tbody.appendChild(tr);
    });
  }
  renderConsensoTable();

  document.getElementById("consenso-search").addEventListener("input", (e) => { consensoFilter = e.target.value; renderConsensoTable(); });
  consensoComponenteSelect.addEventListener("change", (e) => { activeConsensoComponente = e.target.value; renderConsensoTable(); });
  consensoDirectionSelect.addEventListener("change", (e) => { activeConsensoDirection = e.target.value; renderConsensoTable(); });

  // ---- custo real (USD) já gasto, por juiz ----
  function renderCostTiles() {
    const container = document.getElementById("cost-tiles");
    container.innerHTML = "";
    DATA.custo_real.forEach(c => {
      const tile = document.createElement("div");
      tile.className = "tile";
      const valorHTML = c.gratuito
        ? `<span class="value tabular" style="color:var(--status-good)">Grátis</span>`
        : `<span class="value tabular">US$ ${c.custo_total_usd.toFixed(4)}</span>`;
      const subParts = [`${c.avaliacoes} avaliações`];
      if (c.gratuito) subParts.push("free tier / local");
      if (c.custo_nao_modelado > 0) subParts.push(`${c.custo_nao_modelado} sem preço modelado`);
      tile.innerHTML = `<span class="label">${escapeHtml(c.juiz)}</span>${valorHTML}<span class="sub">${subParts.join(" · ")}</span>`;
      container.appendChild(tile);
    });
  }
  renderCostTiles();

  // ---- custo-benefício: vale a pena o código? ----
  function renderTradeoffTable() {
    const tbody = document.getElementById("tradeoff-rows");
    const emptyEl = document.getElementById("tradeoff-empty");
    tbody.innerHTML = "";
    const rows = DATA.tradeoff.filter(v => activeDivJudge === "__all__" || v.juiz === activeDivJudge);
    emptyEl.hidden = rows.length > 0;

    const valeAPenaClass = { "sim": "ok", "não": "error", "inconclusivo": "" };

    rows.forEach(v => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${escapeHtml(v.juiz)}</td>
        <td>${escapeHtml(v.componente_label)}</td>
        <td><span class="pill">${escapeHtml(v.direcao)}</span></td>
        <td><span class="pill ${v["significativo_bh_0.05"] ? "ok" : ""}">${v["significativo_bh_0.05"] ? "sim" : "não"}</span></td>
        <td class="num tabular">${v.pct_sem_motivo === null ? "—" : v.pct_sem_motivo.toFixed(1) + "%"}</td>
        <td class="num tabular">${v.delta_input_tokens === null ? "—" : (v.delta_input_tokens > 0 ? "+" : "") + v.delta_input_tokens.toFixed(0)}</td>
        <td class="num tabular">${v.custo_percentual_extra === null ? "—" : v.custo_percentual_extra.toFixed(1) + "%"}</td>
        <td class="num tabular">${v.custo_extra_usd === null ? "—" : "US$ " + (v.custo_extra_usd * 1000).toFixed(2)}</td>
        <td><span class="pill ${valeAPenaClass[v.vale_a_pena] || ""}">${escapeHtml(v.vale_a_pena)}</span></td>`;
      tbody.appendChild(tr);
    });
  }

  function selectDivergencesAiTab(key, label) {
    document.querySelectorAll(".div-ai-tab-btn").forEach(b => b.setAttribute("aria-selected", String(b.dataset.key === key)));
    activeDivJudge = key;
    renderDivergencesBoxplot(key);
    renderDivergencesTable();
    renderTradeoffTable();
  }

  const divAiTabsEl = document.getElementById("divergences-ai-tabs");
  const divAiTabDefs = [{ key: "__all__", label: "Todos" }, ...DATA.judges.map(j => ({ key: j.id, label: j.id }))];
  divAiTabDefs.forEach(({ key, label }) => {
    const btn = document.createElement("button");
    btn.className = "ai-tab-btn div-ai-tab-btn";
    btn.type = "button";
    btn.dataset.key = key;
    btn.setAttribute("role", "tab");
    btn.setAttribute("aria-selected", key === "__all__" ? "true" : "false");
    btn.textContent = label;
    btn.addEventListener("click", () => selectDivergencesAiTab(key, label));
    divAiTabsEl.appendChild(btn);
  });

  document.getElementById("divergences-search").addEventListener("input", (e) => { divergencesFilter = e.target.value; renderDivergencesTable(); });
  divLanguageSelect.addEventListener("change", (e) => { activeDivLanguage = e.target.value; renderDivergencesTable(); });
  divComponenteSelect.addEventListener("change", (e) => { activeDivComponente = e.target.value; renderDivergencesTable(); });
  divDirectionSelect.addEventListener("change", (e) => { activeDivDirection = e.target.value; renderDivergencesTable(); });

  selectDivergencesAiTab("__all__", "Todos");

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


def load_narrative_analysis(path: Path) -> dict | None:
    """Lê o cache escrito por scripts/generate_narrative_analysis.py, se existir -- esta
    função NUNCA chama a API do Gemini (essa chamada é cara/rate-limited/de rede, então fica
    isolada num script separado, rodado manualmente). Sem o arquivo, o dashboard renderiza
    normalmente com a seção de análise narrativa oculta, não como um erro.
    """
    if not path.exists():
        logger.info("%s não encontrado -- dashboard sem seção de análise narrativa (rode scripts.generate_narrative_analysis para gerá-la).", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera dashboard HTML dos resultados da Etapa 3.")
    parser.add_argument("--judge", type=str, default=None, help="judge_id único (default: todos em data/evaluations/).")
    parser.add_argument("--output", type=Path, default=None, help="Caminho de saída (default: data/analysis/dashboard.html).")
    parser.add_argument(
        "--prompt-version", type=str, default=None,
        help="Restringe Visão geral/Tools a este prompt_version (default: o mais recente presente nos dados -- ver aba Versões).",
    )
    parser.add_argument(
        "--narrative", type=Path, default=None,
        help="Caminho do cache de análise narrativa (default: data/analysis/narrative_analysis.json, se existir -- ver scripts.generate_narrative_analysis).",
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
    data["narrative"] = load_narrative_analysis(args.narrative or (DATA_DIR / "analysis" / "narrative_analysis.json"))
    data["dataset"] = build_dataset_data()
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
