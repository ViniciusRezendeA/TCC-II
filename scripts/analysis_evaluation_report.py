from __future__ import annotations

"""Gera gráficos e planilhas da Etapa 3 (avaliação via LLM-as-a-Judge): confiabilidade
técnica do lote (ok/refused/error), notas médias por componente/juiz/cenário, comparação
description_only vs with_source (a pergunta de pesquisa central do TCC), concordância entre
juízes, custo/latência e notas por linguagem do repositório.

Companion de scripts/analysis_report.py (Etapas 1-2): mesma convenção de saída
(charts/*.png, tables/*.csv, uma planilha resumo_etapa_3.xlsx em vez de
resumo_etapas_1_2.xlsx), reaproveitando os helpers de gráfico/planilha de lá em vez de
duplicá-los.

Lê só data/evaluations/{judge_id}.jsonl (saída de pipeline/run_step3.py) -- não faz
nenhuma chamada de API nem depende de mcp_pipeline.evaluation.judges.*. Não precisa cruzar
com data/dataset.jsonl: repo.primary_language já vem denormalizado em cada registro de
avaliação (ver pipeline/run_step3.py::_base_record).

Pareamento por tool feito via tool_key_for() (tool_uid + tool.name), não pelo tool_uid bruto
do registro: para os padrões de SDK "lowlevel" (ver pipeline/run_step3.py::tool_uid_for),
várias tools distintas compartilham um tool_uid porque herdam a localização do handler que
as registra. tool_key_for() corrige isso a partir de dados já presentes em todo registro já
coletado (tool.name), sem precisar reprocessar nenhuma avaliação.

A maioria das tabelas aqui é descritiva (médias, medianas, contagens, correlação).
wilcoxon_por_componente() implementa o teste de significância pareado da Seção "Análise
Comparativa dos Experimentos" do TCC: Teste de Postos Sinalizados de Wilcoxon, por
componente da rubrica e por juiz, com correção de Benjamini-Hochberg (FDR) para as 6
comparações simultâneas dentro de cada juiz, e relato da proporção de pares empatados
(diferença zero) descartados por dimensão -- conforme especificado no texto.

Uso:
    uv run python -m scripts.analysis_evaluation_report [--evaluations-dir PATH] [--output-dir PATH]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging
from scripts.analysis_report import (
    CHART_STYLE,
    _bar_chart,
    _grouped_bar_chart,
    export_tables,
    load_jsonl,
)

import matplotlib.pyplot as plt  # noqa: E402 -- depois de scripts.analysis_report para herdar seu matplotlib.use("Agg")

logger = setup_logging("analysis_evaluation_report")

RUBRIC_COMPONENTS = [
    "purpose",
    "guidelines",
    "limitations",
    "parameter_explanation",
    "length_completeness",
    "examples",
]


# --- Carregamento -------------------------------------------------------------


def load_evaluations(evaluations_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(evaluations_dir.glob("*.jsonl")):
        records.extend(load_jsonl(path))
    return records


def registros_versao_ativa(records: list[dict]) -> list[dict]:
    """Restringe records à versão mais recente de prompt_version presente neles -- mesma lógica
    que generate_dashboard.py::build_dashboard_data() já usa para as abas Visão Geral/Tools/
    Divergências (fatorada aqui para as duas pontas compartilharem uma só implementação).

    Necessário porque cada prompt_version pode ter um texto de rubrica/instruções diferente para
    o juiz (ver RUBRIC_SYSTEM_PROMPT em evaluation/prompts.py, que muda a cada bump de
    PROMPT_VERSION) -- misturar avaliações de versões diferentes do prompt na mesma tabela
    compararia respostas a instruções distintas como se fossem a mesma coisa. Esta função (e
    tudo que se baseia nela, como motivos_por_divergencia() e veredito_custo_beneficio())
    considera só a instrução que está de fato ativa hoje (a versão mais recente presente nos
    dados) -- versões anteriores do prompt, mesmo que documentadas no changelog de
    PROMPT_VERSION, não entram em nenhuma conta.
    """
    if not records:
        return []
    active_version = max(r.get("prompt_version") or "(sem versão)" for r in records)
    return [r for r in records if (r.get("prompt_version") or "(sem versão)") == active_version]


def tool_key_for(record: dict) -> str:
    """Chave de pareamento por tool, corrigida para colisões de tool_uid vindas dos padrões
    "lowlevel" de SDK (python.list_tools_lowlevel, *.set_request_handler_lowlevel): nesses
    padrões, várias tools distintas (nomes/descrições diferentes) compartilham o mesmo
    tool_uid porque todas herdam a localização do handler que as registra, não uma
    localização própria (ver tool_uid_for() em pipeline/run_step3.py). Sem essa correção,
    pivot_table() agrega essas tools diferentes pela média em notas_por_componente(),
    comparacao_cenarios() e concordancia_entre_juizes(), e o pareamento
    description_only/with_source pode juntar a nota de uma tool com a de outra.

    Aplicado incondicionalmente (não só nos padrões lowlevel): registros de avaliação já
    coletados não guardam sdk_pattern, só tool.name -- e para os demais padrões, que já têm
    localização própria por tool, tool.name é redundante com tool_uid mas nunca o contradiz,
    então incluí-lo aqui não muda nenhum agrupamento que já estava correto. Não requer
    reprocessar nenhuma avaliação: tool.name já é gravado em cada registro por
    pipeline/run_step3.py::_base_record.

    Independente de safra do dado: pipeline/run_step3.py::tool_uid_for() também passou a
    gravar ::{name} no próprio tool_uid para os padrões lowlevel (avaliações novas), enquanto
    avaliações coletadas antes dessa correção guardam o tool_uid sem esse sufixo -- sem
    normalizar, a mesma tool real acabaria com duas chaves diferentes (uma por safra),
    quebrando o merge entre juízes quando um avaliou antes da correção e outro depois. Faz um
    strip do sufixo se ele já estiver lá antes de reanexar, para as duas safras convergirem
    na mesma chave.
    """
    tool_uid = record["tool_uid"]
    name = record["tool"]["name"]
    suffix = f"::{name}"
    base = tool_uid[: -len(suffix)] if tool_uid.endswith(suffix) else tool_uid
    return f"{base}{suffix}"


def scores_long(records: list[dict]) -> pd.DataFrame:
    """Formato longo (uma linha por tool x cenário x juiz x componente), só para
    avaliações com status "ok" -- base compartilhada pela maioria das métricas abaixo, em
    vez de cada uma reimplementar sua própria navegação em scores["componente"]["score"].
    status "refused"/"error" não têm scores e ficam de fora por construção; ver
    status_por_juiz_cenario() para a contagem dessas.
    """
    rows = []
    for r in records:
        if r["status"] != "ok":
            continue
        for component in RUBRIC_COMPONENTS:
            rows.append(
                {
                    "tool_uid": tool_key_for(r),
                    "cenario": r["scenario"],
                    "juiz": r["judge"]["id"],
                    "linguagem": r["repo"].get("primary_language"),
                    "componente": component,
                    "nota": r["scores"][component]["score"],
                }
            )
    return pd.DataFrame(rows)


# --- Métricas -------------------------------------------------------------------


def status_por_juiz_cenario(records: list[dict]) -> pd.DataFrame:
    """Confiabilidade técnica do lote: quantas avaliações terminaram ok/refused/error, por
    juiz e cenário. "refused" (recusa de segurança do provedor) é um resultado esperado
    para uma fração do dataset (ex: tools de servidores de pentest), não o mesmo que
    "error" (falha técnica) -- ver JudgeRefusal em evaluation/judges/base.py.
    """
    df = pd.DataFrame([{"juiz": r["judge"]["id"], "cenario": r["scenario"], "status": r["status"]} for r in records])
    counts = df.groupby(["juiz", "cenario", "status"]).size().unstack(fill_value=0).reset_index()
    for col in ("ok", "refused", "error"):
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts["ok"] + counts["refused"] + counts["error"]
    counts["taxa_ok_percentual"] = (counts["ok"] / counts["total"] * 100).round(1)
    return counts[["juiz", "cenario", "ok", "refused", "error", "total", "taxa_ok_percentual"]]


def notas_por_componente(long_df: pd.DataFrame) -> pd.DataFrame:
    """Tabela central de resultado da Etapa 3: nota média (Likert 1-5) de cada um dos 6
    componentes da rubrica, por juiz e cenário."""
    grouped = long_df.groupby(["juiz", "cenario", "componente"])["nota"].agg(["mean", "median", "std", "count"]).reset_index()
    grouped.columns = ["juiz", "cenario", "componente", "media", "mediana", "desvio_padrao", "n"]
    for col in ("media", "mediana", "desvio_padrao"):
        grouped[col] = grouped[col].round(2)
    return grouped.sort_values(["componente", "juiz", "cenario"]).reset_index(drop=True)


def comparacao_cenarios(long_df: pd.DataFrame) -> pd.DataFrame:
    """A pergunta de pesquisa central do TCC: dar acesso ao SOURCE_CODE (cenário
    with_source) muda a nota, por componente e por juiz, em relação a só a descrição
    (description_only)? `n_pareado` conta só as tools avaliadas em AMBOS os cenários pelo
    mesmo juiz (join por tool_uid) -- a base correta para uma futura análise pareada
    (Wilcoxon, Etapa 5), não o total de linhas de cada cenário isoladamente.
    """
    pivot = long_df.pivot_table(index=["juiz", "componente", "tool_uid"], columns="cenario", values="nota")
    rows = []
    for (juiz, componente), group in pivot.groupby(level=["juiz", "componente"]):
        desc = group["description_only"].dropna() if "description_only" in group else pd.Series(dtype=float)
        src = group["with_source"].dropna() if "with_source" in group else pd.Series(dtype=float)
        n_pareado = int(group.dropna().shape[0])
        rows.append(
            {
                "juiz": juiz,
                "componente": componente,
                "media_description_only": round(desc.mean(), 2) if len(desc) else None,
                "media_with_source": round(src.mean(), 2) if len(src) else None,
                "delta_with_source_menos_description_only": (
                    round(src.mean() - desc.mean(), 2) if len(desc) and len(src) else None
                ),
                "n_pareado": n_pareado,
            }
        )
    return pd.DataFrame(rows).sort_values(["componente", "juiz"]).reset_index(drop=True)


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Correção de Benjamini-Hochberg (FDR), método step-up padrão -- implementada aqui em
    vez de puxar statsmodels (não é dependência do projeto) só para uma fórmula que numpy já
    resolve. NaN (ex: componente sem par suficiente para o teste) passa direto, sem entrar
    na família de comparações nem herdar um valor de outra linha.
    """
    valid = pvalues.dropna()
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    if valid.empty:
        return result

    m = len(valid)
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy()
    raw_adjusted = ranked * m / np.arange(1, m + 1)
    # step-up: q(i) = min(q(i), q(i+1), ..., q(m)), acumulado do maior p-valor para o menor.
    adjusted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    result.loc[order] = np.minimum(adjusted, 1.0)
    return result


def wilcoxon_por_componente(long_df: pd.DataFrame) -> pd.DataFrame:
    """Teste de Postos Sinalizados de Wilcoxon (Seção "Análise Comparativa dos Experimentos"
    do TCC), aplicado por componente da rubrica e separadamente por juiz -- não misturado
    entre juízes, para poder verificar se o efeito da inclusão do código se mantém
    consistente entre diferentes avaliadores.

    H0: a distribuição das diferenças pareadas (with_source - description_only) é simétrica
    em torno de zero, por componente e juiz. H1: a inclusão do código desloca essa
    distribuição. Pares com diferença zero (mesma nota nos dois cenários) são descartados do
    teste -- tratamento padrão de empates do Wilcoxon (scipy zero_method="wilcox", o
    default) -- e sua proporção é reportada em `proporcao_empates_percentual` por
    componente/juiz, como indicador de quanto a inclusão do código muda (ou não) a nota.

    `p_valor_bh` corrige, dentro de cada juiz, os p-valores das 6 comparações simultâneas
    (uma por componente) via Benjamini-Hochberg -- a família de testes repetidos que o texto
    do TCC identifica como precisando de correção, não as 6 dimensões x N juízes juntas
    (juízes são reportados separadamente por design, não como uma família de testes conjunta).

    Usa o `tool_uid` de `long_df`, que já vem corrigido por tool_key_for() em scores_long()
    -- nunca o tool_uid bruto do registro, que colide entre tools distintas para os padrões
    de SDK "lowlevel" (ver tool_key_for()).
    """
    pivot = long_df.pivot_table(index=["juiz", "componente", "tool_uid"], columns="cenario", values="nota")
    rows = []
    for (juiz, componente), group in pivot.groupby(level=["juiz", "componente"]):
        if {"description_only", "with_source"} <= set(group.columns):
            paired = group.dropna(subset=["description_only", "with_source"])
        else:
            paired = group.iloc[0:0]
        n_pareado = len(paired)
        diffs = (paired["with_source"] - paired["description_only"]) if n_pareado else pd.Series(dtype=float)
        n_empates = int((diffs == 0).sum())
        n_efetivo = n_pareado - n_empates

        estatistica_w, p_valor = float("nan"), float("nan")
        if n_efetivo >= 1:
            try:
                estatistica_w, p_valor = wilcoxon(paired["with_source"], paired["description_only"], zero_method="wilcox")
            except ValueError:
                # todas as diferenças não-nulas se cancelam em soma de postos zero, ou outro
                # caso degenerado que o scipy recusa -- reportado como sem teste, não como erro.
                pass

        rows.append(
            {
                "juiz": juiz,
                "componente": componente,
                "n_pareado": n_pareado,
                "n_empates": n_empates,
                "proporcao_empates_percentual": round(n_empates / n_pareado * 100, 1) if n_pareado else None,
                "n_efetivo_teste": n_efetivo,
                "mediana_diferenca": round(diffs.median(), 2) if n_pareado else None,
                "estatistica_w": round(estatistica_w, 2) if estatistica_w == estatistica_w else None,
                "p_valor": p_valor,
            }
        )

    df = pd.DataFrame(rows)
    df["p_valor_bh"] = df.groupby("juiz")["p_valor"].transform(_benjamini_hochberg)
    df["significativo_bh_0.05"] = df["p_valor_bh"] < 0.05
    df["p_valor"] = df["p_valor"].round(4)
    df["p_valor_bh"] = df["p_valor_bh"].round(4)
    return df.sort_values(["juiz", "componente"]).reset_index(drop=True)


# Migrações de só 1 faixa (ex: Q1->Q2) não contam como divergência -- só a partir de 2
# faixas (Q1->Q3, Q2->Q4, ...). Valor escolhido com o usuário depois de comparar volumes:
# qualquer mudança gera ~5000 linhas só no deepseek-flash (ruído), >=2 faixas gera ~1300.
MUDANCA_MINIMA_QUARTIS = 2


def quartil_notas(long_df: pd.DataFrame) -> pd.DataFrame:
    """Classifica cada avaliação (uma linha de long_df) no quartil (1-4) da distribuição de
    notas do seu próprio (juiz, componente, cenário) -- rank percentual médio
    (`Series.rank(method="average", pct=True)`) em vez de comparar a nota contra os cortes
    brutos Q1/Q2/Q3 (`Series.quantile()`): a escala Likert 1-5 é discreta e MUITO concentrada
    em poucos valores (ex: 88% das notas de "examples"/deepseek-flash valem 1), o que faz
    Q1/Q2/Q3 colidirem no mesmo valor e a comparação direta falhar (não dá pra saber se uma
    nota "no limite" é Q1, Q2 ou Q3). Notas empatadas recebem o mesmo rank médio e caem,
    por construção, no mesmo quartil.
    """
    df = long_df.copy()
    rank_percentual = df.groupby(["juiz", "componente", "cenario"])["nota"].rank(method="average", pct=True)
    df["quartil"] = pd.cut(rank_percentual, [0, 0.25, 0.5, 0.75, 1.0], labels=[1, 2, 3, 4], include_lowest=True).astype(int)
    return df


def migracao_quartil_por_tool(long_df: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (juiz, componente, tool_uid): quartil da nota (ver quartil_notas()) em
    cada cenário e se essa migração conta como divergência.

    Pares empatados (mesma nota, logo mesmo quartil, em description_only e with_source) são
    descartados -- não migraram por definição. Entre os que sobram, só migrações de pelo
    menos MUDANCA_MINIMA_QUARTIS faixas contam como divergência (`diverge=True`); migrações
    de 1 faixa só (Q1->Q2) ficam registradas em `diff_quartil` mas não marcadas.

    Usa o mesmo pivot por (juiz, componente, tool_uid) de wilcoxon_por_componente() -- tool_uid
    de long_df já vem corrigido por tool_key_for() via scores_long().
    """
    notas = quartil_notas(long_df)
    nota_pivot = notas.pivot_table(index=["juiz", "componente", "tool_uid"], columns="cenario", values="nota")
    quartil_pivot = notas.pivot_table(index=["juiz", "componente", "tool_uid"], columns="cenario", values="quartil")
    combinado = nota_pivot.join(quartil_pivot, lsuffix="_nota", rsuffix="_quartil")

    obrigatorias = ["description_only_nota", "with_source_nota", "description_only_quartil", "with_source_quartil"]
    if not set(obrigatorias) <= set(combinado.columns):
        return pd.DataFrame(
            columns=[
                "juiz", "componente", "tool_uid", "quartil_description_only", "quartil_with_source",
                "diff_quartil", "empate", "diverge",
            ]
        )

    combinado = combinado.dropna(subset=obrigatorias).reset_index()
    combinado["quartil_description_only"] = combinado["description_only_quartil"].astype(int)
    combinado["quartil_with_source"] = combinado["with_source_quartil"].astype(int)
    combinado["diff_quartil"] = combinado["quartil_with_source"] - combinado["quartil_description_only"]
    combinado["empate"] = combinado["description_only_nota"] == combinado["with_source_nota"]
    combinado["diverge"] = (~combinado["empate"]) & (combinado["diff_quartil"].abs() >= MUDANCA_MINIMA_QUARTIS)

    return combinado[
        ["juiz", "componente", "tool_uid", "quartil_description_only", "quartil_with_source", "diff_quartil", "empate", "diverge"]
    ]


def migracao_quartil_por_componente(long_df: pd.DataFrame) -> pd.DataFrame:
    """Resumo por juiz/componente da migração de quartil de nota entre cenários (ver
    migracao_quartil_por_tool()): quantos pares empataram, quantos migraram só 1 faixa (não
    contam como divergência) e quantos migraram >= MUDANCA_MINIMA_QUARTIS faixas (contam),
    separado entre subiu (with_source melhor que description_only) e desceu.
    """
    df = migracao_quartil_por_tool(long_df)
    rows = []
    for (juiz, componente), group in df.groupby(["juiz", "componente"]):
        nao_empatados = group[~group["empate"]]
        n_migrou_1_faixa = int((nao_empatados["diff_quartil"].abs() == 1).sum())
        rows.append(
            {
                "juiz": juiz,
                "componente": componente,
                "n_pares": len(group),
                "n_empates": int(group["empate"].sum()),
                "n_migrou_1_faixa": n_migrou_1_faixa,
                "n_diverge": int(group["diverge"].sum()),
                "n_diverge_subiu": int((group["diverge"] & (group["diff_quartil"] > 0)).sum()),
                "n_diverge_desceu": int((group["diverge"] & (group["diff_quartil"] < 0)).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["componente", "juiz"]).reset_index(drop=True)


# --- Motivos de divergência (junção por palavra-chave) -----------------------

# Termos em inglês porque o reasoning das avaliações é sempre em inglês (ver
# evaluation/prompts.py). Derivado das definições dos 6 componentes da rubrica
# (RUBRIC_COMPONENTS em prompts.py) e do próprio texto de "Handling SOURCE_CODE" do prompt
# ativo (RUBRIC_SYSTEM_PROMPT): a nota só deveria ser ajustada "based on ... findings" de
# inconsistências entre descrição e código, então contradição/omissão são os motivos mais
# diretamente ligados à instrução em vigor. Uma divergência pode casar mais de uma categoria
# -- é uma junção por palavra-chave, não uma classificação exclusiva.
MOTIVO_KEYWORDS: dict[str, list[str]] = {
    "parametro": ["parameter", "argument", "data type", "type hint"],
    "limitacao_erro": ["error", "exception", "raise", "constraint", "edge case", "corner case", "fail"],
    "exemplo": ["example", "usage", "demonstrat"],
    "contradicao": ["contradict", "inconsistent", "mismatch", "does not match", "does not align", "conflicts with"],
    "omissao": ["omit", "missing", "does not mention", "lacks", "no mention of", "fails to mention", "not disclosed", "undisclosed"],
    "escopo_proposito": ["actually does", "actually performs", "purpose is", "the code reveals", "the implementation shows"],
}

# Retornado quando nenhuma categoria de MOTIVO_KEYWORDS casa -- ver classificar_motivos().
MOTIVO_FALLBACK = "sem_justificativa_especifica"


def classificar_motivos(reasoning: str) -> list[str]:
    """Junção por palavra-chave: casa o texto de reasoning contra MOTIVO_KEYWORDS,
    case-insensitive, por substring. Retorna TODAS as categorias que casarem (ex: "the
    parameter's type is missing" casa tanto "parametro" quanto "omissao"), não uma
    classificação exclusiva.

    Quando nenhuma categoria casa, retorna [MOTIVO_FALLBACK]: o texto ativo de "Handling
    SOURCE_CODE" (RUBRIC_SYSTEM_PROMPT, ver registros_versao_ativa()) diz que o ajuste de nota
    deve ser feito "based on ... findings" de inconsistências entre descrição e código -- uma
    divergência cuja reasoning não nomeia nenhum termo de problema reconhecível não está
    claramente fundamentada nesses termos, independente de a nota ter subido ou descido.
    """
    texto = (reasoning or "").lower()
    motivos = [motivo for motivo, termos in MOTIVO_KEYWORDS.items() if any(termo in texto for termo in termos)]
    return motivos or [MOTIVO_FALLBACK]


def motivos_por_divergencia(records: list[dict]) -> pd.DataFrame:
    """Aplica classificar_motivos() a cada divergência (ver migracao_quartil_por_tool()): para
    cada (juiz, componente, tool_uid) cuja nota migrou pelo menos MUDANCA_MINIMA_QUARTIS faixas
    de quartil, busca a reasoning do with_source diretamente nos records brutos (mesma técnica
    de lookup por chave usada em generate_dashboard.py::build_divergences_data()) e classifica.
    Usa só a reasoning do with_source, não a do description_only: é nela que o juiz, seguindo o
    texto ativo de "Handling SOURCE_CODE", deveria registrar a inconsistência encontrada.

    Uma linha por (divergência x motivo casado) -- fan-out proposital de uma junção real, não
    uma classificação 1:1. `records` deve já ter passado por registros_versao_ativa(); esta
    função não filtra por conta própria (para poder ser testada isoladamente com fixtures de
    uma só versão).
    """
    long_df = scores_long(records)
    migracao = migracao_quartil_por_tool(long_df)
    divergentes = migracao[migracao["diverge"]]
    if divergentes.empty:
        return pd.DataFrame(columns=["juiz", "componente", "tool_uid", "motivo", "diff_quartil", "subiu"])

    reasoning_with_source: dict[tuple[str, str, str], str] = {}
    for r in records:
        if r.get("status") != "ok" or not r.get("scores") or r.get("scenario") != "with_source":
            continue
        tool_uid = tool_key_for(r)
        juiz = r["judge"]["id"]
        for componente, dados in r["scores"].items():
            if dados:
                reasoning_with_source[(juiz, componente, tool_uid)] = dados.get("reasoning") or ""

    rows = []
    for row in divergentes.to_dict("records"):
        chave = (row["juiz"], row["componente"], row["tool_uid"])
        reasoning = reasoning_with_source.get(chave, "")
        for motivo in classificar_motivos(reasoning):
            rows.append(
                {
                    "juiz": row["juiz"],
                    "componente": row["componente"],
                    "tool_uid": row["tool_uid"],
                    "motivo": motivo,
                    "diff_quartil": row["diff_quartil"],
                    "subiu": row["diff_quartil"] > 0,
                }
            )
    return pd.DataFrame(rows)


def resumo_motivos_por_componente(motivos_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega motivos_por_divergencia() por juiz/componente/motivo -- responde "quais são os
    motivos da mudança de nota": quantas divergências cada motivo cobre, quantas foram subida
    vs descida de quartil. Granularidade por juiz incluída porque veredito_custo_beneficio()
    precisa dela nesse nível; uma visão agregada entre juízes pode somar esta tabela por
    (componente, motivo) na camada de apresentação (ex: dashboard).

    `ocorrencias` conta linhas já explodidas por motivo (uma divergência com 2 motivos conta 2
    vezes aqui) -- correto para responder "quantas vezes esse motivo apareceu", mas não deve ser
    somado entre motivos para estimar o total de divergências (ver n_diverge em
    migracao_quartil_por_componente() para isso).
    """
    if motivos_df.empty:
        return pd.DataFrame(columns=["juiz", "componente", "motivo", "ocorrencias", "n_subiu", "n_desceu", "pct_subiu"])

    grouped = (
        motivos_df.groupby(["juiz", "componente", "motivo"])
        .agg(ocorrencias=("subiu", "size"), n_subiu=("subiu", "sum"))
        .reset_index()
    )
    grouped["n_desceu"] = grouped["ocorrencias"] - grouped["n_subiu"]
    grouped["pct_subiu"] = (grouped["n_subiu"] / grouped["ocorrencias"] * 100).round(1)
    return grouped.sort_values(["componente", "juiz", "ocorrencias"], ascending=[True, True, False]).reset_index(drop=True)


def concordancia_entre_juizes(long_df: pd.DataFrame) -> pd.DataFrame:
    """Concordância par-a-par entre juízes: correlação de Pearson e diferença média
    absoluta das notas dadas ao mesmo (tool, cenário, componente) -- indica se o júri
    multi-provedor converge ou se cada modelo julga de forma sistematicamente diferente.
    """
    wide = long_df.pivot_table(index=["tool_uid", "cenario", "componente"], columns="juiz", values="nota")
    juizes = sorted(wide.columns)
    rows = []
    for i, a in enumerate(juizes):
        for b in juizes[i + 1 :]:
            pair = wide[[a, b]].dropna()
            if pair.empty:
                continue
            rows.append(
                {
                    "juiz_a": a,
                    "juiz_b": b,
                    "n_comparacoes": len(pair),
                    "correlacao_pearson": round(pair[a].corr(pair[b]), 2),
                    "diferenca_media_absoluta": round((pair[a] - pair[b]).abs().mean(), 2),
                }
            )
    return pd.DataFrame(rows)


def custo_latencia_por_juiz(records: list[dict]) -> pd.DataFrame:
    """Tokens e latência médios por juiz -- só entre avaliações "ok" (usage/latency não
    são preenchidos para refused/error). Insumo para a seção de metodologia/limitações do
    TCC (custo e tempo de execução da Etapa 3)."""
    rows = [
        {
            "juiz": r["judge"]["id"],
            "input_tokens": r["usage"]["input_tokens"],
            "output_tokens": r["usage"]["output_tokens"],
            "cache_read_input_tokens": r["usage"]["cache_read_input_tokens"],
            "latency_ms": r["latency_ms"],
        }
        for r in records
        if r["status"] == "ok"
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "juiz", "avaliacoes", "media_input_tokens", "media_output_tokens",
                "media_cache_read_tokens", "media_latencia_ms", "mediana_latencia_ms",
            ]
        )
    grouped = (
        df.groupby("juiz")
        .agg(
            avaliacoes=("latency_ms", "count"),
            media_input_tokens=("input_tokens", "mean"),
            media_output_tokens=("output_tokens", "mean"),
            media_cache_read_tokens=("cache_read_input_tokens", "mean"),
            media_latencia_ms=("latency_ms", "mean"),
            mediana_latencia_ms=("latency_ms", "median"),
        )
        .round(1)
        .reset_index()
    )
    return grouped


def custo_latencia_por_juiz_e_cenario(records: list[dict]) -> pd.DataFrame:
    """Como custo_latencia_por_juiz(), mas separado por cenário -- necessário para isolar o
    custo extra especificamente atribuível a mandar SOURCE_CODE (with_source vs
    description_only), que a tabela agregada por juiz não distingue. Insumo de
    _delta_custo_com_codigo() e, por extensão, de veredito_custo_beneficio().
    """
    rows = [
        {
            "juiz": r["judge"]["id"],
            "cenario": r["scenario"],
            "input_tokens": r["usage"]["input_tokens"],
            "output_tokens": r["usage"]["output_tokens"],
            "latency_ms": r["latency_ms"],
        }
        for r in records
        if r["status"] == "ok"
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["juiz", "cenario", "avaliacoes", "media_input_tokens", "media_output_tokens", "media_latencia_ms"]
        )
    grouped = (
        df.groupby(["juiz", "cenario"])
        .agg(
            avaliacoes=("latency_ms", "count"),
            media_input_tokens=("input_tokens", "mean"),
            media_output_tokens=("output_tokens", "mean"),
            media_latencia_ms=("latency_ms", "mean"),
        )
        .round(1)
        .reset_index()
    )
    return grouped


def _delta_custo_com_codigo(custo_por_cenario: pd.DataFrame) -> pd.DataFrame:
    """Custo extra (with_source menos description_only) por juiz, a partir de
    custo_latencia_por_juiz_e_cenario() -- o preço específico de mandar SOURCE_CODE, isolado do
    custo-base de avaliar só a description. Helper interno de veredito_custo_beneficio(), não
    uma tabela exportada por conta própria.
    """
    colunas = ["juiz", "delta_input_tokens", "delta_latencia_ms", "custo_percentual_extra"]
    if custo_por_cenario.empty:
        return pd.DataFrame(columns=colunas)

    desc = custo_por_cenario[custo_por_cenario["cenario"] == "description_only"].set_index("juiz")
    src = custo_por_cenario[custo_por_cenario["cenario"] == "with_source"].set_index("juiz")
    juizes_comuns = desc.index.intersection(src.index)
    if juizes_comuns.empty:
        return pd.DataFrame(columns=colunas)

    delta_tokens = (src.loc[juizes_comuns, "media_input_tokens"] - desc.loc[juizes_comuns, "media_input_tokens"]).round(1)
    delta_latencia = (src.loc[juizes_comuns, "media_latencia_ms"] - desc.loc[juizes_comuns, "media_latencia_ms"]).round(1)
    custo_percentual = (delta_tokens / desc.loc[juizes_comuns, "media_input_tokens"] * 100).round(1)

    return pd.DataFrame(
        {
            "juiz": juizes_comuns,
            "delta_input_tokens": delta_tokens.values,
            "delta_latencia_ms": delta_latencia.values,
            "custo_percentual_extra": custo_percentual.values,
        }
    ).reset_index(drop=True)


def _mediana_diferenca_efetiva_por_componente(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mediana das diferenças pareadas NÃO-NULAS (with_source - description_only) por (juiz,
    componente) -- diferente de wilcoxon_por_componente()::mediana_diferenca, que inclui pares
    empatados (diferença 0) e por isso fica arrastada para perto de 0 sempre que a maioria dos
    pares empata (comum na escala Likert 1-5 concentrada, ver docstring de quartil_notas()). O
    teste de Wilcoxon também descarta os empates (zero_method="wilcox") -- esta função usa a
    mesma base de pares efetivos, para que `direcao` em veredito_custo_beneficio() reflita a
    direção do efeito que o teste de significância está de fato testando, não diluída pelos
    pares que não mudaram.

    Recalcula o pivot em vez de reaproveitar wilcoxon_por_componente() porque ela não expõe os
    `diffs` intermediários -- duplica a montagem do pivot (mesmo padrão de
    comparacao_cenarios()/migracao_quartil_por_tool()), não a lógica do teste em si.
    """
    pivot = long_df.pivot_table(index=["juiz", "componente", "tool_uid"], columns="cenario", values="nota")
    rows = []
    for (juiz, componente), group in pivot.groupby(level=["juiz", "componente"]):
        if {"description_only", "with_source"} <= set(group.columns):
            paired = group.dropna(subset=["description_only", "with_source"])
        else:
            paired = group.iloc[0:0]
        diffs_efetivas = (paired["with_source"] - paired["description_only"]) if len(paired) else pd.Series(dtype=float)
        diffs_efetivas = diffs_efetivas[diffs_efetivas != 0]
        rows.append(
            {
                "juiz": juiz,
                "componente": componente,
                "mediana_diferenca_efetiva": round(diffs_efetivas.median(), 2) if len(diffs_efetivas) else None,
            }
        )
    return pd.DataFrame(rows)


# Limiar usado por veredito_custo_beneficio() para separar mudanças de nota bem fundamentadas
# de mudanças sem justificativa clara: abaixo de 50% de divergências no motivo de fallback
# (MOTIVO_FALLBACK, "sem_justificativa_especifica"), o efeito é considerado confiável o
# bastante para justificar o custo extra de mandar o código.
LIMIAR_SEM_MOTIVO_PERCENTUAL = 50.0


def veredito_custo_beneficio(records: list[dict]) -> pd.DataFrame:
    """Veredito por (juiz, componente): vale a pena pagar o custo extra de mandar SOURCE_CODE
    ao juiz? Cruza significância estatística (wilcoxon_por_componente), direção e volume da
    migração de quartil (migracao_quartil_por_componente), a proporção de divergências sem
    justificativa específica (resumo_motivos_por_componente, ver classificar_motivos()) e o
    custo extra do with_source (_delta_custo_com_codigo()).

    Filtra por registros_versao_ativa() internamente -- não confia em quem chama já ter
    filtrado, porque misturar prompt_version misturaria respostas a textos de rubrica
    diferentes (ver docstring de registros_versao_ativa()) e quebraria a comparabilidade dos
    p-valores entre componentes.

    O custo extra é por juiz, não por componente (uma chamada ao juiz avalia os 6 componentes de
    uma vez) -- por isso o mesmo delta_input_tokens/delta_latencia_ms se repete em todas as
    linhas de componente daquele juiz; não é um erro de junção.

    `pct_sem_motivo` usa `n_diverge` de migracao_quartil_por_componente() como denominador -- não a
    soma de `ocorrencias` de resumo_motivos_por_componente(), que infla o total ao contar uma
    mesma divergência uma vez por motivo casado quando ela bate em mais de uma categoria.

    IMPORTANTE: esta função não assume qualquer direção como "certa" ou "errada". O texto ativo
    de "Handling SOURCE_CODE" (RUBRIC_SYSTEM_PROMPT, prompt_version vigente) só diz que a nota
    "pode ser ajustada" a partir de inconsistências encontradas -- não restringe a direção desse
    ajuste. (Um changelog de uma versão anterior do prompt, não mais em vigor, chegou a
    restringir a direção; como essa versão não é a que está ativa hoje, ela não entra em
    nenhuma conta aqui -- ver registros_versao_ativa().) Por isso `direcao` usa rótulos
    puramente descritivos ("sobe"/"desce"), sem juízo de valor sobre qual é a legítima.

    `direcao`: "sobe" se a mediana das diferenças pareadas NÃO-NULAS (ver
    _mediana_diferenca_efetiva_por_componente() -- exclui empates, a mesma base que o teste de
    Wilcoxon usa) for positiva e significativa (p_valor_bh < 0.05), "desce" se negativa e
    significativa, "neutro" caso contrário.
    `vale_a_pena`: "sim" quando houver um efeito significativo (direção "sobe" ou "desce") E
    menos de LIMIAR_SEM_MOTIVO_PERCENTUAL das divergências daquele juiz/componente ficarem sem
    justificativa específica -- um efeito real e majoritariamente explicado pelo juiz. "não"
    quando houver efeito significativo mas a maioria das divergências não tiver justificativa
    específica (efeito real, mas mal fundamentado). "inconclusivo" quando não houver efeito
    significativo (direção "neutro").
    """
    scoped = registros_versao_ativa(records)
    long_df = scores_long(scoped)

    wilcoxon_df = wilcoxon_por_componente(long_df)
    migracao_df = migracao_quartil_por_componente(long_df)
    mediana_df = _mediana_diferenca_efetiva_por_componente(long_df)
    motivos_df = resumo_motivos_por_componente(motivos_por_divergencia(scoped))
    custo_df = custo_latencia_por_juiz_e_cenario(scoped)
    delta_df = _delta_custo_com_codigo(custo_df).set_index("juiz")

    sem_motivo_por_chave = (
        motivos_df[motivos_df["motivo"] == MOTIVO_FALLBACK].set_index(["juiz", "componente"])["ocorrencias"]
        if not motivos_df.empty
        else pd.Series(dtype=float)
    )

    base = wilcoxon_df.merge(migracao_df, on=["juiz", "componente"]).merge(mediana_df, on=["juiz", "componente"])

    rows = []
    for row in base.to_dict("records"):
        n_sem_motivo = int(sem_motivo_por_chave.get((row["juiz"], row["componente"]), 0))
        n_total_diverg = int(row["n_diverge"])
        pct_sem_motivo = round(n_sem_motivo / n_total_diverg * 100, 1) if n_total_diverg else None

        significativo = bool(row.get("significativo_bh_0.05"))
        mediana = row.get("mediana_diferenca_efetiva")
        if significativo and mediana is not None and mediana > 0:
            direcao = "sobe"
        elif significativo and mediana is not None and mediana < 0:
            direcao = "desce"
        else:
            direcao = "neutro"

        if direcao == "neutro":
            vale_a_pena = "inconclusivo"
        elif pct_sem_motivo is None or pct_sem_motivo < LIMIAR_SEM_MOTIVO_PERCENTUAL:
            vale_a_pena = "sim"
        else:
            vale_a_pena = "não"

        delta = delta_df.loc[row["juiz"]] if row["juiz"] in delta_df.index else None
        rows.append(
            {
                "juiz": row["juiz"],
                "componente": row["componente"],
                "mediana_diferenca_efetiva": mediana,
                "significativo_bh_0.05": significativo,
                "n_diverge_subiu": row["n_diverge_subiu"],
                "n_diverge_desceu": row["n_diverge_desceu"],
                "pct_sem_motivo": pct_sem_motivo,
                "delta_input_tokens": float(delta["delta_input_tokens"]) if delta is not None else None,
                "delta_latencia_ms": float(delta["delta_latencia_ms"]) if delta is not None else None,
                "custo_percentual_extra": float(delta["custo_percentual_extra"]) if delta is not None else None,
                "direcao": direcao,
                "vale_a_pena": vale_a_pena,
            }
        )

    return pd.DataFrame(rows).sort_values(["componente", "juiz"]).reset_index(drop=True)


def notas_por_linguagem(long_df: pd.DataFrame) -> pd.DataFrame:
    """Nota média (todos os componentes juntos) por linguagem do repositório -- só
    descritivo; não controla por juiz/cenário nem tenta atribuir causalidade."""
    grouped = long_df.groupby("linguagem")["nota"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["linguagem", "nota_media", "n"]
    grouped["nota_media"] = grouped["nota_media"].round(2)
    return grouped.sort_values("nota_media", ascending=False).reset_index(drop=True)


# --- Gráficos ----------------------------------------------------------------


def generate_charts(tables: dict[str, pd.DataFrame], charts_dir: Path) -> None:
    charts_dir.mkdir(parents=True, exist_ok=True)

    status = tables["status_por_juiz_cenario"]
    if not status.empty:
        by_juiz = status.groupby("juiz")[["ok", "total"]].sum().reset_index()
        by_juiz["taxa_ok_percentual"] = (by_juiz["ok"] / by_juiz["total"] * 100).round(1)
        _bar_chart(
            by_juiz, "juiz", "taxa_ok_percentual",
            "Taxa de avaliações concluídas (status ok) por juiz\n(agregado entre os 2 cenários)",
            "Juiz", "% ok",
            charts_dir / "01_taxa_ok_por_juiz.png",
        )

    componente_df = tables["notas_por_componente"]
    if not componente_df.empty:
        nota_por_componente = componente_df.groupby("componente")["media"].mean().round(2).reset_index()
        nota_por_componente.columns = ["componente", "nota_media"]
        _bar_chart(
            nota_por_componente, "componente", "nota_media",
            "Nota média por componente da rubrica\n(escala Likert 1-5; média entre juízes e cenários)",
            "Componente", "Nota média",
            charts_dir / "02_nota_media_por_componente.png",
        )

    comparacao = tables["comparacao_cenarios"]
    if not comparacao.empty:
        by_componente = (
            comparacao.groupby("componente")[["media_description_only", "media_with_source"]].mean().round(2).reset_index()
        )
        _grouped_bar_chart(
            by_componente, "componente",
            ["media_description_only", "media_with_source"],
            ["Só descrição", "Com código-fonte"],
            "Nota média por componente: descrição vs. com código-fonte\n(média entre juízes)",
            "Componente", "Nota média",
            charts_dir / "03_comparacao_cenarios_por_componente.png",
        )

    concordancia = tables["concordancia_entre_juizes"]
    if not concordancia.empty:
        pares = concordancia.copy()
        pares["par"] = pares["juiz_a"] + " x " + pares["juiz_b"]
        _bar_chart(
            pares, "par", "correlacao_pearson",
            "Concordância entre pares de juízes\n(correlação de Pearson, mesmo tool/cenário/componente)",
            "Par de juízes", "Correlação de Pearson",
            charts_dir / "04_concordancia_entre_juizes.png",
        )

    notas_lang = tables["notas_por_linguagem"]
    if not notas_lang.empty:
        _bar_chart(
            notas_lang, "linguagem", "nota_media",
            "Nota média (todos os componentes) por linguagem do repositório", "Linguagem", "Nota média",
            charts_dir / "05_notas_por_linguagem.png", horizontal=True,
        )

    custo = tables["custo_latencia_por_juiz"]
    if not custo.empty:
        _bar_chart(
            custo, "juiz", "media_latencia_ms",
            "Latência média por juiz\n(ms, só avaliações concluídas)", "Juiz", "Latência média (ms)",
            charts_dir / "06_latencia_por_juiz.png",
        )

    wilcoxon_df = tables["wilcoxon_por_componente"]
    if not wilcoxon_df.empty:
        _wilcoxon_pvalue_chart(wilcoxon_df, charts_dir / "07_wilcoxon_p_valor_bh.png")

    logger.info("Gráficos salvos em %s", charts_dir)


def _wilcoxon_pvalue_chart(wilcoxon_df: pd.DataFrame, path: Path) -> None:
    """Não usa _grouped_bar_chart (helper compartilhado com analysis_report.py) porque
    precisa da linha de referência em alpha=0.05, que esse helper não expõe -- gráfico
    dedicado em vez de mudar um helper usado também pelas Etapas 1-2."""
    wide = wilcoxon_df.pivot_table(index="componente", columns="juiz", values="p_valor_bh")
    juizes = list(wide.columns)

    fig, ax = plt.subplots(figsize=CHART_STYLE["figsize"])
    x = range(len(wide))
    width = 0.8 / max(len(juizes), 1)
    colors = ["#3b6ea5", "#e07b39", "#4a9e6f"]
    for i, juiz in enumerate(juizes):
        offset = (i - (len(juizes) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], wide[juiz], width=width, label=juiz, color=colors[i % len(colors)])
    ax.axhline(0.05, color="#c0392b", linestyle="--", linewidth=1.2, label="α = 0,05")
    ax.set_xticks(list(x))
    ax.set_xticklabels(wide.index, rotation=35, ha="right")
    ax.set_xlabel("Componente da rubrica")
    ax.set_ylabel("p-valor (corrigido, Benjamini-Hochberg)")
    ax.set_title("Teste de Wilcoxon: p-valor corrigido por componente e juiz\n(abaixo da linha = diferença estatisticamente significativa)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=CHART_STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)


# --- CLI -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos e planilhas de validação da Etapa 3 a partir de data/evaluations/*.jsonl")
    parser.add_argument("--evaluations-dir", type=Path, default=None, help="Diretório com {judge_id}.jsonl (default: data/evaluations).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Diretório de saída (default: data/analysis).")
    args = parser.parse_args()

    evaluations_dir = args.evaluations_dir or (DATA_DIR / "evaluations")
    output_dir = args.output_dir or (DATA_DIR / "analysis")

    if not evaluations_dir.exists() or not any(evaluations_dir.glob("*.jsonl")):
        logger.error(
            "Nenhum {judge_id}.jsonl encontrado em %s -- rode a Etapa 3 "
            "(pipeline/run_step3.py) primeiro.",
            evaluations_dir,
        )
        sys.exit(1)

    records = load_evaluations(evaluations_dir)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    logger.info(
        "Carregadas %s avaliações (tool x cenário x juiz) de %s, %s com status ok",
        len(records), evaluations_dir, n_ok,
    )

    long_df = scores_long(records)

    # Tabelas de motivo/veredito ficam restritas à versão vigente do prompt (ver
    # registros_versao_ativa()): as tabelas legadas acima continuam com todos os records, sem
    # esse filtro, para não mudar números já publicados no TCC.
    scoped = registros_versao_ativa(records)

    tables = {
        "status_por_juiz_cenario": status_por_juiz_cenario(records),
        "notas_por_componente": notas_por_componente(long_df),
        "comparacao_cenarios": comparacao_cenarios(long_df),
        "wilcoxon_por_componente": wilcoxon_por_componente(long_df),
        "migracao_quartil_por_componente": migracao_quartil_por_componente(long_df),
        "concordancia_entre_juizes": concordancia_entre_juizes(long_df),
        "custo_latencia_por_juiz": custo_latencia_por_juiz(records),
        "notas_por_linguagem": notas_por_linguagem(long_df),
        "motivos_por_divergencia": motivos_por_divergencia(scoped),
        "resumo_motivos_por_componente": resumo_motivos_por_componente(motivos_por_divergencia(scoped)),
        "custo_latencia_por_juiz_e_cenario": custo_latencia_por_juiz_e_cenario(scoped),
        "veredito_custo_beneficio": veredito_custo_beneficio(scoped),
    }

    export_tables(tables, output_dir / "tables", workbook_name="resumo_etapa_3.xlsx")
    generate_charts(tables, output_dir / "charts")

    logger.info("Concluído: %s avaliações carregadas (%s ok), tabelas geradas: %s.", len(records), n_ok, ", ".join(tables.keys()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Geração do relatório de avaliação (Etapa 3) falhou")
        sys.exit(1)
