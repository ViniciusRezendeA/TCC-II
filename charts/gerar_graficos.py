"""
Análise RQ2 — Quais atributos da rubrica são mais afetados pela adição do
código na avaliação da qualidade documental das tools?

Entrada: lê diretamente data/evaluations/{judge_id}.jsonl (saída da Etapa 3),
com o mesmo carregamento, deduplicação, filtro de versão de prompt e chave de
pareamento de scripts/analysis_evaluation_report.py (load_evaluations,
dedupe_records, registros_versao_ativa, tool_key_for).

Só entram na análise as tools avaliadas pelos 3 modelos, nos dois cenários
(description_only e with_source).

Ordem dos atributos (SEMPRE, em todos os gráficos e tabelas):
  Purpose, Parameter Explanation, Length & Completeness,
  Guidelines, Limitations, Examples

Gera EXATAMENTE estes 8 arquivos em charts/outputs/:
  diminuiu_empatou_aumentou_por_modelo.png   (3 painéis, 1 por modelo)
  diminuiu_empatou_aumentou_media.png        (1 painel, média dos 3 modelos)
  boxplot_com_sem_codigo_apenas_mudou_por_modelo.png
  boxplot_com_sem_codigo_apenas_mudou_media.png
  tabela_wilcoxon_por_modelo.tex             (1 tabela por modelo)
  tabela_wilcoxon_media.tex
  tabela_transicao_quartis_por_modelo.tex    (1 tabela por modelo)
  tabela_transicao_quartis_media.tex

Rodar da raiz do projeto:
    uv run python charts/gerar_graficos.py
"""

import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from scripts.analysis_evaluation_report import (  # noqa: E402
    load_evaluations,
    registros_versao_ativa,
    tool_key_for,
)
from scripts.dedupe_evaluations import dedupe_records  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
EVALUATIONS_DIR = RAIZ / "data" / "evaluations"
OUTPUT_DIR = str(RAIZ / "charts" / "outputs")
N_MODELOS_ESPERADO = 3  # só tools avaliadas por TODOS os modelos

FIGSIZE = (8, 8)


NOME_MEDIA = "Média dos modelos"

ATTRIBUTES = [
    "purpose",
    "guidelines",
    "limitations",
    "parameter_explanation",
    "length_completeness",
    "examples",
]

ATTR_LABELS = {
    "purpose": "Purpose",
    "guidelines": "Guidelines",
    "limitations": "Limitations",
    "parameter_explanation": "Parameter Explanation",
    "length_completeness": "Length & Completeness",
    "examples": "Examples",
}

# ORDEM FIXA — usada em TODOS os gráficos e tabelas
ORDEM_ATRIBUTOS_FIXA = [
    "Purpose",
    "Parameter Explanation",
    "Length & Completeness",
    "Guidelines",
    "Limitations",
    "Examples",
]

COR_DIMINUIU = "#d62728"
COR_EMPATOU = "#bbbbbb"
COR_AUMENTOU = "#1f77b4"
COR_MEDIANA = "#ff7f0e"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "font.size": 12,
})


def _sem_grade_vertical(ax):
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _slug(texto):
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", texto).strip("_").lower()


def _escapar_latex(texto):
    return str(texto).replace("_", r"\_").replace("&", r"\&")


# ---------------------------------------------------------------------------
# 1. CARREGAR DADOS (nível tool x modelo x atributo)
# ---------------------------------------------------------------------------
def carregar_dados(evaluations_dir):
    records = registros_versao_ativa(dedupe_records(load_evaluations(evaluations_dir)))

    scores = {}
    for r in records:
        if r.get("status") != "ok" or not r.get("scores"):
            continue
        scores.setdefault((tool_key_for(r), r["judge"]["id"]), {})[r["scenario"]] = r["scores"]

    rows = []
    for (tool_id, model), cenarios in scores.items():
        with_src = cenarios.get("with_source")
        without_src = cenarios.get("description_only")
        if with_src is None or without_src is None:
            continue

        for attr in ATTRIBUTES:
            score_com = (with_src.get(attr) or {}).get("score")
            score_sem = (without_src.get(attr) or {}).get("score")
            if score_com is None or score_sem is None:
                continue

            rows.append(
                {
                    "tool_id": tool_id,
                    "model": model,
                    "attribute": attr,
                    "attribute_label": ATTR_LABELS.get(attr, attr),
                    "score_com_codigo": score_com,
                    "score_sem_codigo": score_sem,
                    "diff": score_com - score_sem,
                }
            )

    df = pd.DataFrame(rows)

    # Só tools avaliadas por TODOS os modelos
    n_modelos_total = df["model"].nunique()
    if n_modelos_total != N_MODELOS_ESPERADO:
        print(f"[AVISO] Encontrados {n_modelos_total} modelos nos dados "
              f"(esperado: {N_MODELOS_ESPERADO}): {sorted(df['model'].unique())}")
    n_modelos_por_tool = df.groupby("tool_id")["model"].nunique()
    tools_validas = n_modelos_por_tool[n_modelos_por_tool == n_modelos_total].index
    return df[df["tool_id"].isin(tools_validas)].reset_index(drop=True)


def agregar_entre_modelos(df):
    """Média das notas entre os modelos, por (tool, atributo)."""
    agg = (
        df.groupby(["tool_id", "attribute", "attribute_label"])
        .agg(
            score_com_codigo=("score_com_codigo", "mean"),
            score_sem_codigo=("score_sem_codigo", "mean"),
        )
        .reset_index()
    )
    agg["diff"] = agg["score_com_codigo"] - agg["score_sem_codigo"]
    return agg


def dados_de_um_modelo(df, modelo):
    cols = ["tool_id", "attribute", "attribute_label",
            "score_com_codigo", "score_sem_codigo", "diff"]
    return df.loc[df["model"] == modelo, cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# INFRAESTRUTURA DE FIGURAS
#   salvar_graficos gera <nome>_por_modelo.png (N painéis) e <nome>_media.png
#   ocultar_y_repetido=True: com N painéis, só o primeiro mostra os rótulos do eixo Y
# ---------------------------------------------------------------------------
def salvar_graficos(nome, desenhar, agg_modelos, agg_media,
                    figsize_painel=(10, 6), sharey=False, ocultar_y_repetido=False):
    modelos = list(agg_modelos)
    n = len(modelos)

    fig, axes = plt.subplots(
        1, n, figsize=(figsize_painel[0] * n, figsize_painel[1]),
        sharey=(sharey or ocultar_y_repetido), squeeze=False,
    )
    for ax, modelo in zip(axes[0], modelos):
        desenhar(agg_modelos[modelo], ax)
        ax.set_title(modelo, fontsize=13, fontweight="bold")
    if ocultar_y_repetido:
        for ax in axes[0][1:]:
            ax.tick_params(axis="y", labelleft=False)
    fig.tight_layout()
    caminho = f"{OUTPUT_DIR}/{nome}_por_modelo.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {caminho}")

    fig, ax = plt.subplots(figsize=figsize_painel)
    desenhar(agg_media, ax)
    ax.set_title(NOME_MEDIA, fontsize=13, fontweight="bold")
    fig.tight_layout()
    caminho = f"{OUTPUT_DIR}/{nome}_media.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {caminho}")


# ---------------------------------------------------------------------------
# GRÁFICO — DIMINUIU / EMPATOU / AUMENTOU (barras empilhadas, soma 100%)
# ---------------------------------------------------------------------------
def _desenhar_diminuiu_empatou_aumentou(agg, ax):
    # barh desenha de baixo pra cima: invertemos para a ordem fixa
    # ficar de cima pra baixo (Purpose no topo, Examples embaixo)
    ordem = ORDEM_ATRIBUTOS_FIXA[::-1]

    pct = pd.DataFrame(0.0, index=ordem, columns=["Diminuiu", "Empatou", "Aumentou"])
    for attr in ordem:
        s = agg.loc[agg["attribute_label"] == attr, "diff"]
        n = len(s)
        if n == 0:
            continue
        pct.loc[attr, "Diminuiu"] = (s < 0).sum() / n * 100
        pct.loc[attr, "Empatou"] = (s == 0).sum() / n * 100
        pct.loc[attr, "Aumentou"] = (s > 0).sum() / n * 100

    y = np.arange(len(ordem))
    esquerda = np.zeros(len(ordem))
    for cat, cor in [("Diminuiu", COR_DIMINUIU), ("Aumentou", COR_AUMENTOU), ("Empatou", COR_EMPATOU)]:
        valores = pct[cat].values
        barras = ax.barh(y, valores, left=esquerda, color=cor, label=cat)
        for rect, valor in zip(barras, valores):
            if valor <= 0:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                f"{valor:.0f}%", va="center", ha="center", fontsize=10,
                color="black" if cat == "Empatou" else "white",
            )
        esquerda = esquerda + valores

    ax.set_yticks(y)
    ax.set_yticklabels(ordem)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de casos")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    _sem_grade_vertical(ax)


def grafico_diminuiu_empatou_aumentou(agg_modelos, agg_media):
    salvar_graficos("diminuiu_empatou_aumentou", _desenhar_diminuiu_empatou_aumentou,
                    agg_modelos, agg_media, figsize_painel=FIGSIZE,
                    ocultar_y_repetido=True)


# ---------------------------------------------------------------------------
# GRÁFICO — BOXPLOT COM x SEM CÓDIGO, SÓ TOOLS QUE MUDARAM (diff != 0)
# ---------------------------------------------------------------------------
def _plot_boxplot_com_sem(longo, ordem, ax):
    largura = 0.35
    x = np.arange(len(ordem))

    for i, fonte in enumerate(["Sem código", "Com código"]):
        dados = [
            longo[(longo["attribute_label"] == attr) & (longo["fonte"] == fonte)]["score"].values
            for attr in ordem
        ]
        ax.boxplot(
            dados,
            positions=x + (i - 0.5) * largura,
            widths=largura * 0.9,
            patch_artist=True,
            showmeans=True,
            boxprops=dict(facecolor=("#dddddd" if fonte == "Sem código" else "#1f77b4"), alpha=0.7),
            medianprops=dict(color=COR_MEDIANA, linewidth=2.5),
            meanprops=dict(marker="D", markerfacecolor="white",
                           markeredgecolor="black", markersize=4),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ordem, rotation=15, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Nota (escala Likert)")
    _sem_grade_vertical(ax)

    ax.legend(
        handles=[
            Patch(facecolor="#dddddd", alpha=0.7, label="Sem código"),
            Patch(facecolor="#1f77b4", alpha=0.7, label="Com código"),
            Line2D([0], [0], color=COR_MEDIANA, linewidth=2.5, label="Mediana"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=5, label="Média"),
        ],
        loc="upper right",
    )


def _formato_longo(agg):
    longo = agg.melt(
        id_vars=["attribute_label"],
        value_vars=["score_sem_codigo", "score_com_codigo"],
        var_name="fonte",
        value_name="score",
    )
    longo["fonte"] = longo["fonte"].map(
        {"score_sem_codigo": "Sem código", "score_com_codigo": "Com código"}
    )
    return longo


def _desenhar_boxplot_apenas_mudou(agg, ax):
    ordem = ORDEM_ATRIBUTOS_FIXA  # ordem fixa, esquerda -> direita
    agg_mudou = agg[agg["diff"] != 0].copy()

    n_total = agg.groupby("attribute_label").size().reindex(ordem, fill_value=0)
    n_mudou = agg_mudou.groupby("attribute_label").size().reindex(ordem, fill_value=0)
    pct_mudou = (n_mudou / n_total.replace(0, np.nan) * 100).fillna(0).round(0)
    ordem_rotulada = [f"{a}\n(n={int(n_mudou[a])}, {pct_mudou[a]:.0f}%)" for a in ordem]

    longo = _formato_longo(agg_mudou)
    longo["attribute_label"] = longo["attribute_label"].map(dict(zip(ordem, ordem_rotulada)))

    _plot_boxplot_com_sem(longo, ordem_rotulada, ax)
    ax.set_xlabel("(n = tools que mudaram naquele atributo, % do total)")


def grafico_boxplot_apenas_mudou(agg_modelos, agg_media):
    salvar_graficos("boxplot_com_sem_codigo_apenas_mudou", _desenhar_boxplot_apenas_mudou,
                    agg_modelos, agg_media, figsize_painel=FIGSIZE, sharey=True)


# ---------------------------------------------------------------------------
# TABELAS LaTeX — helpers
# ---------------------------------------------------------------------------
def _formatar_numero_br(valor, casas=3):
    return f"{valor:.{casas}f}".replace(".", ",")


def _formatar_cientifico_latex(valor, casas=2):
    if valor == 0:
        return "$< 10^{-300}$"
    mantissa, expoente = f"{valor:.{casas}e}".split("e")
    return f"${mantissa.replace('.', ',')} \\times 10^{{{int(expoente)}}}$"


def _estrelas_significancia(p):
    if pd.isna(p):
        return ""
    if p < 0.05:
        return "*"
    return ""


def _salvar_texto(caminho, blocos):
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(blocos))
    print(f"[OK] {caminho}")


# ---------------------------------------------------------------------------
# TABELA — WILCOXON (postos sinalizados), POR ATRIBUTO
# ---------------------------------------------------------------------------
def teste_wilcoxon_por_atributo(agg):
    resultados = []
    for attr in ORDEM_ATRIBUTOS_FIXA:
        grupo = agg[agg["attribute_label"] == attr]
        sem = grupo["score_sem_codigo"].values
        com = grupo["score_com_codigo"].values

        n_diferentes = int(np.sum(com != sem))
        if n_diferentes == 0:
            estatistica, p_valor = np.nan, np.nan
        else:
            try:
                estatistica, p_valor = wilcoxon(com, sem, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                estatistica, p_valor = np.nan, np.nan

        resultados.append({
            "attribute_label": attr,
            "n_diferentes": n_diferentes,
            "estatistica_W": estatistica,
            "p_valor": p_valor,
        })
    return pd.DataFrame(resultados)


def montar_tabela_latex_wilcoxon(df_resultados, nome_escopo, label):
    linhas = []
    for _, row in df_resultados.iterrows():
        p_valor = row["p_valor"]
        if pd.isna(p_valor):
            p_fmt = "--"
        else:
            p_fmt = (
                _formatar_cientifico_latex(p_valor)
                if p_valor < 0.001
                else _formatar_numero_br(p_valor, 3)
            )
            p_fmt += _estrelas_significancia(p_valor)
            if p_valor >= 0.05:
                p_fmt = "--"

        w_fmt = (
            "--"
            if pd.isna(row["estatistica_W"])
            else _formatar_numero_br(row["estatistica_W"], 1)
        )
        linhas.append(
            f"{_escapar_latex(row['attribute_label'])} & {row['n_diferentes']} & {w_fmt} & {p_fmt} \\\\"
        )

    corpo = "\n".join(linhas)
    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Resultados do Teste de Postos Sinalizados de Wilcoxon comparando as "
        f"notas com e sem código, por atributo ({_escapar_latex(nome_escopo)}; pareado por tool).}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Atributo & $n$ diferentes & $W$ & $p$-valor \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\\\[4pt]\n"
        "{\\footnotesize * $p<0.05$}\n"
        "\\end{table}\n"
    )


def gerar_tabelas_wilcoxon(agg_modelos, agg_media):
    blocos_modelos = [
        montar_tabela_latex_wilcoxon(teste_wilcoxon_por_atributo(a), m, f"tab:wilcoxon_{_slug(m)}")
        for m, a in agg_modelos.items()
    ]
    _salvar_texto(f"{OUTPUT_DIR}/tabela_wilcoxon_por_modelo.tex", blocos_modelos)
    _salvar_texto(
        f"{OUTPUT_DIR}/tabela_wilcoxon_media.tex",
        [montar_tabela_latex_wilcoxon(teste_wilcoxon_por_atributo(agg_media),
                                      NOME_MEDIA, "tab:wilcoxon_media")],
    )


# ---------------------------------------------------------------------------
# TABELA — TRANSIÇÃO DE QUARTIS, POR ATRIBUTO
#   Faixas definidas pela distribuição "sem código" (referência).
# ---------------------------------------------------------------------------
def _classificar_quartil(valor, q1, mediana, q3):
    if valor <= q1:
        return 0
    elif valor <= mediana:
        return 1
    elif valor <= q3:
        return 2
    return 3


def calcular_transicao_quartis(agg):
    resumo = []
    for attr in ORDEM_ATRIBUTOS_FIXA:
        grupo = agg[agg["attribute_label"] == attr]
        sem = grupo["score_sem_codigo"].values
        com = grupo["score_com_codigo"].values
        n = len(sem)
        if n == 0:
            continue

        q1, mediana, q3 = np.percentile(sem, [25, 50, 75])
        idx_sem = np.array([_classificar_quartil(v, q1, mediana, q3) for v in sem])
        idx_com = np.array([_classificar_quartil(v, q1, mediana, q3) for v in com])

        resumo.append({
            "attribute_label": attr,
            "pct_manteve": np.sum(idx_com == idx_sem) / n * 100,
            "pct_subiu": np.sum(idx_com > idx_sem) / n * 100,
            "pct_desceu": np.sum(idx_com < idx_sem) / n * 100,
        })
    return pd.DataFrame(resumo)


def montar_tabela_latex_quartis(df_resumo, nome_escopo, label):
    linhas = [
        f"{_escapar_latex(r['attribute_label'])} & "
        f"{_formatar_numero_br(r['pct_manteve'], 1)}\\% & "
        f"{_formatar_numero_br(r['pct_subiu'], 1)}\\% & "
        f"{_formatar_numero_br(r['pct_desceu'], 1)}\\% \\\\"
        for _, r in df_resumo.iterrows()
    ]
    corpo = "\n".join(linhas)
    return (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Transição de quartil das notas (com vs. sem código), por atributo "
        f"({_escapar_latex(nome_escopo)}). Faixas de quartil definidas a partir da "
        "distribuição sem código (pareado por tool).}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Atributo & Manteve o quartil & Subiu de quartil & Desceu de quartil \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def gerar_tabelas_quartis(agg_modelos, agg_media):
    blocos_modelos = [
        montar_tabela_latex_quartis(calcular_transicao_quartis(a), m,
                                    f"tab:transicao_quartis_{_slug(m)}")
        for m, a in agg_modelos.items()
    ]
    _salvar_texto(f"{OUTPUT_DIR}/tabela_transicao_quartis_por_modelo.tex", blocos_modelos)
    _salvar_texto(
        f"{OUTPUT_DIR}/tabela_transicao_quartis_media.tex",
        [montar_tabela_latex_quartis(calcular_transicao_quartis(agg_media),
                                     NOME_MEDIA, "tab:transicao_quartis_media")],
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = carregar_dados(EVALUATIONS_DIR)
    print(f"Tools avaliadas por todos os modelos: {df['tool_id'].nunique()}; "
          f"modelos: {sorted(df['model'].unique())}")

    modelos = sorted(df["model"].unique())
    agg_modelos = {m: dados_de_um_modelo(df, m) for m in modelos}
    agg_media = agregar_entre_modelos(df)

    grafico_diminuiu_empatou_aumentou(agg_modelos, agg_media)
    grafico_boxplot_apenas_mudou(agg_modelos, agg_media)
    gerar_tabelas_wilcoxon(agg_modelos, agg_media)
    gerar_tabelas_quartis(agg_modelos, agg_media)


if __name__ == "__main__":
    main()