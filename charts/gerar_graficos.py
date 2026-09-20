"""
Análise RQ2 — Quais atributos da rubrica são mais afetados pela adição do
código-fonte na avaliação da qualidade documental das tools?

Entrada esperada (JSON): lista de tools, cada uma com uma lista "evaluations",
cada evaluation com "model", "with_source" e "description_only", e dentro de
cada um desses, os atributos da rubrica (score + reasoning).

Estrutura esperada de cada evaluation:
{
  "model": "nome-do-modelo",
  "with_source": {
      "purpose": {"score": int, "reasoning": str},
      "guidelines": {...}, "limitations": {...},
      "parameter_explanation": {...}, "length_completeness": {...},
      "examples": {...}
  },
  "description_only": { ... mesma estrutura ... }
}

Como há mais de um modelo de IA avaliando cada tool, todos os gráficos usam
a MÉDIA entre os modelos (um único gráfico, não um por modelo). O boxplot
compara "Com código" vs "Sem código" — não compara modelos entre si.

Gera 3 figuras em /mnt/user-data/outputs/:
  1. barras_divergentes.png       — diferença média por atributo
  2. boxplot_com_sem_codigo.png   — distribuição das notas, com vs sem código
  3. quantidade_magnitude.png     — % de casos que mudaram + magnitude média

E salva um CSV "long format" (nível tool x modelo x atributo) para reuso.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO
# ---------------------------------------------------------------------------
INPUT_JSON = "file.json"   # ajuste para o caminho real do seu arquivo
OUTPUT_DIR = "outputs"

ATTRIBUTES = [
    "purpose",
    "guidelines",
    "limitations",
    "parameter_explanation",
    "length_completeness",
    "examples",
]

# Nomes mais legíveis para os gráficos (edite à vontade)
ATTR_LABELS = {
    "purpose": "Propósito",
    "guidelines": "Diretrizes de uso",
    "limitations": "Limitações",
    "parameter_explanation": "Explicação de parâmetros",
    "length_completeness": "Completude/extensão",
    "examples": "Exemplos",
}

COR_PIOROU = "#d62728"     # vermelho padrão matplotlib
COR_MELHOROU = "#1f77b4"   # azul padrão matplotlib (mesmo tom da imagem de referência)

# Estilo simples: fundo branco, só grade horizontal tracejada, sem grade vertical
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


# ---------------------------------------------------------------------------
# 1. CARREGAR DADOS (nível tool x modelo x atributo)
# ---------------------------------------------------------------------------
def carregar_dados(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for tool in data:
        tool_id = tool.get("tool_id")
        for ev in tool.get("evaluations", []):
            model = ev.get("model")
            with_src = ev.get("with_source", {})
            without_src = ev.get("description_only", {})

            for attr in ATTRIBUTES:
                score_com = (with_src or {}).get(attr, {}).get("score")
                score_sem = (without_src or {}).get(attr, {}).get("score")

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

    return pd.DataFrame(rows)


def agregar_entre_modelos(df):
    """
    Colapsa os modelos: para cada (tool_id, atributo), tira a média das
    notas 'com' e 'sem' código entre os modelos. Todos os gráficos usam
    esta tabela agregada — um único valor por tool/atributo, não um por
    modelo.
    """
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


# ---------------------------------------------------------------------------
# 2. GRÁFICO 1 — BARRAS DIVERGENTES (DIFERENÇA MÉDIA POR ATRIBUTO)
#    Valor da diferença escrito diretamente na ponta de cada barra.
# ---------------------------------------------------------------------------
def grafico_barras_divergentes(agg, output_path):
    media = agg.groupby("attribute_label")["diff"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    cores = [COR_PIOROU if v < 0 else COR_MELHOROU for v in media.values]
    barras = ax.barh(media.index, media.values, color=cores)

    limite = max(abs(media.min()), abs(media.max()))
    ax.set_xlim(-limite * 1.6 - 0.05, limite * 1.3 + 0.05)

    for barra, valor in zip(barras, media.values):
        offset = limite * 0.04 * (1 if valor >= 0 else -1)
        ha = "left" if valor >= 0 else "right"
        ax.text(
            valor + offset, barra.get_y() + barra.get_height() / 2,
            f"{valor:.2f}", va="center", ha=ha, fontsize=11,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Diferença média (com − sem código)")
    _sem_grade_vertical(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico 1 salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 3. GRÁFICO 2 — BOXPLOT (DESIGN PADRÃO) COMPARANDO COM x SEM CÓDIGO
#    Por atributo, duas caixas lado a lado: notas com código / sem código.
# ---------------------------------------------------------------------------
def _plot_boxplot_com_sem(longo, ordem, ax):
    largura = 0.35
    x = np.arange(len(ordem))
    fontes = ["Sem código", "Com código"]

    for i, fonte in enumerate(fontes):
        dados = [
            longo[(longo["attribute_label"] == attr) & (longo["fonte"] == fonte)]["score"].values
            for attr in ordem
        ]
        pos = x + (i - 0.5) * largura
        ax.boxplot(
            dados,
            positions=pos,
            widths=largura * 0.9,
            patch_artist=True,
            boxprops=dict(facecolor=("#dddddd" if fonte == "Sem código" else "#1f77b4"), alpha=0.7),
            medianprops=dict(color="black"),
            flierprops=dict(marker="o", markersize=3, alpha=0.4),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ordem, rotation=15, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Nota (escala Likert)")
    _sem_grade_vertical(ax)

    from matplotlib.patches import Patch
    legenda = [
        Patch(facecolor="#dddddd", alpha=0.7, label="Sem código"),
        Patch(facecolor="#1f77b4", alpha=0.7, label="Com código"),
    ]
    ax.legend(handles=legenda, loc="upper right")


def grafico_boxplot_com_sem(agg, output_path):
    ordem = agg.groupby("attribute_label")["diff"].mean().sort_values().index.tolist()

    longo = agg.melt(
        id_vars=["attribute_label"],
        value_vars=["score_sem_codigo", "score_com_codigo"],
        var_name="fonte",
        value_name="score",
    )
    longo["fonte"] = longo["fonte"].map(
        {"score_sem_codigo": "Sem código", "score_com_codigo": "Com código"}
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_boxplot_com_sem(longo, ordem, ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico 2 salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 3b. GRÁFICO EXTRA — MESMO BOXPLOT, MAS SÓ COM TOOLS QUE MUDARAM (diff != 0)
#     NAQUELE atributo. Remove os empates, que colapsam a caixa em casos onde
#     a maioria das notas nunca muda (ex.: nota sempre 1).
# ---------------------------------------------------------------------------
def grafico_boxplot_com_sem_apenas_mudou(agg, output_path):
    ordem = agg.groupby("attribute_label")["diff"].mean().sort_values().index.tolist()

    agg_mudou = agg[agg["diff"] != 0].copy()

    # % de tools usadas em cada caixa (as que mudaram), pra dar contexto no eixo
    n_total = agg.groupby("attribute_label").size()
    n_mudou = agg_mudou.groupby("attribute_label").size().reindex(ordem, fill_value=0)
    pct_mudou = (n_mudou / n_total.reindex(ordem) * 100).round(0)
    ordem_rotulada = [f"{a}\n(n={int(n_mudou[a])}, {pct_mudou[a]:.0f}%)" for a in ordem]

    longo = agg_mudou.melt(
        id_vars=["attribute_label"],
        value_vars=["score_sem_codigo", "score_com_codigo"],
        var_name="fonte",
        value_name="score",
    )
    longo["fonte"] = longo["fonte"].map(
        {"score_sem_codigo": "Sem código", "score_com_codigo": "Com código"}
    )
    mapa_rotulo = dict(zip(ordem, ordem_rotulada))
    longo["attribute_label"] = longo["attribute_label"].map(mapa_rotulo)

    fig, ax = plt.subplots(figsize=(12, 7))
    _plot_boxplot_com_sem(longo, ordem_rotulada, ax)
    ax.set_xlabel("(n = quantidade de tools que mudaram naquele atributo, % do total)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico 2b (apenas mudanças) salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 4. GRÁFICO 3 — % DE CASOS QUE MUDARAM + MAGNITUDE MÉDIA
#    Valores escritos diretamente nas barras (nenhum texto descritivo).
# ---------------------------------------------------------------------------
def grafico_quantidade_magnitude(agg, output_path):
    grupos = agg.groupby("attribute_label")

    pct_piorou = grupos["diff"].apply(lambda s: (s < 0).mean() * 100)
    pct_melhorou = grupos["diff"].apply(lambda s: (s > 0).mean() * 100)

    total = (pct_piorou + pct_melhorou).sort_values()
    ordem = total.index.tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(ordem))

    barras_piorou = ax.barh(y, pct_piorou.reindex(ordem), color=COR_PIOROU, label="Piorou")
    barras_melhorou = ax.barh(
        y, pct_melhorou.reindex(ordem), left=pct_piorou.reindex(ordem),
        color=COR_MELHOROU, label="Melhorou",
    )

    # valor (%) escrito dentro de cada segmento da barra
    for barra, valores in [(barras_piorou, pct_piorou.reindex(ordem)),
                            (barras_melhorou, pct_melhorou.reindex(ordem))]:
        for rect, valor in zip(barra, valores):
            if valor <= 0:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                f"{valor:.0f}%", va="center", ha="center", color="white", fontsize=10,
            )

    # total (%) escrito na ponta da barra, sem nenhum texto adicional
    for yi, attr in zip(y, ordem):
        ax.text(total[attr] + 1.5, yi, f"{total[attr]:.0f}%", va="center", ha="left", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(ordem)
    ax.set_xlabel("% de casos que mudaram")
    ax.set_xlim(0, max(total.max() * 1.25, 15))
    ax.legend(loc="lower right")
    _sem_grade_vertical(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico 3 salvo em: {output_path}")


# ---------------------------------------------------------------------------
# GRÁFICO EXTRA — STACKED: DIMINUIU / EMPATOU / AUMENTOU
#    Uma barra por atributo, dividida em 3 segmentos, todas começando em 0%
#    e somando 100%. Valor (%) escrito dentro de cada segmento.
# ---------------------------------------------------------------------------
COR_DIMINUIU = "#d62728"   # vermelho padrão
COR_EMPATOU = "#bbbbbb"    # cinza claro
COR_AUMENTOU = "#1f77b4"   # azul padrão

ORDEM_ATRIBUTOS_FIXA = [
    "Propósito",
    "Explicação de parâmetros",
    "Completude/extensão",
    "Diretrizes de uso",
    "Limitações",
    "Exemplos",
]

def grafico_diminuiu_empatou_aumentou(agg, output_path):
    # ordem fixa solicitada (de cima para baixo no gráfico); como barh empilha
    # de baixo pra cima, invertemos a lista antes de plotar
    ordem = ORDEM_ATRIBUTOS_FIXA[::-1]

    pct = pd.DataFrame(index=ordem, columns=["Diminuiu", "Empatou", "Aumentou"], dtype=float)
    for attr in ordem:
        s = agg.loc[agg["attribute_label"] == attr, "diff"]
        n = len(s)
        pct.loc[attr, "Diminuiu"] = (s < 0).sum() / n * 100
        pct.loc[attr, "Empatou"] = (s == 0).sum() / n * 100
        pct.loc[attr, "Aumentou"] = (s > 0).sum() / n * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(ordem))

    esquerda_acumulado = np.zeros(len(ordem))
    segmentos = [
        ("Diminuiu", COR_DIMINUIU),
        ("Aumentou", COR_AUMENTOU),
        ("Empatou", COR_EMPATOU),
    ]
    for cat, cor in segmentos:
        valores = pct[cat].values
        barras = ax.barh(y, valores, left=esquerda_acumulado, color=cor, label=cat)
        for rect, valor in zip(barras, valores):
            if valor <= 0:
                continue
            cor_texto = "white" if cat != "Empatou" else "black"
            ax.text(
                rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                f"{valor:.0f}%", va="center", ha="center", color=cor_texto, fontsize=10,
            )
        esquerda_acumulado = esquerda_acumulado + valores

    ax.set_yticks(y)
    ax.set_yticklabels(ordem)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de casos")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    _sem_grade_vertical(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico diminuiu/empatou/aumentou salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 5. TESTE DE CONCORDÂNCIA ENTRE MODELOS (só informativo no terminal)
# ---------------------------------------------------------------------------
def checar_concordancia(df):
    tabela = df.groupby(["model", "attribute_label"])["diff"].mean().unstack(0)
    print("\n=== Diferença média por atributo e modelo (antes de agregar) ===")
    print(tabela.round(2))
    print("\n=== Correlação entre modelos ===")
    print(tabela.corr(method="spearman").round(2))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = carregar_dados(INPUT_JSON)
    print(f"Total de linhas (tool x modelo x atributo): {len(df)}")

    csv_path = f"{OUTPUT_DIR}/dados_long_format.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV long-format salvo em: {csv_path}")

    checar_concordancia(df)

    agg = agregar_entre_modelos(df)
    agg.to_csv(f"{OUTPUT_DIR}/dados_agregados_entre_modelos.csv", index=False)

    grafico_barras_divergentes(agg, f"{OUTPUT_DIR}/barras_divergentes.png")
    grafico_boxplot_com_sem(agg, f"{OUTPUT_DIR}/boxplot_com_sem_codigo.png")
    grafico_boxplot_com_sem_apenas_mudou(agg, f"{OUTPUT_DIR}/boxplot_com_sem_codigo_apenas_mudou.png")
    grafico_quantidade_magnitude(agg, f"{OUTPUT_DIR}/quantidade_magnitude.png")
    grafico_diminuiu_empatou_aumentou(agg, f"{OUTPUT_DIR}/diminuiu_empatou_aumentou.png")


if __name__ == "__main__":
    main()