"""
Análise RQ2 — Quais atributos da rubrica são mais afetados pela adição do
código-fonte na avaliação da qualidade documental das tools?

Entrada esperada (JSON): lista de tools, cada uma com uma lista "evaluations",
cada evaluation com "model", "with_source" e "description_only", e dentro de
cada um desses, os 5 atributos da rubrica (score + reasoning).

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

Gera 3 figuras em /mnt/user-data/outputs/:
  1. barras_divergentes_por_modelo.png
  2. boxplot_strip_por_atributo_modelo.png
  3. stacked_diverging_quantidade_magnitude.png

E salva um CSV "long format" com todos os dados calculados, para reuso.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Faixas de magnitude para o gráfico 3 (diverging stacked bar).
# Ajuste os cortes conforme a distribuição real dos seus dados
# (ex.: baseado em quartis, depois de rodar uma vez e olhar describe()).
def categorizar_magnitude(diff):
    if diff <= -2:
        return "Piorou muito"
    elif diff < 0:
        return "Piorou pouco"
    elif diff == 0:
        return "Não mudou"
    elif diff < 2:
        return "Melhorou pouco"
    else:
        return "Melhorou muito"

CATEGORY_ORDER = ["Piorou muito", "Piorou pouco", "Não mudou", "Melhorou pouco", "Melhorou muito"]
CATEGORY_COLORS = {
    "Piorou muito": "#b2182b",
    "Piorou pouco": "#ef8a62",
    "Não mudou": "#d9d9d9",
    "Melhorou pouco": "#67a9cf",
    "Melhorou muito": "#2166ac",
}

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# 1. CARREGAR E TRANSFORMAR OS DADOS PARA FORMATO LONGO
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
                    continue  # pula se faltar algum score (dado incompleto)

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
    df["magnitude_cat"] = df["diff"].apply(categorizar_magnitude)
    return df


# ---------------------------------------------------------------------------
# 2. GRÁFICO 1 — BARRAS DIVERGENTES (MÉDIA DA DIFF POR ATRIBUTO), PAINEL POR MODELO
# ---------------------------------------------------------------------------
def grafico_barras_divergentes(df, output_path):
    modelos = sorted(df["model"].unique())
    n = len(modelos)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharex=True)
    if n == 1:
        axes = [axes]

    # ordena atributos pela magnitude média global (consistência entre os painéis)
    ordem_global = (
        df.groupby("attribute_label")["diff"].mean().sort_values().index.tolist()
    )

    for ax, modelo in zip(axes, modelos):
        sub = (
            df[df["model"] == modelo]
            .groupby("attribute_label")["diff"]
            .mean()
            .reindex(ordem_global)
        )
        cores = ["#b2182b" if v < 0 else "#2166ac" for v in sub.values]
        ax.barh(sub.index, sub.values, color=cores)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(modelo)
        ax.set_xlabel("Diferença média (com − sem código)")

    axes[0].set_ylabel("Atributo da rubrica")
    fig.suptitle("Impacto médio do código-fonte por atributo, por modelo", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico 1 salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 3. GRÁFICO 2 — BOXPLOT AGRUPADO POR ATRIBUTO, COR POR MODELO
#
# IMPORTANTE (2 ajustes motivados pelos dados reais):
#
# (a) "diff" é um valor inteiro discreto (escala Likert): um strip/jitter
#     plot com milhares de tools sobrepõe os pontos em faixas horizontais e
#     esconde as caixas por baixo — por isso usamos boxplot "puro", sem
#     overlay de pontos individuais (correto para amostras grandes).
#
# (b) a maioria dos pares (com/sem código) EMPATA (diff = 0). Isso é
#     esperado e é exatamente o que o gráfico 3 (stacked diverging) mede.
#     Mas incluir os empates aqui faz a caixa inteira colapsar em zero
#     (mediana = Q1 = Q3 = 0), escondendo a pergunta que esse gráfico
#     deveria responder: "quando a nota MUDA, de quanto é a mudança, e
#     pra qual direção?". Por isso o boxplot é calculado apenas sobre os
#     casos com diff != 0 — os empates ficam reportados como % ao lado de
#     cada atributo, e sua contagem completa já está no gráfico 3.
# ---------------------------------------------------------------------------
def grafico_boxplot_strip(df, output_path):
    ordem_global = (
        df.groupby("attribute_label")["diff"].mean().sort_values().index.tolist()
    )

    df_mudou = df[df["diff"] != 0].copy()

    # % de empates por atributo (para anotar no eixo X)
    pct_empate = (
        df.groupby("attribute_label")["diff"]
        .apply(lambda s: (s == 0).mean() * 100)
        .reindex(ordem_global)
    )
    novos_labels = [f"{a}\n({pct_empate[a]:.0f}% empate)" for a in ordem_global]
    label_map = dict(zip(ordem_global, novos_labels))
    df_mudou["attribute_label_anotado"] = df_mudou["attribute_label"].map(label_map)
    ordem_anotada = [label_map[a] for a in ordem_global]

    fig, ax = plt.subplots(figsize=(15, 8))

    sns.boxplot(
        data=df_mudou,
        x="attribute_label_anotado",
        y="diff",
        hue="model",
        order=ordem_anotada,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.3, markeredgewidth=0),
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6
        ),
        linewidth=1.2,
        ax=ax,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Atributo da rubrica  (% de empates indicado abaixo do nome)", labelpad=15)
    ax.set_ylabel("Diferença (com − sem código)\napenas casos que mudaram", labelpad=10)
    ax.set_title("Distribuição da diferença por atributo e por modelo\n(excluindo empates, diff = 0)", pad=15)
    ax.legend(title="Modelo", bbox_to_anchor=(1.02, 1), loc="upper left")
    y_min, y_max = int(df_mudou["diff"].min()), int(df_mudou["diff"].max())
    ax.set_yticks(range(y_min, y_max + 1))
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico 2 salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 4. GRÁFICO 3 — DIVERGING STACKED BAR (QUANTIDADE x MAGNITUDE), POR ATRIBUTO
#    (uma versão agregando todos os modelos, e opcionalmente uma por modelo)
# ---------------------------------------------------------------------------
def _plot_stacked(ax, df_subset, titulo):
    contagem = (
        df_subset.groupby(["attribute_label", "magnitude_cat"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CATEGORY_ORDER, fill_value=0)
    )

    # ordena atributos pelo total de mudanças (não-neutras), decrescente
    total_mudancas = contagem.drop(columns=["Não mudou"]).sum(axis=1)
    contagem = contagem.loc[total_mudancas.sort_values(ascending=True).index]

    # calcula porcentagens para plot centrado em zero (diverging)
    pct = contagem.div(contagem.sum(axis=1), axis=0) * 100

    # metade das categorias negativas + metade de "não mudou" empurradas p/ esquerda
    esquerda = pct[["Piorou muito", "Piorou pouco"]].sum(axis=1) + pct["Não mudou"] / 2
    offset = -esquerda

    left_acc = offset.copy()
    for cat in CATEGORY_ORDER:
        vals = pct[cat]
        ax.barh(pct.index, vals, left=left_acc, color=CATEGORY_COLORS[cat], label=cat)
        left_acc = left_acc + vals

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("% de casos")


def grafico_stacked_diverging(df, output_path, por_modelo=False):
    if not por_modelo:
        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_stacked(ax, df, "Distribuição das mudanças por atributo (todos os modelos)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(
            by_label.values(),
            by_label.keys(),
            title="Magnitude",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Gráfico 3 salvo em: {output_path}")
    else:
        modelos = sorted(df["model"].unique())
        n = len(modelos)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, modelo in zip(axes, modelos):
            _plot_stacked(ax, df[df["model"] == modelo], modelo)
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            title="Magnitude",
            bbox_to_anchor=(1.0, 1.05),
            loc="upper left",
        )
        fig.suptitle("Distribuição das mudanças por atributo, por modelo", y=1.08)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Gráfico 3 (por modelo) salvo em: {output_path}")


# ---------------------------------------------------------------------------
# 5. TESTE RÁPIDO DE CONCORDÂNCIA ENTRE MODELOS (para decidir se agrega ou não)
# ---------------------------------------------------------------------------
def checar_concordancia(df):
    tabela = df.groupby(["model", "attribute_label"])["diff"].mean().unstack(0)
    print("\n=== Diferença média por atributo e modelo ===")
    print(tabela.round(2))
    print("\n=== Correlação entre modelos (ordem de atributos mais afetados) ===")
    print(tabela.corr(method="spearman").round(2))
    return tabela


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = carregar_dados(INPUT_JSON)
    print(f"Total de linhas (tool x modelo x atributo): {len(df)}")

    # salva o dataset já processado, para reuso em outras análises/gráficos
    csv_path = f"{OUTPUT_DIR}/dados_long_format.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV long-format salvo em: {csv_path}")

    checar_concordancia(df)

    grafico_barras_divergentes(df, f"{OUTPUT_DIR}/barras_divergentes_por_modelo.png")
    grafico_boxplot_strip(df, f"{OUTPUT_DIR}/boxplot_strip_por_atributo_modelo.png")
    grafico_stacked_diverging(
        df, f"{OUTPUT_DIR}/stacked_diverging_quantidade_magnitude.png", por_modelo=False
    )
    # versão extra por modelo, para checar consistência (ex.: usar no apêndice)
    grafico_stacked_diverging(
        df,
        f"{OUTPUT_DIR}/stacked_diverging_quantidade_magnitude_por_modelo.png",
        por_modelo=True,
    )


if __name__ == "__main__":
    main()