"""
Resultados (RQ2/RQ3) a partir da base ATUAL de avaliações, separados por juiz.

Diferente de gerar_graficos.py (que lia o snapshot charts/file.json e tirava
a MÉDIA entre modelos), este script lê diretamente data/evaluations/*.jsonl e
reporta cada juiz separadamente: os juízes reagem ao código em sentidos
opostos, e a média entre eles cancela parte do efeito.

Para cada juiz: pareia description_only x with_source pela mesma chave de
tool, deduplicação e filtro de versão de prompt de
scripts/analysis_evaluation_report.py (tool_key_for, dedupe_records,
registros_versao_ativa), e calcula, por componente: médias, % diminuiu/
empatou/aumentou, Wilcoxon (zero_method="wilcox") com correção de
Benjamini-Hochberg entre os seis componentes, e tamanho de efeito r
(correlação rank-biserial pareada, Kerby 2014).

Saídas em charts/outputs_por_juiz/ (CSVs + PNGs). Rodar da raiz do projeto:
    uv run python charts/gerar_resultados_por_juiz.py
"""

import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import false_discovery_control, rankdata, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.analysis_evaluation_report import (  # noqa: E402
    load_evaluations,
    motivos_por_divergencia,
    registros_versao_ativa,
    tool_key_for,
)
from scripts.dedupe_evaluations import dedupe_records  # noqa: E402

EVAL_DIR = "data/evaluations"
OUTPUT_DIR = "charts/outputs_por_juiz"

JUDGES = {
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "qwen3-14b-ollama": "Qwen3-14B",
}

ATTRIBUTES = [
    "purpose",
    "parameter_explanation",
    "length_completeness",
    "guidelines",
    "limitations",
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

COR_DIMINUIU = "#d62728"
COR_EMPATOU = "#bbbbbb"
COR_AUMENTOU = "#1f77b4"

VIRGULA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "font.size": 12,
})


# ---------------------------------------------------------------------------
# 1. CARREGAR DADOS (nível juiz x tool x atributo, apenas pares completos)
# ---------------------------------------------------------------------------
def carregar_registros():
    records = dedupe_records(load_evaluations(Path(EVAL_DIR)))
    return [r for r in registros_versao_ativa(records) if r["judge"]["id"] in JUDGES]


def carregar_pares(records):
    scores = {}
    for r in records:
        if r.get("status") != "ok":
            continue
        scores[(r["judge"]["id"], tool_key_for(r), r["scenario"])] = {
            a: r["scores"][a]["score"] for a in ATTRIBUTES
        }

    rows = []
    for (judge_id, tool, scenario), sem in scores.items():
        if scenario != "description_only":
            continue
        com = scores.get((judge_id, tool, "with_source"))
        if com is None:
            continue
        for a in ATTRIBUTES:
            rows.append({
                "judge": judge_id,
                "tool_uid": tool,
                "attribute": a,
                "score_sem_codigo": sem[a],
                "score_com_codigo": com[a],
                "diff": com[a] - sem[a],
            })
    df = pd.DataFrame(rows)
    df["judge"] = pd.Categorical(df["judge"], categories=list(JUDGES), ordered=True)
    return df.sort_values(["judge", "tool_uid"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. ESTATÍSTICAS POR JUIZ x COMPONENTE
# ---------------------------------------------------------------------------
def rank_biserial(diff):
    """Correlação rank-biserial pareada (Kerby, 2014), ignorando empates."""
    d = diff[diff != 0]
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def estatisticas(df):
    resultados = []
    for judge_id, g_judge in df.groupby("judge", sort=False, observed=True):
        linhas = []
        for a in ATTRIBUTES:
            g = g_judge[g_judge["attribute"] == a]
            diff = g["diff"].to_numpy()
            p = wilcoxon(g["score_com_codigo"], g["score_sem_codigo"],
                         zero_method="wilcox").pvalue
            linhas.append({
                "judge": judge_id,
                "attribute": a,
                "n_pareado": len(g),
                "media_sem_codigo": g["score_sem_codigo"].mean(),
                "media_com_codigo": g["score_com_codigo"].mean(),
                "delta": diff.mean(),
                "pct_diminuiu": (diff < 0).mean() * 100,
                "pct_empatou": (diff == 0).mean() * 100,
                "pct_aumentou": (diff > 0).mean() * 100,
                "n_diferentes": int((diff != 0).sum()),
                "r_rank_biserial": rank_biserial(diff),
                "p_valor": p,
            })
        p_bh = false_discovery_control([l["p_valor"] for l in linhas])
        for l, pb in zip(linhas, p_bh):
            l["p_valor_bh"] = pb
        resultados.extend(linhas)
    return pd.DataFrame(resultados)


def subconjunto_comum(df):
    """Restringe às tools pareadas pelos três juízes."""
    comuns = set.intersection(*[
        set(g["tool_uid"]) for _, g in df.groupby("judge", observed=True)
    ])
    return df[df["tool_uid"].isin(comuns)]


def robustez_subconjunto_comum(df):
    """Mesmo delta médio, restrito às tools pareadas pelos três juízes."""
    sub = subconjunto_comum(df)
    comuns = set(sub["tool_uid"])
    tabela = sub.groupby(["judge", "attribute"], observed=True)["diff"].mean().unstack()
    return len(comuns), tabela


def correlacao_entre_juizes(df):
    """Spearman entre cada par de juízes, por componente: nota sem código (S), nota com
    código (C) e mudança da nota (Δ = C - S). Chamado com subconjunto_comum(df), para que
    os três pares sejam comparados sobre as mesmas tools."""
    colunas = {"S": "score_sem_codigo", "C": "score_com_codigo", "Δ": "diff"}
    largos = {k: df.pivot_table(index=["tool_uid", "attribute"], columns="judge",
                                values=v, observed=True) for k, v in colunas.items()}
    linhas = []
    for a, b in combinations(JUDGES, 2):
        for medida, largo in largos.items():
            par = largo[[a, b]].dropna()
            for attr in ATTRIBUTES + ["geral"]:
                s = par if attr == "geral" else par.xs(attr, level="attribute")
                linhas.append({
                    "juiz_a": a, "juiz_b": b, "medida": medida, "attribute": attr,
                    "n_tools": len(par) // len(ATTRIBUTES),
                    "spearman": spearmanr(s[a], s[b]).statistic,
                })
    return pd.DataFrame(linhas)


def gerar_tabela_latex_correlacao(df_corr, output_path):
    pares = list(combinations(JUDGES, 2))
    nomes_curtos = {"deepseek-flash": "DeepSeek", "gemini-3.5-flash-lite": "Gemini",
                    "qwen3-14b-ollama": "Qwen3"}
    fmt = lambda v: f"{v:.2f}".replace(".", ",").replace("-", "$-$")

    cabecalho_pares = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{{nomes_curtos[a]} $\\times$ {nomes_curtos[b]}}}}}"
        for a, b in pares)
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(pares)))
    linhas = []
    for attr in ATTRIBUTES + ["geral"]:
        valores = []
        for a, b in pares:
            for medida in ["S", "C", "Δ"]:
                v = df_corr.query("juiz_a == @a and juiz_b == @b and medida == @medida "
                                  "and attribute == @attr")["spearman"].iloc[0]
                valores.append(fmt(v))
        rotulo = "\\textbf{Geral}" if attr == "geral" else ATTR_LABELS[attr].replace("&", "\\&")
        if attr == "geral":
            linhas.append("\\midrule")
        linhas.append(f"{rotulo} & " + " & ".join(valores) + " \\\\")

    n_tools = f"{df_corr['n_tools'].iloc[0]:,}".replace(",", ".")
    corpo = "\n".join(linhas)
    tabela = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Correlação de Spearman ($\\rho$) entre os juízes, por componente da rubrica: "
        "nota no cenário sem código (S), nota no cenário com código (C) e mudança da nota "
        f"($\\Delta = C - S$), sobre as {n_tools} ferramentas avaliadas pelos três juízes.}}\n"
        "\\label{tab:correlacao-juizes}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{lrrrrrrrrr}\n"
        "\\toprule\n"
        f" & {cabecalho_pares} \\\\\n"
        f"{cmid}\n"
        "\\textbf{Componente}" + " & S & C & $\\Delta$" * len(pares) + " \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tabela)
    print(f"[OK] {output_path}")


def gerar_tabela_latex_correlacao_geral(df_corr, output_path):
    """Versão resumida de gerar_tabela_latex_correlacao(): só a linha Geral, um par por linha."""
    nomes_curtos = {"deepseek-flash": "DeepSeek", "gemini-3.5-flash-lite": "Gemini",
                    "qwen3-14b-ollama": "Qwen3"}
    fmt = lambda v: f"{v:.2f}".replace(".", ",").replace("-", "$-$")
    geral = df_corr[df_corr["attribute"] == "geral"]

    linhas = []
    for a, b in combinations(JUDGES, 2):
        par = geral[(geral["juiz_a"] == a) & (geral["juiz_b"] == b)].set_index("medida")["spearman"]
        linhas.append(f"{nomes_curtos[a]} $\\times$ {nomes_curtos[b]} & "
                      f"{fmt(par['S'])} & {fmt(par['C'])} & {fmt(par['Δ'])} \\\\")

    n_tools = f"{df_corr['n_tools'].iloc[0]:,}".replace(",", ".")
    corpo = "\n".join(linhas)
    tabela = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Correlação de Spearman ($\\rho$) geral entre os juízes, considerando os seis "
        "componentes da rubrica em conjunto: nota no cenário sem código (S), nota no cenário com "
        f"código (C) e mudança da nota ($\\Delta = C - S$), sobre as {n_tools} ferramentas "
        "avaliadas pelos três juízes.}\n"
        "\\label{tab:correlacao-juizes-geral}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "\\textbf{Par de juízes} & \\textbf{S} & \\textbf{C} & \\textbf{$\\Delta$} \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tabela)
    print(f"[OK] {output_path}")


def subconjunto_consenso(df):
    """Pares (tool, componente) em que os três juízes mudaram a nota (Δ != 0 nos três),
    o mesmo critério da amostra de consenso (scripts/export_consenso_sample.py), restrito
    às tools avaliadas pelos três juízes."""
    sub = subconjunto_comum(df)
    mudou = (sub.pivot_table(index=["tool_uid", "attribute"], columns="judge",
                             values="diff", observed=True) != 0).all(axis=1)
    chaves = set(mudou[mudou].index)
    return sub[[k in chaves for k in zip(sub["tool_uid"], sub["attribute"])]]


def correlacao_consenso(df_consenso):
    """Por par de juízes, sobre os pares (tool, componente) do consenso: Spearman de S, C
    e Δ, e % dos pares em que os dois juízes mudaram a nota no mesmo sentido."""
    largos = {k: df_consenso.pivot_table(index=["tool_uid", "attribute"], columns="judge",
                                         values=v, observed=True)
              for k, v in {"S": "score_sem_codigo", "C": "score_com_codigo", "Δ": "diff"}.items()}
    linhas = []
    for a, b in combinations(JUDGES, 2):
        d = largos["Δ"]
        linhas.append({
            "juiz_a": a, "juiz_b": b,
            "n_pares": len(d),
            "n_tools": d.index.get_level_values("tool_uid").nunique(),
            "rho_S": spearmanr(largos["S"][a], largos["S"][b]).statistic,
            "rho_C": spearmanr(largos["C"][a], largos["C"][b]).statistic,
            "rho_delta": spearmanr(d[a], d[b]).statistic,
            "pct_mesmo_sentido": ((d[a] > 0) == (d[b] > 0)).mean() * 100,
        })
    return pd.DataFrame(linhas)


def gerar_tabela_latex_correlacao_consenso(df_cons, output_path):
    nomes_curtos = {"deepseek-flash": "DeepSeek", "gemini-3.5-flash-lite": "Gemini",
                    "qwen3-14b-ollama": "Qwen3"}
    fmt = lambda v: f"{v:.2f}".replace(".", ",").replace("-", "$-$")

    linhas = [
        f"{nomes_curtos[r.juiz_a]} $\\times$ {nomes_curtos[r.juiz_b]} & {fmt(r.rho_S)} & "
        f"{fmt(r.rho_C)} & {fmt(r.rho_delta)} & "
        + f"{r.pct_mesmo_sentido:.1f}".replace(".", ",") + "\\% \\\\"
        for r in df_cons.itertuples()
    ]
    n_pares = f"{df_cons['n_pares'].iloc[0]:,}".replace(",", ".")
    n_tools = f"{df_cons['n_tools'].iloc[0]:,}".replace(",", ".")
    corpo = "\n".join(linhas)
    tabela = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Correlação de Spearman ($\\rho$) entre os juízes restrita à amostra de "
        f"consenso: {n_pares} pares ferramenta-componente ({n_tools} ferramentas) em que os três "
        "juízes alteraram a nota entre os cenários. S: nota sem código; C: nota com código; "
        "$\\Delta = C - S$; última coluna: proporção dos pares em que os dois juízes alteraram "
        "a nota no mesmo sentido.}\n"
        "\\label{tab:correlacao-juizes-consenso}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "\\textbf{Par de juízes} & \\textbf{S} & \\textbf{C} & \\textbf{$\\Delta$} & "
        "\\textbf{Mesmo sentido} \\\\\n"
        "\\midrule\n"
        f"{corpo}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tabela)
    print(f"[OK] {output_path}")


# ---------------------------------------------------------------------------
# 3. TRANSIÇÃO DE QUARTIS (faixas definidas pela distribuição sem código)
# ---------------------------------------------------------------------------
def _faixa(v, q1, q2, q3):
    return np.select([v <= q1, v <= q2, v <= q3], [0, 1, 2], default=3)


def transicao_quartis(df):
    linhas = []
    for (judge_id, a), g in df.groupby(["judge", "attribute"], sort=False, observed=True):
        sem = g["score_sem_codigo"].to_numpy()
        com = g["score_com_codigo"].to_numpy()
        q1, q2, q3 = np.percentile(sem, [25, 50, 75])
        fs, fc = _faixa(sem, q1, q2, q3), _faixa(com, q1, q2, q3)
        linhas.append({
            "judge": judge_id,
            "attribute": a,
            "pct_manteve": (fc == fs).mean() * 100,
            "pct_subiu": (fc > fs).mean() * 100,
            "pct_desceu": (fc < fs).mean() * 100,
        })
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# 4. MOTIVOS (busca por palavras-chave), RESUMO POR JUIZ
# ---------------------------------------------------------------------------
def resumo_motivos(records):
    m = motivos_por_divergencia(records)
    linhas = []
    for (judge_id, motivo), g in m.groupby(["juiz", "motivo"]):
        total_div = len(m.loc[m["juiz"] == judge_id, ["componente", "tool_uid"]].drop_duplicates())
        linhas.append({
            "judge": judge_id,
            "motivo": motivo,
            "ocorrencias": len(g),
            "total_divergencias": total_div,
            "pct_das_divergencias": len(g) / total_div * 100,
            "subiu": int(g["subiu"].sum()),
            "desceu": int((~g["subiu"]).sum()),
        })
    return pd.DataFrame(linhas).sort_values(["judge", "ocorrencias"], ascending=[True, False])


# ---------------------------------------------------------------------------
# 4b. CONCORDÂNCIA ENTRE JUÍZES (notas e direção da mudança)
# ---------------------------------------------------------------------------
DIRECOES = (-1, 0, 1)  # desceu, empatou, subiu
NOMES_CURTOS = {"deepseek-flash": "DeepSeek", "gemini-3.5-flash-lite": "Gemini",
                "qwen3-14b-ollama": "Qwen3"}


def _kappa_cohen(a, b):
    """Kappa de Cohen (Cohen, 1960) para dois juízes sobre as três direções."""
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in DIRECOES)
    return po, pe, (po - pe) / (1 - pe)


def _kappa_fleiss(rotulos):
    """Kappa de Fleiss (Fleiss, 1971); `rotulos` é uma matriz itens x juízes."""
    n, k = rotulos.shape
    contagens = np.stack([(rotulos == c).sum(axis=1) for c in DIRECOES], axis=1)
    po = ((contagens * (contagens - 1)).sum(axis=1) / (k * (k - 1))).mean()
    pe = ((contagens.sum(axis=0) / (n * k)) ** 2).sum()
    return po, pe, (po - pe) / (1 - pe)


def concordancia_entre_juizes(df):
    """Sobre as tools avaliadas pelos três juízes (subconjunto_comum), por componente e
    no geral (componentes empilhados):
      - rho_S / rho_C: Spearman das notas sem e com código, por par de juízes;
      - kappa: Cohen da direção da mudança (desceu/empatou/subiu), por par;
      - po / pe: concordância observada e esperada ao acaso que compõem o kappa;
    e uma linha extra por componente com o kappa de Fleiss dos três juízes juntos."""
    largos = {k: df.pivot_table(index=["tool_uid", "attribute"], columns="judge",
                                values=v, observed=True)
              for k, v in {"S": "score_sem_codigo", "C": "score_com_codigo",
                           "D": "diff"}.items()}
    direcao = np.sign(largos["D"])
    linhas = []
    for attr in ATTRIBUTES + ["geral"]:
        sel = (lambda x: x) if attr == "geral" else (lambda x: x.xs(attr, level="attribute"))
        s, c, d = sel(largos["S"]), sel(largos["C"]), sel(direcao)
        for a, b in combinations(JUDGES, 2):
            po, pe, kappa = _kappa_cohen(d[a].to_numpy(), d[b].to_numpy())
            linhas.append({
                "attribute": attr, "par": f"{NOMES_CURTOS[a]} × {NOMES_CURTOS[b]}",
                "n_itens": len(d),
                "rho_S": spearmanr(s[a], s[b]).statistic,
                "rho_C": spearmanr(c[a], c[b]).statistic,
                "po": po, "pe": pe, "kappa": kappa,
            })
        po, pe, kappa = _kappa_fleiss(d[list(JUDGES)].to_numpy())
        linhas.append({"attribute": attr, "par": "Três juízes (Fleiss)", "n_itens": len(d),
                       "rho_S": np.nan, "rho_C": np.nan, "po": po, "pe": pe, "kappa": kappa})
    return pd.DataFrame(linhas)


def grafico_concordancia(df_conc, output_path):
    """Mapa de calor em três painéis, mesma escala sequencial (0 a 1) para todos, para
    que a diferença de magnitude entre concordar na nota e concordar na mudança seja
    visível: (a) Spearman sem código, (b) Spearman com código, (c) kappa da direção."""
    pares = [p for p in df_conc["par"].unique() if "Fleiss" not in p]
    linhas_ordem = ATTRIBUTES + ["geral"]
    rotulos_linhas = [ATTR_LABELS.get(a, "Geral") for a in linhas_ordem]
    paineis = [
        ("(a) Notas sem código\nSpearman ρ", "rho_S", pares),
        ("(b) Notas com código\nSpearman ρ", "rho_C", pares),
        ("(c) Direção da mudança\nkappa κ", "kappa", pares + ["Três juízes (Fleiss)"]),
    ]
    rampa = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "azul", ["#f7fbff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2),
                             gridspec_kw={"width_ratios": [3, 3, 4]})
    for ax, (titulo, coluna, cols) in zip(axes, paineis):
        matriz = np.array([[df_conc.query("attribute == @a and par == @p")[coluna].iloc[0]
                            for p in cols] for a in linhas_ordem])
        ax.imshow(np.clip(matriz, 0, 1), cmap=rampa, vmin=0, vmax=1, aspect="auto")
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                v = matriz[i, j]
                ax.text(j, i, f"{v:.2f}".replace(".", ",").replace("-", "\u2212"),
                        ha="center", va="center", fontsize=11,
                        fontweight="bold" if linhas_ordem[i] == "geral" else "normal",
                        color="white" if v >= 0.55 else "#1a1a1a")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([p.replace(" × ", "\n× ").replace(" (Fleiss)", "\n(Fleiss)")
                            for p in cols], fontsize=10)
        ax.set_yticks(range(len(linhas_ordem)))
        ax.set_yticklabels(rotulos_linhas if ax is axes[0] else [])
        ax.axhline(len(ATTRIBUTES) - 0.5, color="white", linewidth=3)
        ax.set_title(titulo, fontsize=12)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

    barra = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=rampa),
                         ax=axes, orientation="horizontal", fraction=0.05, pad=0.12,
                         aspect=50)
    barra.set_label("Valor do coeficiente (0 = sem concordância; 1 = concordância total)")
    barra.ax.xaxis.set_major_formatter(VIRGULA)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def gerar_tabela_latex_kappa(df_conc, output_path):
    """Detalhe do painel (c): concordância observada, esperada ao acaso e kappa."""
    fmt = lambda v: f"{v:.2f}".replace(".", ",").replace("-", "$-$")
    pares = list(df_conc["par"].unique())
    cab = " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{p.replace(' × ', ' $\\times$ ').replace('Três juízes (Fleiss)', 'Fleiss (3 juízes)')}}}}}"
                     for p in pares)
    cmid = " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(pares)))
    linhas = []
    for attr in ATTRIBUTES + ["geral"]:
        valores = []
        for p in pares:
            r = df_conc.query("attribute == @attr and par == @p").iloc[0]
            valores += [fmt(r.po), fmt(r.pe), fmt(r.kappa)]
        if attr == "geral":
            linhas.append("\\midrule")
        rotulo = "\\textbf{Geral}" if attr == "geral" else ATTR_LABELS[attr].replace("&", "\\&")
        linhas.append(f"{rotulo} & " + " & ".join(valores) + " \\\\")
    n = f"{df_conc['n_itens'].iloc[0]:,}".replace(",", ".")
    tabela = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\caption{Concordância entre os juízes quanto à direção da mudança da nota com a "
        "inclusão do código (diminuiu, manteve ou aumentou), por componente, sobre as "
        f"{n} ferramentas avaliadas pelos três juízes: concordância observada ($p_o$), "
        "concordância esperada ao acaso ($p_e$) e kappa ($\\kappa$) de Cohen, por par, e de "
        "Fleiss, para os três juízes.}\n"
        "\\label{tab:kappa-direcao}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\begin{tabular}{l" + "rrr" * len(pares) + "}\n"
        "\\toprule\n"
        f" & {cab} \\\\\n"
        f"{cmid}\n"
        "\\textbf{Componente}" + " & $p_o$ & $p_e$ & $\\kappa$" * len(pares) + " \\\\\n"
        "\\midrule\n"
        + "\n".join(linhas) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tabela)
    print(f"[OK] {output_path}")


# ---------------------------------------------------------------------------
# 5. FIGURAS
# ---------------------------------------------------------------------------
def grafico_diminuiu_empatou_aumentou(df_stats, output_path):
    ordem = ATTRIBUTES[::-1]
    fig, axes = plt.subplots(1, len(JUDGES), figsize=(16, 5.5), sharey=True)
    for ax, (judge_id, nome) in zip(axes, JUDGES.items()):
        s = df_stats[df_stats["judge"] == judge_id].set_index("attribute").loc[ordem]
        y = np.arange(len(ordem))
        esquerda = np.zeros(len(ordem))
        for col, cor, rotulo in [("pct_diminuiu", COR_DIMINUIU, "Diminuiu"),
                                 ("pct_aumentou", COR_AUMENTOU, "Aumentou"),
                                 ("pct_empatou", COR_EMPATOU, "Empatou")]:
            valores = s[col].to_numpy()
            barras = ax.barh(y, valores, left=esquerda, color=cor, label=rotulo)
            for rect, v in zip(barras, valores):
                if v < 4:
                    continue
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + rect.get_height() / 2,
                        f"{v:.0f}%", va="center", ha="center", fontsize=9,
                        color="black" if col == "pct_empatou" else "white")
            esquerda += valores
        n = int(s["n_pareado"].iloc[0])
        ax.set_title(f"{nome}\n(n = {n:,} tools)".replace(",", "."), fontsize=12)
        ax.set_yticks(y)
        ax.set_yticklabels([ATTR_LABELS[a] for a in ordem])
        ax.set_xlim(0, 100)
        ax.set_xlabel("% de tools")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


def grafico_delta_por_juiz(df_stats, output_path):
    ordem = ATTRIBUTES[::-1]
    cores = {"deepseek-flash": "#4c72b0", "gemini-3.5-flash-lite": "#dd8452",
             "qwen3-14b-ollama": "#55a868"}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = np.arange(len(ordem))
    altura = 0.26
    for i, (judge_id, nome) in enumerate(JUDGES.items()):
        s = df_stats[df_stats["judge"] == judge_id].set_index("attribute").loc[ordem]
        pos = y + (1 - i) * altura
        ax.barh(pos, s["delta"], height=altura, color=cores[judge_id], label=nome)
        for yi, v in zip(pos, s["delta"]):
            ax.text(v + (0.01 if v >= 0 else -0.01), yi, f"{v:+.2f}".replace(".", ","),
                    va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([ATTR_LABELS[a] for a in ordem])
    ax.set_xlabel("Diferença média da nota (com código − sem código)")
    ax.set_xlim(-0.25, 0.72)
    ax.xaxis.set_major_formatter(VIRGULA)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    records = carregar_registros()
    df = carregar_pares(records)

    df_stats = estatisticas(df)
    df_stats.to_csv(os.path.join(OUTPUT_DIR, "comparacao_por_juiz.csv"), index=False)

    n_comuns, delta_comum = robustez_subconjunto_comum(df)
    delta_comum.to_csv(os.path.join(OUTPUT_DIR, "delta_subconjunto_comum.csv"))

    df_corr = correlacao_entre_juizes(subconjunto_comum(df))
    df_corr.to_csv(os.path.join(OUTPUT_DIR, "correlacao_entre_juizes.csv"), index=False)
    gerar_tabela_latex_correlacao(df_corr, os.path.join(OUTPUT_DIR, "tabela_correlacao_juizes.tex"))
    gerar_tabela_latex_correlacao_geral(
        df_corr, os.path.join(OUTPUT_DIR, "tabela_correlacao_juizes_geral.tex"))

    df_cons = correlacao_consenso(subconjunto_consenso(df))
    df_cons.to_csv(os.path.join(OUTPUT_DIR, "correlacao_entre_juizes_consenso.csv"), index=False)
    gerar_tabela_latex_correlacao_consenso(
        df_cons, os.path.join(OUTPUT_DIR, "tabela_correlacao_juizes_consenso.tex"))

    df_conc = concordancia_entre_juizes(subconjunto_comum(df))
    df_conc.to_csv(os.path.join(OUTPUT_DIR, "concordancia_entre_juizes.csv"), index=False)
    grafico_concordancia(df_conc, os.path.join(OUTPUT_DIR, "concordancia_entre_juizes.png"))
    gerar_tabela_latex_kappa(df_conc, os.path.join(OUTPUT_DIR, "tabela_kappa_direcao.tex"))

    df_quartis = transicao_quartis(df)
    df_quartis.to_csv(os.path.join(OUTPUT_DIR, "transicao_quartis_por_juiz.csv"), index=False)

    df_motivos = resumo_motivos(records)
    df_motivos.to_csv(os.path.join(OUTPUT_DIR, "motivos_por_juiz.csv"), index=False)

    grafico_diminuiu_empatou_aumentou(
        df_stats, os.path.join(OUTPUT_DIR, "diminuiu_empatou_aumentou_por_juiz.png"))
    grafico_delta_por_juiz(df_stats, os.path.join(OUTPUT_DIR, "delta_medio_por_juiz.png"))

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df_stats.round(3).to_string())
    print(f"\nSubconjunto comum aos três juízes: {n_comuns} tools")
    print(delta_comum.round(3).to_string())
    print(df_quartis.round(1).to_string())
    print(df_motivos.round(1).to_string())


if __name__ == "__main__":
    main()
