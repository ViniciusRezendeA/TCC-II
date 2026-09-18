"""Gera os diagramas do TCC (visao geral, Etapas 1/2/3, exemplo de call graph)
em layout horizontal, com texto reduzido a titulos curtos, salvando PNG em 300dpi.

Fontes grandes + figuras fisicamente compactas (poucas polegadas de largura),
porque a figura final entra no PDF encolhida para a largura da coluna
(~5.9in, A4 com margens de 3cm) -- se a figura-fonte for muito larga em
polegadas, o texto encolhe junto e fica ilegivel. Conectores usam cotovelo
reto (horizontal-vertical-horizontal) em vez de diagonal sempre que os dois
pontos nao estao alinhados em x ou em y.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT_DIR = "/private/tmp/claude-502/-Users-Viniciusrezende-Desktop-TCC-II/8f62c6e6-9aef-4cc6-b5a0-c5246113498b/scratchpad/out"
os.makedirs(OUT_DIR, exist_ok=True)

BLUE_FILL = "#dbe8fb"
BLUE_EDGE = "#2b5fad"
GRAY_FILL = "#eeeeee"
GRAY_EDGE = "#888888"
GREEN_FILL = "#d9f2d9"
GREEN_EDGE = "#3a8f3a"
RED_FILL = "#fbdada"
RED_EDGE = "#b23b3b"
ORANGE_FILL = "#fdeacb"

# Uma cor por etapa da metodologia (Figura 1), reaproveitada no diagrama
# detalhado de cada etapa (Figuras 2, 5, 10) para deixar visualmente
# explicito a qual etapa geral cada fluxograma detalhado pertence. As cores
# semanticas (verde=sucesso, vermelho=erro/desqualificado, laranja=backfill,
# cinza=descartado/externo) continuam reservadas para esses significados e
# nao sao reusadas aqui.
STEP_COLETA = ("#dbe8fb", "#2b5fad")        # azul
STEP_EXTRACAO = ("#ecdff5", "#7b3fa0")      # roxo
STEP_CLASSIFICACAO = ("#d7f2ef", "#0f7a72")  # verde-azulado
STEP_CONSOLIDACAO = ("#e4e8f0", "#4a5a78")   # azul-acinzentado
STEP_ANALISE = ("#f8dde4", "#a83f5a")        # rose
ORANGE_EDGE = "#c58a2b"
DIAMOND_FILL = "#eaf3fd"

FONT = "DejaVu Sans"
LW = 2.0


def box(ax, x, y, w, h, text, fill=BLUE_FILL, edge=BLUE_EDGE, fontsize=16, dashed=False, textcolor="#1a1a1a"):
    style = "round,pad=0.02,rounding_size=0.08"
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style, linewidth=LW,
        facecolor=fill, edgecolor=edge,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, family=FONT, linespacing=1.25)
    return (x, y, w, h)


def diamond(ax, x, y, w, h, text, fontsize=15, edge=BLUE_EDGE):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    poly = Polygon(pts, closed=True, facecolor=DIAMOND_FILL, edgecolor=edge, linewidth=LW)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, family=FONT, linespacing=1.2)
    return (x, y, w, h)


def straight(ax, p1, p2, label=None, color="#333333", label_dx=0, label_dy=0.22, ls="-", fontsize=13):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=20, linewidth=LW,
        color=color, connectionstyle="arc3,rad=0", linestyle=ls,
    )
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + label_dx, (p1[1] + p2[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                 family=FONT, backgroundcolor="white")


def elbow(ax, p1, p2, bend="x", at=None, label=None, color="#333333", ls="-",
          label_dx=0, label_dy=0.22, fontsize=13, lw=LW):
    """Conector em angulo reto (sem diagonal), com um cotovelo.

    bend="x": sai de p1 na horizontal, dobra em x=`at` (ou no meio), entra em
    p2 na horizontal (usado quando p1 e p2 tem y diferentes).
    bend="y": sai de p1 na vertical, dobra em y=`at`, entra em p2 na vertical.
    """
    x1, y1 = p1
    x2, y2 = p2
    if bend == "x":
        cx = at if at is not None else (x1 + x2) / 2
        corner1 = (cx, y1)
        corner2 = (cx, y2)
    else:
        cy = at if at is not None else (y1 + y2) / 2
        corner1 = (x1, cy)
        corner2 = (x2, cy)
    ax.add_line(Line2D([x1, corner1[0]], [y1, corner1[1]], color=color, linewidth=lw, linestyle=ls, solid_capstyle="butt"))
    ax.add_line(Line2D([corner1[0], corner2[0]], [corner1[1], corner2[1]], color=color, linewidth=lw, linestyle=ls, solid_capstyle="butt"))
    arr = FancyArrowPatch(corner2, p2, arrowstyle="-|>", mutation_scale=20, linewidth=lw, color=color, linestyle=ls)
    ax.add_patch(arr)
    if label:
        mx, my = (corner1[0] + corner2[0]) / 2 + label_dx, (corner1[1] + corner2[1]) / 2 + label_dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fontsize,
                 family=FONT, backgroundcolor="white")


def conn(ax, b1, b2, **kw):
    """Escolhe reto se os centros estao alinhados, cotovelo caso contrario."""
    p1, p2 = edge_right(b1), edge_left(b2)
    if abs(p1[1] - p2[1]) < 1e-6:
        straight(ax, p1, p2, **kw)
    else:
        elbow(ax, p1, p2, bend="x", **kw)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def edge_left(b):
    x, y, w, h = b
    return (x - w / 2, y)


def edge_right(b):
    x, y, w, h = b
    return (x + w / 2, y)


def edge_bottom(b):
    x, y, w, h = b
    return (x, y - h / 2)


def edge_top(b):
    x, y, w, h = b
    return (x, y + h / 2)


def layout_row(start_x, gap, widths):
    centers = []
    x = start_x
    for w in widths:
        cx = x + w / 2
        centers.append(cx)
        x = cx + w / 2 + gap
    return centers


# ---------------------------------------------------------------------------
# 1. Visao geral (metodologia_v7.png)
# ---------------------------------------------------------------------------
def gen_metodologia():
    w1, w2, w3, w4, w5 = 2.15, 2.5, 2.75, 2.3, 2.15
    h = 1.6
    xs = layout_row(0.2, 0.22, [w1, w2, w3, w4, w5])
    fig, ax = new_ax(xs[-1] + w5 / 2 + 0.2, 2.3)
    y = 1.15
    b1 = box(ax, xs[0], y, w1, h, "1. Coleta dos\nRepositórios MCP", fontsize=14.5,
             fill=STEP_COLETA[0], edge=STEP_COLETA[1])
    b2 = box(ax, xs[1], y, w2, h, "2. Extração das\nFerramentas e Código", fontsize=14,
             fill=STEP_EXTRACAO[0], edge=STEP_EXTRACAO[1])
    b3 = box(ax, xs[2], y, w3, h, "3. Classificação\n2 cenários × múltiplos\nmodelos de IA (juízes)", fontsize=12.5,
             fill=STEP_CLASSIFICACAO[0], edge=STEP_CLASSIFICACAO[1])
    b4 = box(ax, xs[3], y, w4, h, "4. Consolidação\ndos Resultados", fontsize=14,
             fill=STEP_CONSOLIDACAO[0], edge=STEP_CONSOLIDACAO[1])
    b5 = box(ax, xs[4], y, w5, h, "5. Análise\nComparativa", fontsize=14.5,
             fill=STEP_ANALISE[0], edge=STEP_ANALISE[1])
    conn(ax, b1, b2)
    conn(ax, b2, b3)
    conn(ax, b3, b4)
    conn(ax, b4, b5)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT_DIR}/metodologia_v7.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Etapa 1 (fluxo_etapa1_detalhado.png) -- layout em 2 fileiras (cotovelo),
# para caber mais largura por caixa e permitir fonte maior:
#   fileira 1: 1 -> 2 -> 3
#   fileira 2: 5 <- 4 <- (sim) <- Elegivel? -(nao)-> Descartado
# ---------------------------------------------------------------------------
def gen_etapa1():
    fill, edge = STEP_COLETA
    w, h = 2.7, 1.75
    dw, dh = 2.5, 2.3
    w5 = 2.35
    wdesc = 2.1
    gap = 0.35
    margin = 0.3

    row1_y = 3.75
    row2_y = 1.4

    x1 = margin + w / 2
    x2 = x1 + w + gap
    x3 = x2 + w + gap

    xd = x3  # losango alinhado sob a caixa 3
    x4 = xd - dw / 2 - gap - w / 2
    x5 = x4 - w / 2 - gap - w5 / 2
    xdesc = xd + dw / 2 + gap + wdesc / 2

    fig_w = max(x3 + w / 2, xdesc + wdesc / 2) + margin
    fig_h = (row1_y + h / 2) + margin + (dh / 2 - row2_y) + margin if row2_y < dh / 2 else (row1_y + h / 2) + margin
    fig, ax = new_ax(fig_w, 4.9)

    b1 = box(ax, x1, row1_y, w, h, "1. Mineração via\nREST Code Search", fontsize=15.5, fill=fill, edge=edge)
    b2 = box(ax, x2, row1_y, w, h, "2. Hidratação\nvia GraphQL", fontsize=17, fill=fill, edge=edge)
    b3 = box(ax, x3, row1_y, w, h, "3. Deduplicação\npor identificador", fontsize=15.5, fill=fill, edge=edge)
    d1 = diamond(ax, xd, row2_y, dw, dh, "Elegível?", fontsize=17, edge=edge)
    b4 = box(ax, x4, row2_y, w, h, "4. Ordenação\npor estrelas", fontsize=16, fill=fill, edge=edge)
    b5 = box(ax, x5, row2_y, w5, h, "5. Seleção\ntop-N", fontsize=17, fill=fill, edge=edge)
    desc = box(ax, xdesc, row2_y, wdesc, 1.0, "Descartado", fill=GRAY_FILL, edge=GRAY_EDGE, fontsize=15.5, dashed=True)

    straight(ax, edge_right(b1), edge_left(b2), fontsize=15)
    straight(ax, edge_right(b2), edge_left(b3), fontsize=15)
    elbow(ax, edge_bottom(b3), edge_top(d1), bend="y", fontsize=15)
    straight(ax, edge_left(d1), edge_right(b4), label="sim", label_dy=1.2, fontsize=15)
    straight(ax, edge_left(b4), edge_right(b5), fontsize=15)
    straight(ax, edge_right(d1), edge_left(desc), label="não", label_dy=0.36, fontsize=15)

    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT_DIR}/fluxo_etapa1_detalhado.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Etapa 2 (fluxo_etapa2_detalhado.png) -- layout em 3 fileiras (cotovelo):
#   fileira 1: 1 -> 2 -> 3
#   fileira 2: backfill <- Desqualificado <- (nao) <- >=1 ferramenta? (sim desce)
#   fileira 3: 6 <- 5 <- 4
# ---------------------------------------------------------------------------
def gen_etapa2():
    fill, edge = STEP_EXTRACAO
    w, h = 2.6, 1.7
    dw, dh = 2.4, 2.2
    wdesq, hdesq = 2.3, 0.95
    wbf, hbf = 2.5, 1.0
    w6 = 2.3
    gap, margin = 0.35, 0.3

    row1_y, row2_y, row3_y = 5.8, 3.5, 1.2

    x1 = margin + w / 2
    x2 = x1 + w + gap
    x3 = x2 + w + gap
    xd = x3

    x_desq = xd - dw / 2 - gap - wdesq / 2
    x_bf = x1

    x4 = xd
    x5 = x4 - w - gap
    x6 = x5 - w / 2 - gap - w6 / 2

    fig_w = max(x3 + w / 2, xd + dw / 2) + margin
    fig, ax = new_ax(fig_w, 7.0)

    b1 = box(ax, x1, row1_y, w, h, "1. Clonagem\nrasa", fontsize=17, fill=fill, edge=edge)
    b2 = box(ax, x2, row1_y, w, h, "2. Indexação\nestática", fontsize=17, fill=fill, edge=edge)
    b3 = box(ax, x3, row1_y, w, h, "3. Detecção de\nferramentas", fontsize=15.5, fill=fill, edge=edge)
    d1 = diamond(ax, xd, row2_y, dw, dh, "≥1\nferramenta?", fontsize=15.5, edge=edge)
    desq = box(ax, x_desq, row2_y, wdesq, hdesq, "Desqualificado", fill=RED_FILL, edge=RED_EDGE, fontsize=14.5)
    bf = box(ax, x_bf, row2_y, wbf, hbf, "Próximo candidato\ndo pool (backfill)", fill=ORANGE_FILL, edge=ORANGE_EDGE, fontsize=13.5)
    b4 = box(ax, x4, row3_y, w, h, "4. Grafo de\nchamadas", fontsize=16, fill=fill, edge=edge)
    b5 = box(ax, x5, row3_y, w, h, "5. Classificação\nde chamadas", fontsize=15, fill=fill, edge=edge)
    b6 = box(ax, x6, row3_y, w6, h, "6. Consolidação\nem tools.jsonl", fontsize=13.5, fill=fill, edge=edge)

    straight(ax, edge_right(b1), edge_left(b2), fontsize=15)
    straight(ax, edge_right(b2), edge_left(b3), fontsize=15)
    elbow(ax, edge_bottom(b3), edge_top(d1), bend="y", fontsize=15)
    straight(ax, edge_left(d1), edge_right(desq), label="não", label_dy=0.95, fontsize=14.5)
    straight(ax, edge_left(desq), edge_right(bf), fontsize=14.5)
    straight(ax, edge_top(bf), edge_bottom(b1), label="re-entra", label_dx=0.55, color="#555555", ls="--", fontsize=14)
    elbow(ax, edge_bottom(d1), edge_top(b4), bend="y", label="sim", label_dx=0.5, fontsize=14.5)
    straight(ax, edge_left(b4), edge_right(b5), fontsize=15)
    straight(ax, edge_left(b5), edge_right(b6), fontsize=15)

    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT_DIR}/fluxo_etapa2_detalhado.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Etapa 3 (fluxo_etapa3_detalhado.png)
# ---------------------------------------------------------------------------
def gen_etapa3():
    fill, edge = STEP_CLASSIFICACAO
    w, h = 1.9, 1.6
    dw = 1.85
    wr = 2.2
    hr = 1.0
    wres, hres = 2.2, 1.55
    wlog, hlog = 2.2, 0.95
    stub = 0.45
    xs = layout_row(0.2, 0.2, [w, w, w, w, dw])
    y = 2.9
    d1_right_x = xs[-1] + dw / 2

    bus_x = d1_right_x + stub
    xr = bus_x + stub + wr / 2
    outcome_right_x = xr + wr / 2
    bus2_x = outcome_right_x + stub
    xc = bus2_x + stub + wres / 2

    fig_w = xc + wres / 2 + 0.2
    fig_h = 5.4
    fig, ax = new_ax(fig_w, fig_h)

    b1 = box(ax, xs[0], y, w, h, "Montagem\ndos 2 cenários", fontsize=15, fill=fill, edge=edge)
    b2 = box(ax, xs[1], y, w, h, "Payload com\ndescrição + código", fontsize=13.5, fill=fill, edge=edge)
    b3 = box(ax, xs[2], y, w, h, "Envio concorrente\naos juízes", fontsize=14, fill=fill, edge=edge)
    b4 = box(ax, xs[3], y, w, h, "Resposta\ndo juiz", fontsize=16, fill=fill, edge=edge)
    d1 = diamond(ax, xs[4], y, dw + 0.15, 2.05, "Desfecho?", fontsize=15.5, edge=edge)

    conn(ax, b1, b2)
    conn(ax, b2, b3)
    conn(ax, b3, b4)
    conn(ax, b4, d1)

    # "ok" e "refused" sao gravados no JSONL de resultados + checkpoint
    # (run_step3.py::run_judge grava os dois com write_result=True); so o
    # erro tecnico fica de fora do checkpoint, indo so para um log de erros
    # separado (step3_errors_{judge_id}.jsonl), sem registro nos resultados.
    ok_y, ref_y, err_y = y + 1.7, y, y - 1.7
    ok = box(ax, xr, ok_y, wr, hr, "Avaliação\nconcluída", fill=GREEN_FILL, edge=GREEN_EDGE, fontsize=15)
    ref = box(ax, xr, ref_y, wr, hr, "Recusa de\nsegurança", fill=ORANGE_FILL, edge=ORANGE_EDGE, fontsize=14)
    err = box(ax, xr, err_y, wr, hr, "Erro\ntécnico", fill=RED_FILL, edge=RED_EDGE, fontsize=15)

    dr = edge_right(d1)
    ax.add_line(Line2D([dr[0], bus_x], [dr[1], dr[1]], color="#333333", linewidth=LW))
    ax.add_line(Line2D([bus_x, bus_x], [ok_y, err_y], color="#333333", linewidth=LW))
    for target, ty in ((ok, ok_y), (ref, ref_y), (err, err_y)):
        ax.add_line(Line2D([bus_x, edge_left(target)[0]], [ty, ty], color="#333333", linewidth=LW))
        arr = FancyArrowPatch((bus_x, ty), edge_left(target), arrowstyle="-|>", mutation_scale=20, linewidth=LW, color="#333333")
        ax.add_patch(arr)
    ax.text(bus_x - 0.15, ok_y + 0.34, "ok", ha="right", va="center", fontsize=14, family=FONT, backgroundcolor="white")
    ax.text(bus_x + 0.15, ref_y + 0.34, "refused", ha="left", va="center", fontsize=14, family=FONT, backgroundcolor="white")
    ax.text(bus_x - 0.15, err_y + 0.34, "error", ha="right", va="center", fontsize=14, family=FONT, backgroundcolor="white")

    # "ok" e "refused" convergem para o mesmo destino (resultados); "erro"
    # segue sozinho, sem barramento, para um destino separado (log).
    res_y = (ok_y + ref_y) / 2
    res = box(ax, xc, res_y, wres, hres, "Registro nos\nresultados\n(JSONL + checkpoint)", fontsize=12.5)
    log = box(ax, xc, err_y, wlog, hlog, "Log de erros\n(sem checkpoint)", fill=GRAY_FILL, edge=GRAY_EDGE, fontsize=12.5)

    for target, ty in ((ok, ok_y), (ref, ref_y)):
        tr = edge_right(target)
        ax.add_line(Line2D([tr[0], bus2_x], [ty, ty], color="#333333", linewidth=LW))
    ax.add_line(Line2D([bus2_x, bus2_x], [ok_y, ref_y], color="#333333", linewidth=LW))
    arr_res = FancyArrowPatch((bus2_x, res_y), edge_left(res), arrowstyle="-|>", mutation_scale=20, linewidth=LW, color="#333333")
    ax.add_patch(arr_res)
    straight(ax, edge_right(err), edge_left(log))

    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT_DIR}/fluxo_etapa3_detalhado.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Exemplo de arvore de chamadas (arvoreFinal.png) - horizontal, raiz a esquerda
# ---------------------------------------------------------------------------
def gen_arvore():
    fig, ax = new_ax(9.6, 6.8)

    root_w, root_h = 2.3, 1.35
    l2_w, l2_h = 2.3, 1.1
    l3_w, l3_h = 2.5, 0.9

    root_x = 0.25 + root_w / 2
    root = box(ax, root_x, 3.4, root_w, root_h, "search_documents()\n(nível 1 — ferramenta)", fontsize=13.5)

    l2_x = root_x + root_w / 2 + 0.8 + l2_w / 2
    l2_y = [5.65, 3.4, 1.15]
    l2_labels = ["validate_query()", "run_vector_search()", "format_results()"]
    l2_boxes = []
    for y, label in zip(l2_y, l2_labels):
        b = box(ax, l2_x, y, l2_w, l2_h, f"{label}\n(nível 2)", fontsize=13)
        l2_boxes.append(b)

    bus1_x = root_x + root_w / 2 + 0.4
    rr = edge_right(root)
    ax.add_line(Line2D([rr[0], bus1_x], [rr[1], rr[1]], color="#333333", linewidth=LW))
    ax.add_line(Line2D([bus1_x, bus1_x], [l2_y[0], l2_y[-1]], color="#333333", linewidth=LW))
    for b in l2_boxes:
        ax.add_line(Line2D([bus1_x, edge_left(b)[0]], [b[1], b[1]], color="#333333", linewidth=LW))
        arr = FancyArrowPatch((bus1_x, b[1]), edge_left(b), arrowstyle="-|>", mutation_scale=20, linewidth=LW, color="#333333")
        ax.add_patch(arr)

    l3_x = l2_x + l2_w / 2 + 0.8 + l3_w / 2
    l3_defs = {
        0: [("check_schema()", False), ("<externo: re.match>", True)],
        1: [("embed_text()", False), ("query_index()", False)],
        2: [("truncate_text()", False)],
    }
    for i, b2 in enumerate(l2_boxes):
        children = l3_defs[i]
        n = len(children)
        bus2_x = l2_x + l2_w / 2 + 0.4
        cys = [b2[1] + (j - (n - 1) / 2) * 1.05 for j in range(n)]
        b2r = edge_right(b2)
        if n == 1:
            straight(ax, b2r, (l3_x - l3_w / 2, cys[0]))
        else:
            ax.add_line(Line2D([b2r[0], bus2_x], [b2r[1], b2r[1]], color="#333333", linewidth=LW))
            ax.add_line(Line2D([bus2_x, bus2_x], [cys[0], cys[-1]], color="#333333", linewidth=LW))
            for cy in cys:
                ax.add_line(Line2D([bus2_x, l3_x - l3_w / 2], [cy, cy], color="#333333", linewidth=LW))
                arr = FancyArrowPatch((bus2_x, cy), (l3_x - l3_w / 2, cy), arrowstyle="-|>", mutation_scale=20, linewidth=LW, color="#333333")
                ax.add_patch(arr)
        for (label, external), cy in zip(children, cys):
            fill = GRAY_FILL if external else GREEN_FILL
            edge = GRAY_EDGE if external else GREEN_EDGE
            box(ax, l3_x, cy, l3_w, l3_h, f"{label}\n(nível 3)", fill=fill, edge=edge,
                fontsize=12.5, dashed=external)

    legend_y = 0.25
    box(ax, 1.4, legend_y, 0.5, 0.32, "", fill=GREEN_FILL, edge=GREEN_EDGE, fontsize=1)
    ax.text(1.8, legend_y, "resolvida no repositório", ha="left", va="center", fontsize=13, family=FONT)
    box(ax, 5.6, legend_y, 0.5, 0.32, "", fill=GRAY_FILL, edge=GRAY_EDGE, fontsize=1, dashed=True)
    ax.text(6.0, legend_y, "externa / não resolvida", ha="left", va="center", fontsize=13, family=FONT)

    fig.tight_layout(pad=0.2)
    fig.savefig(f"{OUT_DIR}/arvoreFinal.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    gen_metodologia()
    gen_etapa1()
    gen_etapa2()
    gen_etapa3()
    gen_arvore()
    print("done")
