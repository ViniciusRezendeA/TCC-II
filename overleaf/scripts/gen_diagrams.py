"""Gera os diagramas do TCC (visao geral, Etapas 1/2/3, exemplo de call graph)
em layout horizontal, com texto reduzido a titulos curtos, salvando PNG em
300dpi direto em overleaf/images/.

Roda com: uv run python overleaf/scripts/gen_diagrams.py
(ou `python3 gen_diagrams.py` de dentro de overleaf/scripts/, com matplotlib
instalado -- ja e dependencia do projeto, usada por scripts/analysis_report.py).

Fontes grandes e caixas justas ao texto (pouco preenchimento interno), porque
a figura final entra no PDF encolhida para a largura da coluna (~5.9in, A4
com margens de 3cm) -- se a figura-fonte for muito larga ou tiver caixas
maiores que o texto, tudo encolhe junto e fica ilegivel/com espaco vazio.
Conectores usam cotovelo reto (horizontal-vertical-horizontal) em vez de
diagonal sempre que os dois pontos nao estao alinhados em x ou em y, e as
caixas ficam espacadas o bastante para a linha do conector aparecer (nao so
a ponta da seta encostada na proxima caixa). O tamanho da figura nao e
calculado a mao: cada `gen_*` so posiciona os elementos, e `savefig(...,
bbox_inches="tight")` corta a figura para caber exatamente no conteudo.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "images")
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
ORANGE_EDGE = "#c58a2b"
DIAMOND_FILL = "#eaf3fd"

# Uma cor por etapa da metodologia (Figura 1), reaproveitada no diagrama
# detalhado de cada etapa para deixar visualmente explicito a qual etapa
# geral cada fluxograma detalhado pertence. As cores semanticas (verde =
# sucesso, vermelho = erro/desqualificado, laranja = backfill/recusa, cinza
# = descartado/externo/log) continuam reservadas para esses significados e
# nao sao reusadas aqui.
STEP_COLETA = ("#dbe8fb", "#2b5fad")         # azul
STEP_EXTRACAO = ("#ecdff5", "#7b3fa0")       # roxo
STEP_CLASSIFICACAO = ("#d7f2ef", "#0f7a72")  # verde-azulado
STEP_CONSOLIDACAO = ("#e4e8f0", "#4a5a78")   # azul-acinzentado
STEP_ANALISE = ("#f8dde4", "#a83f5a")        # rose

FONT = "DejaVu Sans"
LW = 2.0

# Espacamento padrao entre caixas vizinhas na mesma fileira/coluna, grande o
# bastante para o segmento de linha do conector ficar visivel entre a caixa
# e a ponta da seta (nao so a seta encostada na caixa seguinte).
GAP = 0.7


def box(ax, x, y, w, h, text, fill=BLUE_FILL, edge=BLUE_EDGE, fontsize=16, dashed=False, textcolor="#1a1a1a"):
    style = "round,pad=0.02,rounding_size=0.06"
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=style, linewidth=LW,
        facecolor=fill, edgecolor=edge,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, family=FONT, linespacing=1.15)
    return (x, y, w, h)


def diamond(ax, x, y, w, h, text, fontsize=15, edge=BLUE_EDGE):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    poly = Polygon(pts, closed=True, facecolor=DIAMOND_FILL, edgecolor=edge, linewidth=LW)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, family=FONT, linespacing=1.1)
    return (x, y, w, h)


def straight(ax, p1, p2, label=None, color="#333333", label_dx=0, label_dy=0.22, ls="-", fontsize=13):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=18, linewidth=LW,
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
    arr = FancyArrowPatch(corner2, p2, arrowstyle="-|>", mutation_scale=18, linewidth=lw, color=color, linestyle=ls)
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


def new_ax():
    """Eixo sem limites pre-calculados: cada elemento adicionado (add_patch/
    add_line) atualiza os limites de dados automaticamente, e o savefig com
    bbox_inches='tight' corta a figura exatamente no conteudo no final. Isso
    evita ter que calcular fig_w/fig_h/margens a mao toda vez que uma caixa
    muda de tamanho.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    return fig, ax


def save(fig, name, scale=1.05):
    """Reescala a figura antes de salvar, usando um fator fixo de polegadas
    por unidade de dado (nao uma largura total fixa).

    figsize comeca generico (new_ax); cada diagrama tem uma extensao de
    dados bem diferente (metodologia e uma fileira so, larga e baixa;
    etapa2 tem 3 fileiras, mais quadrada). Escalar para uma LARGURA TOTAL
    fixa faz diagramas de fileira unica (dados baixos, largos) encolherem
    demais em altura fisica (polegadas) -- a fonte, que tem tamanho fixo em
    pontos, passa a ocupar proporcionalmente mais da caixa e estoura a
    borda. Escalar por um fator fixo de pol/unidade mantem o tamanho fisico
    de uma caixa (e da fonte relativa a ela) igual em qualquer diagrama,
    independente de quantas caixas cabem ao lado.
    """
    ax = fig.axes[0]
    ax.relim()
    ax.autoscale_view()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    data_w = max(xmax - xmin, 1e-6)
    data_h = max(ymax - ymin, 1e-6)
    fig.set_size_inches(data_w * scale, data_h * scale)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


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
    w = [2.3, 2.7, 3.1, 2.5, 2.3]
    h = 1.3
    xs = layout_row(0, GAP, w)
    fig, ax = new_ax()
    y = 0
    b1 = box(ax, xs[0], y, w[0], h, "1. Coleta dos\nRepositórios MCP", fontsize=13.5,
             fill=STEP_COLETA[0], edge=STEP_COLETA[1])
    b2 = box(ax, xs[1], y, w[1], h, "2. Extração das\nFerramentas e Código", fontsize=13,
             fill=STEP_EXTRACAO[0], edge=STEP_EXTRACAO[1])
    b3 = box(ax, xs[2], y, w[2], h, "3. Classificação\n2 cenários × múltiplos\nmodelos de IA (juízes)", fontsize=11.5,
             fill=STEP_CLASSIFICACAO[0], edge=STEP_CLASSIFICACAO[1])
    b4 = box(ax, xs[3], y, w[3], h, "4. Consolidação\ndos Resultados", fontsize=13,
             fill=STEP_CONSOLIDACAO[0], edge=STEP_CONSOLIDACAO[1])
    b5 = box(ax, xs[4], y, w[4], h, "5. Análise\nComparativa", fontsize=13.5,
             fill=STEP_ANALISE[0], edge=STEP_ANALISE[1])
    conn(ax, b1, b2)
    conn(ax, b2, b3)
    conn(ax, b3, b4)
    conn(ax, b4, b5)
    save(fig, "metodologia_v7.png")


# ---------------------------------------------------------------------------
# 2. Etapa 1 (fluxo_etapa1_detalhado.png) -- layout em 2 fileiras (cotovelo):
#   fileira 1: 1 -> 2 -> 3
#   fileira 2: 5 <- 4 <- (sim) <- Elegivel? -(nao)-> Descartado
# ---------------------------------------------------------------------------
def gen_etapa1():
    fill, edge = STEP_COLETA
    w, h = 2.3, 1.25
    dw, dh = 2.1, 1.85
    w5 = 2.0
    wdesc, hdesc = 1.8, 0.8
    vgap = 0.75

    row2_y = 0
    row1_y = row2_y + dh / 2 + vgap + h / 2

    x1 = 0
    x2 = x1 + w + GAP
    x3 = x2 + w + GAP

    xd = x3  # losango alinhado sob a caixa 3
    x4 = xd - dw / 2 - GAP - w / 2
    x5 = x4 - w / 2 - GAP - w5 / 2
    xdesc = xd + dw / 2 + GAP + wdesc / 2

    fig, ax = new_ax()
    b1 = box(ax, x1, row1_y, w, h, "1. Mineração via\nREST Code Search", fontsize=13.5, fill=fill, edge=edge)
    b2 = box(ax, x2, row1_y, w, h, "2. Hidratação\nvia GraphQL", fontsize=14.5, fill=fill, edge=edge)
    b3 = box(ax, x3, row1_y, w, h, "3. Deduplicação\npor identificador", fontsize=13, fill=fill, edge=edge)
    d1 = diamond(ax, xd, row2_y, dw, dh, "Elegível?", fontsize=14.5, edge=edge)
    b4 = box(ax, x4, row2_y, w, h, "4. Ordenação\npor estrelas", fontsize=13.5, fill=fill, edge=edge)
    b5 = box(ax, x5, row2_y, w5, h, "5. Seleção\ntop-N", fontsize=14.5, fill=fill, edge=edge)
    desc = box(ax, xdesc, row2_y, wdesc, hdesc, "Descartado", fill=GRAY_FILL, edge=GRAY_EDGE, fontsize=13, dashed=True)

    straight(ax, edge_right(b1), edge_left(b2), fontsize=13.5)
    straight(ax, edge_right(b2), edge_left(b3), fontsize=13.5)
    elbow(ax, edge_bottom(b3), edge_top(d1), bend="y", fontsize=13.5)
    straight(ax, edge_left(d1), edge_right(b4), label="sim", label_dy=1.05, fontsize=13.5)
    straight(ax, edge_left(b4), edge_right(b5), fontsize=13.5)
    straight(ax, edge_right(d1), edge_left(desc), label="não", label_dy=0.4, fontsize=13.5)

    save(fig, "fluxo_etapa1_detalhado.png")


# ---------------------------------------------------------------------------
# 3. Etapa 2 (fluxo_etapa2_detalhado.png) -- layout em 3 fileiras (cotovelo):
#   fileira 1: 1 -> 2 -> 3
#   fileira 2: backfill <- Desqualificado <- (nao) <- >=1 ferramenta? (sim desce)
#   fileira 3: 6 <- 5 <- 4
# ---------------------------------------------------------------------------
def gen_etapa2():
    fill, edge = STEP_EXTRACAO
    w, h = 2.2, 1.25
    dw, dh = 2.0, 1.8
    wdesq, hdesq = 1.9, 0.8
    wbf, hbf = 2.15, 0.85
    w6 = 1.9
    vgap = 0.75

    row3_y = 0
    row2_y = row3_y + h / 2 + vgap + dh / 2
    row1_y = row2_y + dh / 2 + vgap + h / 2

    x1 = 0
    x2 = x1 + w + GAP
    x3 = x2 + w + GAP
    xd = x3

    x_desq = xd - dw / 2 - GAP - wdesq / 2
    x_bf = x1

    x4 = xd
    x5 = x4 - w - GAP
    x6 = x5 - w / 2 - GAP - w6 / 2

    fig, ax = new_ax()
    b1 = box(ax, x1, row1_y, w, h, "1. Clonagem\nrasa", fontsize=14.5, fill=fill, edge=edge)
    b2 = box(ax, x2, row1_y, w, h, "2. Indexação\nestática", fontsize=14.5, fill=fill, edge=edge)
    b3 = box(ax, x3, row1_y, w, h, "3. Detecção de\nferramentas", fontsize=13, fill=fill, edge=edge)
    d1 = diamond(ax, xd, row2_y, dw, dh, "≥1\nferramenta?", fontsize=13, edge=edge)
    desq = box(ax, x_desq, row2_y, wdesq, hdesq, "Desqualificado", fill=RED_FILL, edge=RED_EDGE, fontsize=12.5)
    bf = box(ax, x_bf, row2_y, wbf, hbf, "Próximo candidato\ndo pool (backfill)", fill=ORANGE_FILL, edge=ORANGE_EDGE, fontsize=11.5)
    b4 = box(ax, x4, row3_y, w, h, "4. Grafo de\nchamadas", fontsize=13.5, fill=fill, edge=edge)
    b5 = box(ax, x5, row3_y, w, h, "5. Classificação\nde chamadas", fontsize=12.5, fill=fill, edge=edge)
    b6 = box(ax, x6, row3_y, w6, h, "6. Consolidação\nem tools.jsonl", fontsize=11.5, fill=fill, edge=edge)

    straight(ax, edge_right(b1), edge_left(b2), fontsize=13.5)
    straight(ax, edge_right(b2), edge_left(b3), fontsize=13.5)
    elbow(ax, edge_bottom(b3), edge_top(d1), bend="y", fontsize=13.5)
    straight(ax, edge_left(d1), edge_right(desq), label="não", label_dy=0.85, fontsize=13)
    straight(ax, edge_left(desq), edge_right(bf), fontsize=13)
    straight(ax, edge_top(bf), edge_bottom(b1), label="re-entra", label_dx=0.55, color="#555555", ls="--", fontsize=12.5)
    elbow(ax, edge_bottom(d1), edge_top(b4), bend="y", label="sim", label_dx=0.5, fontsize=13)
    straight(ax, edge_left(b4), edge_right(b5), fontsize=13.5)
    straight(ax, edge_left(b5), edge_right(b6), fontsize=13.5)

    save(fig, "fluxo_etapa2_detalhado.png")


# ---------------------------------------------------------------------------
# 4. Etapa 3 (fluxo_etapa3_detalhado.png)
# ---------------------------------------------------------------------------
def gen_etapa3():
    """3 fileiras, de cima para baixo, com distancia minima `vgap` entre elas:
      fileira 1: Montagem -> Payload -> Envio -> Resposta (a cadeia principal)
      fileira 2: Desfecho? (losango) com os 3 desfechos (ok/refused/error) em leque a direita
      fileira 3: os 2 destinos (Registro nos resultados / Log de erros), abaixo da fileira 2
    """
    fill, edge = STEP_CLASSIFICACAO
    w, h = 2.15, 1.25
    dw, dh = 1.9, 1.85
    wr, hr = 2.1, 0.9
    wres, hres = 2.3, 1.35
    wlog, hlog = 2.1, 0.85
    stub = 0.65
    vgap = 0.8
    spread = 1.15  # espacamento vertical entre os 3 desfechos, dentro da fileira 2 (> hr, sem sobrepor)

    xs = layout_row(0, GAP, [w, w, w, w])
    row1_y = 0
    xd = xs[-1]  # losango alinhado sob a ultima caixa da fileira 1 (Resposta)

    row2_y = row1_y - h / 2 - vgap - dh / 2
    ok_y, ref_y, err_y = row2_y + spread, row2_y, row2_y - spread

    row3_y = (err_y - hr / 2) - vgap - max(hres, hlog) / 2

    xr = xd + dw / 2 + stub + wr / 2
    x_res = xr - 1.3
    x_log = xr + 1.3

    fig, ax = new_ax()
    b1 = box(ax, xs[0], row1_y, w, h, "Montagem\ndos 2 cenários", fontsize=13, fill=fill, edge=edge)
    b2 = box(ax, xs[1], row1_y, w, h, "Payload com\ndescrição + código", fontsize=11.5, fill=fill, edge=edge)
    b3 = box(ax, xs[2], row1_y, w, h, "Envio concorrente\naos juízes", fontsize=12, fill=fill, edge=edge)
    b4 = box(ax, xs[3], row1_y, w, h, "Resposta\ndo juiz", fontsize=14, fill=fill, edge=edge)
    conn(ax, b1, b2)
    conn(ax, b2, b3)
    conn(ax, b3, b4)

    d1 = diamond(ax, xd, row2_y, dw, dh, "Desfecho?", fontsize=13.5, edge=edge)
    elbow(ax, edge_bottom(b4), edge_top(d1), bend="y", fontsize=13)

    # "ok" e "refused" sao gravados no JSONL de resultados + checkpoint; so o
    # erro tecnico fica de fora do checkpoint, indo so para um log de erros
    # separado, sem registro nos resultados.
    ok = box(ax, xr, ok_y, wr, hr, "Avaliação\nconcluída", fill=GREEN_FILL, edge=GREEN_EDGE, fontsize=13)
    ref = box(ax, xr, ref_y, wr, hr, "Recusa de\nsegurança", fill=ORANGE_FILL, edge=ORANGE_EDGE, fontsize=12.5)
    err = box(ax, xr, err_y, wr, hr, "Erro\ntécnico", fill=RED_FILL, edge=RED_EDGE, fontsize=13)

    dr = edge_right(d1)
    ax.add_line(Line2D([dr[0], xr - wr / 2 - 0.35], [dr[1], dr[1]], color="#333333", linewidth=LW))
    bus_x = xr - wr / 2 - 0.35
    ax.add_line(Line2D([bus_x, bus_x], [ok_y, err_y], color="#333333", linewidth=LW))
    for target, ty in ((ok, ok_y), (ref, ref_y), (err, err_y)):
        ax.add_line(Line2D([bus_x, edge_left(target)[0]], [ty, ty], color="#333333", linewidth=LW))
        arr = FancyArrowPatch((bus_x, ty), edge_left(target), arrowstyle="-|>", mutation_scale=18, linewidth=LW, color="#333333")
        ax.add_patch(arr)
    ax.text(bus_x - 0.15, ok_y + 0.32, "ok", ha="right", va="center", fontsize=12, family=FONT, backgroundcolor="white")
    ax.text(bus_x - 0.15, ref_y + 0.32, "refused", ha="right", va="center", fontsize=12, family=FONT, backgroundcolor="white")
    ax.text(bus_x - 0.15, err_y + 0.32, "error", ha="right", va="center", fontsize=12, family=FONT, backgroundcolor="white")

    # "ok" e "refused" convergem para o mesmo destino (resultados), na fileira
    # 3; "erro" desce sozinho para um destino separado (log), tambem na
    # fileira 3. "ok"/"refused" saem pela direita das caixas antes de descer
    # -- descer reto (pela mesma coluna x das 3 caixas) cruzaria por dentro
    # de "Recusa de seguranca"/"Erro tecnico", que ficam abaixo na mesma
    # coluna. So depois de sair da coluna e ja abaixo de todas as caixas
    # (clear_y) e que a linha pode virar para a esquerda, ate a coluna de
    # "erro" (que nao tem nada abaixo dela) pode descer reto.
    res = box(ax, x_res, row3_y, wres, hres, "Registro nos\nresultados\n(JSONL + checkpoint)", fontsize=11)
    log = box(ax, x_log, row3_y, wlog, hlog, "Log de erros\n(sem checkpoint)", fill=GRAY_FILL, edge=GRAY_EDGE, fontsize=11)

    exit_x = xr + wr / 2 + 0.35
    clear_y = err_y - hr / 2 - 0.3
    for target, ty in ((ok, ok_y), (ref, ref_y)):
        tr = edge_right(target)
        ax.add_line(Line2D([tr[0], exit_x], [ty, ty], color="#333333", linewidth=LW))
    ax.add_line(Line2D([exit_x, exit_x], [ok_y, clear_y], color="#333333", linewidth=LW))
    ax.add_line(Line2D([exit_x, x_res], [clear_y, clear_y], color="#333333", linewidth=LW))
    arr_res = FancyArrowPatch((x_res, clear_y), edge_top(res), arrowstyle="-|>", mutation_scale=18, linewidth=LW, color="#333333")
    ax.add_patch(arr_res)

    elbow(ax, edge_bottom(err), edge_top(log), bend="y", at=clear_y)

    save(fig, "fluxo_etapa3_detalhado.png")


# ---------------------------------------------------------------------------
# 5. Exemplo de arvore de chamadas (arvoreFinal.png) - horizontal, raiz a esquerda
# ---------------------------------------------------------------------------
def gen_arvore():
    # Raiz com mais respiro entre texto e borda (era justa demais). Um dos
    # tres galhos de nivel 2 termina sem resolucao (externo), deixando claro
    # que uma chamada pode ficar sem resolver em QUALQUER nivel da arvore,
    # nao so no nivel 3 -- os outros dois galhos seguem ate o nivel 3 normal.
    root_w, root_h = 2.5, 1.4
    l2_w, l2_h = 2.2, 0.95
    l3_w, l3_h = 2.2, 0.7
    col_gap = 1.0
    bus_stub = 0.5

    root_x = 0
    fig, ax = new_ax()
    root = box(ax, root_x, 0, root_w, root_h, "search_documents()\n(nível 1 — ferramenta)", fontsize=12.5)

    l2_x = root_x + root_w / 2 + col_gap + l2_w / 2
    l2_y = [1.9, 0, -1.9]
    l2_defs = [
        ("validate_query()", False),
        ("run_vector_search()", False),
        ("<externo: json.dumps>", True),
    ]
    l2_boxes = []
    for y, (label, external) in zip(l2_y, l2_defs):
        fill = GRAY_FILL if external else BLUE_FILL
        edge = GRAY_EDGE if external else BLUE_EDGE
        b = box(ax, l2_x, y, l2_w, l2_h, f"{label}\n(nível 2)", fill=fill, edge=edge,
                fontsize=10.5, dashed=external)
        l2_boxes.append(b)

    bus1_x = root_x + root_w / 2 + bus_stub
    rr = edge_right(root)
    ax.add_line(Line2D([rr[0], bus1_x], [rr[1], rr[1]], color="#333333", linewidth=LW))
    ax.add_line(Line2D([bus1_x, bus1_x], [l2_y[0], l2_y[-1]], color="#333333", linewidth=LW))
    for b in l2_boxes:
        ax.add_line(Line2D([bus1_x, edge_left(b)[0]], [b[1], b[1]], color="#333333", linewidth=LW))
        arr = FancyArrowPatch((bus1_x, b[1]), edge_left(b), arrowstyle="-|>", mutation_scale=18, linewidth=LW, color="#333333")
        ax.add_patch(arr)

    l3_x = l2_x + l2_w / 2 + col_gap + l3_w / 2
    # Sem entrada para o indice 2: o galho externo de nivel 2 nao tem filhos.
    l3_defs = {
        0: [("check_schema()", False), ("<externo: re.match>", True)],
        1: [("embed_text()", False), ("query_index()", False)],
    }
    for i, b2 in enumerate(l2_boxes):
        children = l3_defs.get(i)
        if children is None:
            continue
        n = len(children)
        bus2_x = l2_x + l2_w / 2 + bus_stub
        cys = [b2[1] + (j - (n - 1) / 2) * 0.95 for j in range(n)]
        b2r = edge_right(b2)
        if n == 1:
            straight(ax, b2r, (l3_x - l3_w / 2, cys[0]))
        else:
            ax.add_line(Line2D([b2r[0], bus2_x], [b2r[1], b2r[1]], color="#333333", linewidth=LW))
            ax.add_line(Line2D([bus2_x, bus2_x], [cys[0], cys[-1]], color="#333333", linewidth=LW))
            for cy in cys:
                ax.add_line(Line2D([bus2_x, l3_x - l3_w / 2], [cy, cy], color="#333333", linewidth=LW))
                arr = FancyArrowPatch((bus2_x, cy), (l3_x - l3_w / 2, cy), arrowstyle="-|>", mutation_scale=18, linewidth=LW, color="#333333")
                ax.add_patch(arr)
        for (label, external), cy in zip(children, cys):
            fill = GRAY_FILL if external else GREEN_FILL
            edge = GRAY_EDGE if external else GREEN_EDGE
            box(ax, l3_x, cy, l3_w, l3_h, f"{label}\n(nível 3)", fill=fill, edge=edge,
                fontsize=10.5, dashed=external)

    legend_y = -3.0
    box(ax, root_x + 1.1, legend_y, 0.45, 0.28, "", fill=GREEN_FILL, edge=GREEN_EDGE, fontsize=1)
    ax.text(root_x + 1.45, legend_y, "resolvida no repositório", ha="left", va="center", fontsize=11, family=FONT)
    box(ax, root_x + 5.1, legend_y, 0.45, 0.28, "", fill=GRAY_FILL, edge=GRAY_EDGE, fontsize=1, dashed=True)
    ax.text(root_x + 5.45, legend_y, "externa / não resolvida", ha="left", va="center", fontsize=11, family=FONT)

    save(fig, "arvoreFinal.png")


if __name__ == "__main__":
    gen_metodologia()
    gen_etapa1()
    gen_etapa2()
    gen_etapa3()
    gen_arvore()
    print("done")
