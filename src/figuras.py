# ============================================================
# FIGURAS PARA O RELATÓRIO E OS SLIDES
# ============================================================
# Figuras estáticas (PNG, 200 dpi), fundo claro, pensadas para impressão.
# Cores: paleta categórica de referência (slot 1 azul, slot 2 laranja);
# texto sempre em tinta neutra, nunca na cor da série.
# ============================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

AZUL = "#2a78d6"
LARANJA = "#eb6834"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e4e3df"
FUNDO = "#fcfcfb"


def _eixos(ax, escala, unidade):
    def rotulo(v, _):
        x = v / escala
        texto = f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.1f}"
        return texto.replace(",", "X").replace(".", ",").replace("X", ".")
    fmt = FuncFormatter(rotulo)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(f"Custo total TC ({unidade})", color=TINTA_2)
    ax.set_ylabel(f"Emissão de CO$_2$ TCF ({unidade})", color=TINTA_2)
    ax.grid(True, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2)
    ax.set_facecolor(FUNDO)


def fronteira_artigo(fronteira_c, fronteira_a, rodadas_artigo, rodadas_completo, caminho):
    """
    fronteira_c       : pontos do modelo como o artigo resolveu (sem a restrição 4)
    fronteira_a       : pontos do modelo completo
    rodadas_artigo    : [(rótulo, TC, TCF), ...] soluções publicadas
    rodadas_completo  : [(rótulo, TC, TCF), ...] soluções WS no modelo completo
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=FUNDO)
    ax.plot([p["TC"] for p in fronteira_c], [p["TCF"] for p in fronteira_c], color=AZUL, lw=2,
            marker="o", ms=3.5, label="Modelo como resolvido no artigo (sem a restrição 4)")
    ax.plot([p["TC"] for p in fronteira_a], [p["TCF"] for p in fronteira_a], color=LARANJA, lw=2,
            marker="o", ms=3.5, label="Modelo completo (com a restrição 4)")

    ax.scatter([r[1] for r in rodadas_artigo], [r[2] for r in rodadas_artigo], s=70, facecolor=FUNDO,
               edgecolor=TINTA, linewidth=1.6, zorder=5, label="Soluções publicadas (rodadas 1–3)")
    deslocamentos = {"Rodada 1": (10, 8), "Rodada 2": (10, 6), "Rodada 3": (10, -16)}
    for rotulo, tc, tcf in rodadas_artigo:
        ax.annotate(rotulo, (tc, tcf), textcoords="offset points", xytext=deslocamentos.get(rotulo, (8, 6)),
                    color=TINTA, fontsize=9)

    ax.scatter([r[1] for r in rodadas_completo], [r[2] for r in rodadas_completo], s=70, marker="D",
               facecolor=FUNDO, edgecolor=LARANJA, linewidth=1.6, zorder=5,
               label="Rodadas 1–3 no modelo completo")

    _eixos(ax, 1000, "mil")
    ax.set_title("Fronteira de Pareto custo × CO$_2$ (instância do artigo)", color=TINTA, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
              labelcolor=TINTA)
    fig.set_size_inches(7.5, 5.6)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, facecolor=FUNDO, bbox_inches="tight")
    plt.close(fig)
    return caminho


def fronteira_grande(pontos, rodadas, caminho, titulo):
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=FUNDO)
    ax.plot([p["TC"] for p in pontos], [p["TCF"] for p in pontos], color=AZUL, lw=2, marker="o", ms=3.5,
            label="Fronteira de Pareto (ε-restrição)")
    ax.scatter([r[1] for r in rodadas], [r[2] for r in rodadas], s=70, facecolor=FUNDO, edgecolor=TINTA,
               linewidth=1.6, zorder=5, label="Soma ponderada com os pesos do artigo")
    for rotulo, tc, tcf in rodadas:
        ax.annotate(rotulo, (tc, tcf), textcoords="offset points", xytext=(10, 6), color=TINTA, fontsize=9)
    _eixos(ax, 1e6, "milhões")
    ax.set_title(titulo, color=TINTA, loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200, facecolor=FUNDO, bbox_inches="tight")
    plt.close(fig)
    return caminho
