# ============================================================
# PROCEDIMENTO DO ARTIGO (SEÇÃO 4)
# ============================================================
#
#   1. Minimizar cada critério isoladamente (TC, TCF, ATT).
#   2. Tabela payoff: melhor (min) e pior (max) valor de cada critério
#      entre as soluções do passo 1.
#   3. Minimizar a soma ponderada normalizada
#         WS = soma_k  w_k (f_k - f_k_min) / (f_k_max - f_k_min)
#      com os pesos do AHP.
#
# Ótimos alternativos: ao minimizar só TC, podem existir várias soluções
# com o mesmo TC e TCF/ATT diferentes, e a tabela payoff dependeria de qual
# delas o solver devolve. Com lexicografico=True, depois de achar f_k_min
# fixamos f_k <= f_k_min e minimizamos os outros dois critérios
# (normalizados pelo próprio valor mínimo), o que torna a tabela única.
# ============================================================

import numpy as np
from modelo import avaliar
from solver_highs import resolver

CRITERIOS = ("TC", "TCF", "ATT")


def minimizar_criterio(m, k, lexicografico=True, **kw):
    """Minimiza o critério k. Retorna (solucao, valores dos 3 critérios)."""
    status, z, sol, _ = resolver(m, m["custos"][k], **kw)
    if sol is None:
        raise RuntimeError(f"min {k}: {status}")
    if lexicografico:
        outros = [o for o in CRITERIOS if o != k]
        secundario = sum(m["custos"][o] / max(avaliar(m, sol)[o], 1e-9) for o in outros)
        trava = {"coef": m["custos"][k], "tipo": "L", "rhs": z + 1e-9 * max(1.0, abs(z))}
        extras = list(kw.pop("linhas_extras", None) or []) + [trava]
        status, _, sol2, _ = resolver(m, secundario, linhas_extras=extras, **kw)
        if sol2 is not None:
            sol = sol2
    return sol, avaliar(m, sol)


def tabela_payoff(m, lexicografico=True, **kw):
    """Passos 1 e 2: linhas da tabela payoff + melhor/pior de cada critério."""
    linhas = {}
    for k in CRITERIOS:
        _, valores = minimizar_criterio(m, k, lexicografico, **kw)
        linhas[k] = valores
    melhor = {k: linhas[k][k] for k in CRITERIOS}
    pior = {k: max(linhas[o][k] for o in CRITERIOS) for k in CRITERIOS}
    return {"linhas": linhas, "min": melhor, "max": pior}


def custo_ws(m, pesos, payoff):
    """Vetor de custo da soma ponderada normalizada (a constante não altera o ótimo)."""
    custo = np.zeros(m["n_variaveis"])
    for k in CRITERIOS:
        amplitude = payoff["max"][k] - payoff["min"][k]
        if amplitude > 1e-9:
            custo += pesos[k] * m["custos"][k] / amplitude
    return custo


def solucao_ws(m, pesos, payoff, **kw):
    """Passo 3: resolve a WS. Retorna (solucao, valores dos 3 critérios, valor de WS)."""
    status, _, sol, _ = resolver(m, custo_ws(m, pesos, payoff), **kw)
    if sol is None:
        raise RuntimeError(f"WS: {status}")
    valores = avaliar(m, sol)
    ws = sum(
        pesos[k] * (valores[k] - payoff["min"][k]) / (payoff["max"][k] - payoff["min"][k])
        for k in CRITERIOS if payoff["max"][k] - payoff["min"][k] > 1e-9
    )
    return sol, valores, ws


def pesos_rodada(rodada):
    """Converte uma entrada de inst['ahp'] em {'TC': w, 'TCF': w, 'ATT': w}."""
    return {"TC": rodada["w_TC"], "TCF": rodada["w_TCF"], "ATT": rodada["w_ATT"]}
