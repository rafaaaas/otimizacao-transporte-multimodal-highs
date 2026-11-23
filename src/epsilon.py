# ============================================================
# MÉTODO DA ε-RESTRIÇÃO — FRONTEIRA DE PARETO CUSTO × CO2
# ============================================================
#
# Em vez de pedir pesos ao decisor, resolvemos várias vezes:
#
#     Minimizar  TC
#     Sujeito a  restrições do modelo
#                TCF <= ε
#
# apertando ε do CO2 da solução mais barata até o menor CO2 possível.
# Cada ε dá um ponto (TC, TCF) não dominado: não dá para gastar menos
# sem poluir mais, nem poluir menos sem gastar mais.
#
# Desempate: com TCF <= ε podem existir várias soluções de mesmo TC.
# Depois de achar o TC mínimo, fixamos TC <= TC* e minimizamos TCF,
# para que o ponto seja realmente não dominado.
# ============================================================

import time
import numpy as np
from modelo import avaliar
from solver_highs import resolver
from procedimento import minimizar_criterio


def ponto_epsilon(m, epsilon, **kw):
    """Menor TC com TCF <= ε. Retorna (solucao, valores) ou (None, None) se inviável."""
    limite_co2 = {"coef": m["custos"]["TCF"], "tipo": "L", "rhs": epsilon}
    status, tc, sol, _ = resolver(m, m["custos"]["TC"], linhas_extras=[limite_co2], **kw)
    if sol is None:
        return None, None
    trava_custo = {"coef": m["custos"]["TC"], "tipo": "L", "rhs": tc + 1e-9 * max(1.0, abs(tc))}
    _, _, sol2, _ = resolver(m, m["custos"]["TCF"], linhas_extras=[limite_co2, trava_custo], **kw)
    if sol2 is not None:
        sol = sol2
    return sol, avaliar(m, sol)


def fronteira(m, n_pontos=41, **kw):
    """
    Fronteira de Pareto TC × TCF com n_pontos valores de ε igualmente espaçados.
    Retorna uma lista de dicionários com ε, TC, TCF, ATT, tempo e o custo
    marginal do CO2 (quanto TC sobe por unidade de TCF evitada em relação ao
    ponto anterior).
    """
    _, mais_barata = minimizar_criterio(m, "TC", **kw)   # extremo "barato e poluente"
    _, mais_limpa = minimizar_criterio(m, "TCF", **kw)   # extremo "caro e limpo"
    eps_max = mais_barata["TCF"]
    eps_min = mais_limpa["TCF"]

    pontos = []
    for epsilon in np.linspace(eps_max, eps_min, n_pontos):
        inicio = time.perf_counter()
        sol, valores = ponto_epsilon(m, epsilon + 1e-6, **kw)
        tempo = time.perf_counter() - inicio
        if sol is None:
            continue
        ponto = {"epsilon": float(epsilon), **valores, "tempo_s": tempo}
        if pontos:
            delta_co2 = pontos[-1]["TCF"] - valores["TCF"]
            delta_custo = valores["TC"] - pontos[-1]["TC"]
            ponto["custo_marginal_co2"] = delta_custo / delta_co2 if delta_co2 > 1e-9 else None
        else:
            ponto["custo_marginal_co2"] = None
        pontos.append(ponto)
    return pontos


def custo_na_fronteira(pontos, tcf):
    """Interpola o TC da fronteira num dado TCF (a fronteira de uma PL é linear por partes)."""
    tcfs = np.array([p["TCF"] for p in pontos])[::-1]
    tcs = np.array([p["TC"] for p in pontos])[::-1]
    return float(np.interp(tcf, tcfs, tcs))
