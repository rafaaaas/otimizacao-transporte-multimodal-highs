# ============================================================
# MODELO DO ARTIGO — PLANEJAMENTO DE TRANSPORTE MULTIMODAL
# ============================================================
#
# Monta o PPL da Seção 3 do artigo (Sustainability 2025, 17, 6327)
# na forma esparsa:
#
#     Minimizar   custo · v
#     Sujeito a   A v (<=, >=, =) b
#                 limite_inf <= v <= limite_sup
#
# A matriz A é a mesma para todos os experimentos. O que muda de
# um experimento para outro é só o vetor de custo (TC, TCF, ATT ou
# uma soma ponderada deles) e, às vezes, linhas extras (ε-restrição).
#
# Ordem das variáveis no vetor v:
#     x[i,j,t]  envio da fábrica i pelo modal j na semana t
#     y[i,t]    produção da fábrica i na semana t
#     z[i,t]    estoque na fábrica i no fim da semana t
#     Z[t]      estoque no centro logístico no fim da semana t
#     S[t]      falta (entrega atrasada) no fim da semana t
#     z0[i]     estoque inicial da fábrica i (só se z0_variavel=True)
#
# Os índices i, j, t no código começam em 0 (semana 1 -> t = 0).
# ============================================================

import json
import numpy as np

INFINITO = float("inf")


def carregar_instancia(caminho):
    """Lê uma instância em JSON (formato de data/artigo.json)."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def montar_modelo(inst, z0=None, z0_variavel=False, capacidade_producao=True):
    """
    Gera o modelo esparso a partir da instância.

    z0                  : estoque inicial de cada fábrica (lista). Se None, usa inst["z0"]
                          ou zero. O artigo não informa esse valor.
    z0_variavel         : se True, z0 vira variável de decisão (usado para calibrar o
                          menor estoque inicial que torna uma solução viável).
    capacidade_producao : se False, remove a restrição (4) y <= p. Serve para testar
                          a hipótese de que o artigo não aplicou essa restrição.
    """
    I, J, T = inst["I"], inst["J"], inst["T"]
    B, h, H, P = inst["B"], inst["h"], inst["H"], inst["P"]
    p, D, tau = inst["p"], inst["D"], inst["tau"]
    c, e = inst["c"], inst["e"]
    if z0 is None:
        z0 = inst.get("z0") or [0.0] * I

    # --- Índices das variáveis ------------------------------

    indice = {}   # ("x", i, j, t) -> posição no vetor v
    nomes = []    # posição -> ("x", i, j, t)

    def nova_variavel(chave):
        indice[chave] = len(nomes)
        nomes.append(chave)

    for i in range(I):
        for j in range(J):
            for t in range(T):
                nova_variavel(("x", i, j, t))
    for i in range(I):
        for t in range(T):
            nova_variavel(("y", i, t))
    for i in range(I):
        for t in range(T):
            nova_variavel(("z", i, t))
    for t in range(T):
        nova_variavel(("Z", t))
    for t in range(T):
        nova_variavel(("S", t))
    if z0_variavel:
        for i in range(I):
            nova_variavel(("z0", i))

    n = len(nomes)

    # --- Limites das variáveis ------------------------------
    # Algumas restrições do artigo viram limites de variável:
    #   (4)  y[i,t] <= p[i,t]
    #   (7)  z[i,T] = 0      (8) Z[T] = 0      (9) S[T] = 0
    #   (11) x[i,j,t] = 0 se o envio chega depois da semana T

    limite_inf = np.zeros(n)
    limite_sup = np.full(n, INFINITO)

    for i in range(I):
        for j in range(J):
            for t in range(T):
                semana = t + 1
                if semana + tau[i][j] > T:                       # (11)
                    limite_sup[indice[("x", i, j, t)]] = 0.0
        if capacidade_producao:
            for t in range(T):
                limite_sup[indice[("y", i, t)]] = p[i][t]        # (4)
        limite_sup[indice[("z", i, T - 1)]] = 0.0                # (7)
    limite_sup[indice[("Z", T - 1)]] = 0.0                       # (8)
    limite_sup[indice[("S", T - 1)]] = 0.0                       # (9)

    # --- Restrições (trios linha / coluna / valor) -----------

    linhas, colunas, valores = [], [], []
    b, tipos, nomes_restricoes = [], [], []

    def nova_restricao(coeficientes, tipo, rhs, nome):
        r = len(b)
        for chave, valor in coeficientes:
            linhas.append(r)
            colunas.append(indice[chave])
            valores.append(valor)
        b.append(rhs)
        tipos.append(tipo)
        nomes_restricoes.append(nome)

    # (5) z[i,t] - z[i,t-1] - y[i,t] + soma_j x[i,j,t] = 0   (em t=1: = z0[i])
    for i in range(I):
        for t in range(T):
            coef = [(("z", i, t), 1.0), (("y", i, t), -1.0)]
            coef += [(("x", i, j, t), 1.0) for j in range(J)]
            rhs = 0.0
            if t > 0:
                coef.append((("z", i, t - 1), -1.0))
            elif z0_variavel:
                coef.append((("z0", i), -1.0))
            else:
                rhs = float(z0[i])
            nova_restricao(coef, "E", rhs, ("estoque_fabrica", i, t))

    # (6) Z[t] - Z[t-1] - (chegadas em t) - S[t] + S[t-1] = -D[t]
    #     O envio feito na semana s pelo modal j chega na semana s + tau[i][j].
    for t in range(T):
        coef = [(("Z", t), 1.0), (("S", t), -1.0)]
        if t > 0:
            coef += [(("Z", t - 1), -1.0), (("S", t - 1), 1.0)]
        for i in range(I):
            for j in range(J):
                s = t - tau[i][j]
                if s >= 0:
                    coef.append((("x", i, j, s), -1.0))
        nova_restricao(coef, "E", -float(D[t]), ("balanco_centro", t))

    # (10) soma_i x[i,j,t] <= B[j]
    for j in range(J):
        for t in range(T):
            coef = [(("x", i, j, t), 1.0) for i in range(I)]
            nova_restricao(coef, "L", float(B[j]), ("capacidade_modal", j, t))

    # --- Vetores de custo dos 3 critérios --------------------

    custo_TC = np.zeros(n)
    custo_TCF = np.zeros(n)
    custo_ATT = np.zeros(n)
    demanda_total = float(sum(D))

    for i in range(I):
        for j in range(J):
            for t in range(T):
                k = indice[("x", i, j, t)]
                custo_TC[k] = c[i][j]                          # frete
                custo_TCF[k] = e[i][j]                         # CO2
                custo_ATT[k] = tau[i][j] / demanda_total       # tempo médio
        for t in range(T):
            custo_TC[indice[("z", i, t)]] = h[i]               # estoque na fábrica
    for t in range(T):
        custo_TC[indice[("Z", t)]] = H                         # estoque no centro
        custo_TC[indice[("S", t)]] = P                         # atraso

    return {
        "inst": inst,
        "n_variaveis": n,
        "n_restricoes": len(b),
        "indice": indice,
        "nomes": nomes,
        "nomes_restricoes": nomes_restricoes,
        "linhas": np.array(linhas, dtype=np.int32),
        "colunas": np.array(colunas, dtype=np.int32),
        "valores": np.array(valores, dtype=np.double),
        "b": np.array(b, dtype=np.double),
        "tipos": tipos,
        "limite_inf": limite_inf,
        "limite_sup": limite_sup,
        "custos": {"TC": custo_TC, "TCF": custo_TCF, "ATT": custo_ATT},
    }


def contar(m):
    """Tamanho do modelo: total de variáveis e quantas não estão fixadas em zero."""
    livres = int(np.sum(m["limite_sup"] > 0))
    return {
        "variaveis": m["n_variaveis"],
        "variaveis_livres": livres,
        "restricoes": m["n_restricoes"],
        "nao_nulos": len(m["valores"]),
    }


def avaliar(m, sol):
    """Valor dos 3 critérios numa solução."""
    return {k: float(m["custos"][k] @ sol) for k in ("TC", "TCF", "ATT")}


def decompor_TC(m, sol):
    """Separa o custo total em frete, estoque na fábrica, estoque no centro e atraso."""
    partes = {"frete": 0.0, "estoque_fabrica": 0.0, "estoque_centro": 0.0, "atraso": 0.0}
    grupo = {"x": "frete", "z": "estoque_fabrica", "Z": "estoque_centro", "S": "atraso"}
    for k, chave in enumerate(m["nomes"]):
        if chave[0] in grupo:
            partes[grupo[chave[0]]] += m["custos"]["TC"][k] * sol[k]
    return partes


def tabela_envios(m, sol):
    """Envios no formato da Tabela 7: matriz [i][j][t]."""
    inst = m["inst"]
    x = np.zeros((inst["I"], inst["J"], inst["T"]))
    for i in range(inst["I"]):
        for j in range(inst["J"]):
            for t in range(inst["T"]):
                x[i, j, t] = sol[m["indice"][("x", i, j, t)]]
    return x


def serie(m, sol, nome, *prefixo):
    """Extrai uma série temporal, ex.: serie(m, sol, 'y', 0) -> produção da fábrica 1."""
    T = m["inst"]["T"]
    return np.array([sol[m["indice"][(nome, *prefixo, t)]] for t in range(T)])


def fixar_envios(m, x):
    """Devolve limites que fixam todos os x[i,j,t] nos valores dados (ex.: Tabela 7)."""
    limite_inf = m["limite_inf"].copy()
    limite_sup = m["limite_sup"].copy()
    inst = m["inst"]
    for i in range(inst["I"]):
        for j in range(inst["J"]):
            for t in range(inst["T"]):
                k = m["indice"][("x", i, j, t)]
                limite_inf[k] = limite_sup[k] = x[i][j][t]
    return limite_inf, limite_sup
