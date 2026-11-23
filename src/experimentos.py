# ============================================================
# EXPERIMENTOS
# ============================================================
#
# Uso (a partir da pasta p2):
#     python src/experimentos.py e1
#
# E1: reproduz as Tabelas 4–7 do artigo em três variantes do modelo:
#   A  modelo completo, estoque inicial z0 = 0
#   B  modelo completo, z0 calibrado (menor z0 que torna a Tabela 7 viável)
#   C  sem a restrição de capacidade de produção (4), z0 = 0
#      (hipótese: o artigo não aplicou a restrição (4))
# ============================================================

import csv
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, AQUI)

from modelo import (carregar_instancia, montar_modelo, contar, avaliar, decompor_TC,
                    tabela_envios, fixar_envios)
from solver_highs import resolver
from procedimento import CRITERIOS, tabela_payoff, solucao_ws, pesos_rodada

PASTA_RESULTADOS = os.path.join(RAIZ, "resultados")


def salvar_csv(nome, cabecalho, linhas):
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)
    caminho = os.path.join(PASTA_RESULTADOS, nome)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(cabecalho)
        w.writerows(linhas)
    return caminho


def calibrar_z0(inst, x):
    """Menor estoque inicial (soma) que torna os envios x viáveis."""
    m = montar_modelo(inst, z0_variavel=True)
    li, ls = fixar_envios(m, x)
    custo = np.zeros(m["n_variaveis"])
    for i in range(inst["I"]):
        custo[m["indice"][("z0", i)]] = 1.0
    status, _, sol, _ = resolver(m, custo, limite_inf=li, limite_sup=ls)
    if sol is None:
        return None
    return [round(float(sol[m["indice"][("z0", i)]]), 6) for i in range(inst["I"])]


def checar_tabela7(m, x):
    """Fixa os envios da Tabela 7 e verifica se são viáveis; se sim, devolve os critérios
    com o menor custo de estoque/atraso compatível com esses envios."""
    li, ls = fixar_envios(m, x)
    status, _, sol, _ = resolver(m, m["custos"]["TC"], limite_inf=li, limite_sup=ls)
    if sol is None:
        return status, None, None
    return status, avaliar(m, sol), decompor_TC(m, sol)


def fmt(v, k):
    return f"{v:10.3f}" if k == "ATT" else f"{v:10.0f}"


def e1():
    inst = carregar_instancia(os.path.join(RAIZ, "data", "artigo.json"))
    alvos = inst["alvos"]
    x7 = inst["tabela7"]["x"]

    z0_cal = calibrar_z0(inst, x7)
    print(f"z0 mínimo para a Tabela 7 ser viável (com capacidade de produção): {z0_cal}")

    variantes = {
        "A_completo_z0_0": montar_modelo(inst, z0=[0] * inst["I"]),
        "B_completo_z0_calibrado": montar_modelo(inst, z0=z0_cal),
        "C_sem_cap_producao": montar_modelo(inst, z0=[0] * inst["I"], capacidade_producao=False),
    }

    linhas_csv = []
    for nome, m in variantes.items():
        print("\n" + "=" * 70)
        print(f"Variante {nome}   {contar(m)}")
        print("=" * 70)

        payoff = tabela_payoff(m)
        print("Tabela payoff (linha = critério minimizado):")
        for k in CRITERIOS:
            v = payoff["linhas"][k]
            print(f"   min {k:4s}: " + "  ".join(fmt(v[c], c) for c in CRITERIOS))

        resultados = {"min": payoff["min"], "max": payoff["max"]}
        for r in inst["ahp"]:
            _, valores, ws = solucao_ws(m, pesos_rodada(r), payoff)
            resultados[f"rodada{r['rodada']}"] = valores

        st7, val7, dec7 = checar_tabela7(m, x7)
        if val7:
            resultados["tabela7"] = val7

        print(f"\n{'linha':10s} | " + " | ".join(f"{c:>10s} {'artigo':>10s}" for c in CRITERIOS))
        for linha, valores in resultados.items():
            alvo = alvos.get(linha, {"TC": np.nan, "TCF": np.nan, "ATT": np.nan}) if linha != "tabela7" else alvos["rodada3"]
            print(f"{linha:10s} | " + " | ".join(f"{fmt(valores[c], c)} {fmt(alvo[c], c)}" for c in CRITERIOS))
            linhas_csv.append([nome, linha] + [round(valores[c], 4) for c in CRITERIOS]
                              + [alvo[c] for c in CRITERIOS])
        print(f"Tabela 7 com envios fixados: {st7}" + (f"  custo decomposto: {dec7}" if dec7 else ""))

    caminho = salvar_csv("e1_reproducao.csv",
                         ["variante", "linha", "TC", "TCF", "ATT", "TC_artigo", "TCF_artigo", "ATT_artigo"],
                         linhas_csv)
    print(f"\nCSV salvo em {caminho}")


def e2(I=50, J=6, T=52, semente=2026):
    """
    E2: instância de grande porte (padrão 50 fábricas, 6 modais, 52 semanas),
    gerada nas faixas do artigo. Aplica o mesmo procedimento do artigo (payoff +
    WS com os pesos das 3 rodadas do AHP) no modelo completo e mede os tempos.
    """
    import time
    from gerador import gerar, salvar
    from procedimento import minimizar_criterio

    artigo = carregar_instancia(os.path.join(RAIZ, "data", "artigo.json"))
    inst = gerar(I, J, T, semente)
    caminho_inst = salvar(inst, os.path.join(RAIZ, "data", "instancias", f"grande_{I}x{J}x{T}.json"))
    print(f"Instância salva em {caminho_inst}")
    print(f"Demanda total: {sum(inst['D'])} contêineres")

    inicio = time.perf_counter()
    m = montar_modelo(inst)
    tempo_montagem = time.perf_counter() - inicio
    tam = contar(m)
    print(f"Tamanho: {tam}   (montagem em {tempo_montagem:.2f} s)")

    linhas_csv = []

    # Passos 1–2: tabela payoff (com desempate lexicográfico)
    payoff = {"linhas": {}}
    for k in CRITERIOS:
        inicio = time.perf_counter()
        sol, valores = minimizar_criterio(m, k)
        tempo = time.perf_counter() - inicio
        payoff["linhas"][k] = valores
        linhas_csv.append([f"min_{k}", round(tempo, 3)] + [round(valores[c], 4) for c in CRITERIOS]
                          + modal_share(m, sol))
        print(f"min {k:4s}: " + "  ".join(fmt(valores[c], c) for c in CRITERIOS) + f"   ({tempo:.2f} s)")
    payoff["min"] = {k: payoff["linhas"][k][k] for k in CRITERIOS}
    payoff["max"] = {k: max(payoff["linhas"][o][k] for o in CRITERIOS) for k in CRITERIOS}

    # Passo 3: soma ponderada com os pesos do artigo
    for r in artigo["ahp"]:
        inicio = time.perf_counter()
        sol, valores, ws = solucao_ws(m, pesos_rodada(r), payoff)
        tempo = time.perf_counter() - inicio
        linhas_csv.append([f"rodada{r['rodada']}", round(tempo, 3)] + [round(valores[c], 4) for c in CRITERIOS]
                          + modal_share(m, sol))
        print(f"rodada {r['rodada']}: " + "  ".join(fmt(valores[c], c) for c in CRITERIOS) + f"   ({tempo:.2f} s)")

    cab = ["linha", "tempo_s", "TC", "TCF", "ATT"] + [f"pct_{nome}" for nome in inst["modais"]]
    caminho = salvar_csv(f"e2_grande_{I}x{J}x{T}.csv", cab, linhas_csv)
    salvar_csv(f"e2_tamanho_{I}x{J}x{T}.csv",
               ["I", "J", "T", "variaveis", "variaveis_livres", "restricoes", "nao_nulos", "montagem_s",
                "tempo_total_solver_s"],
               [[I, J, T, tam["variaveis"], tam["variaveis_livres"], tam["restricoes"], tam["nao_nulos"],
                 round(tempo_montagem, 3), round(sum(l[1] for l in linhas_csv), 3)]])
    print(f"\nCSV salvo em {caminho}")


def modal_share(m, sol):
    """Percentual dos contêineres enviados por cada modal."""
    x = tabela_envios(m, sol)
    por_modal = x.sum(axis=(0, 2))
    total = por_modal.sum()
    return [round(100 * v / total, 2) if total else 0.0 for v in por_modal]


def comparar_com_fronteira(m, solucoes):
    """Para cada solução (rótulo, TC, TCF), resolve a ε-restrição com ε = TCF dela e mostra
    quanto de custo a mais ela paga em relação à fronteira (em troca de um ATT menor)."""
    from epsilon import ponto_epsilon
    for rotulo, tc, tcf in solucoes:
        _, f = ponto_epsilon(m, tcf + 1e-6)
        print(f"   {rotulo}: TC = {tc:,.0f}; fronteira = {f['TC']:,.0f}  "
              f"(custo a mais: {100 * (tc - f['TC']) / f['TC']:.2f}%)")


def e3():
    """
    E3 (ponto extra): fronteira de Pareto custo × CO2 pela ε-restrição.
      - instância do artigo, nas variantes C (como o artigo resolveu) e A (modelo completo),
        com as 3 soluções publicadas marcadas;
      - instância grande 50×6×52, com as 3 soluções WS marcadas.
    """
    from epsilon import fronteira, ponto_epsilon
    from figuras import fronteira_artigo, fronteira_grande
    from gerador import gerar

    artigo = carregar_instancia(os.path.join(RAIZ, "data", "artigo.json"))
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    def salvar_fronteira(nome, pontos):
        salvar_csv(nome, ["epsilon", "TC", "TCF", "ATT", "custo_marginal_co2", "tempo_s"],
                   [[round(p["epsilon"], 3), round(p["TC"], 2), round(p["TCF"], 2), round(p["ATT"], 4),
                     "" if p["custo_marginal_co2"] is None else round(p["custo_marginal_co2"], 4),
                     round(p["tempo_s"], 3)] for p in pontos])

    def rodadas_ws(m):
        payoff = tabela_payoff(m)
        saida = []
        for r in artigo["ahp"]:
            _, v, _ = solucao_ws(m, pesos_rodada(r), payoff)
            saida.append((f"Rodada {r['rodada']}", v["TC"], v["TCF"]))
        return saida

    # --- Instância do artigo --------------------------------
    m_c = montar_modelo(artigo, capacidade_producao=False)
    m_a = montar_modelo(artigo)
    pontos_c = fronteira(m_c, 41)
    pontos_a = fronteira(m_a, 41)
    salvar_fronteira("e3_fronteira_artigo_C_sem_cap.csv", pontos_c)
    salvar_fronteira("e3_fronteira_artigo_A_completo.csv", pontos_a)

    alvos = artigo["alvos"]
    publicadas = [(f"Rodada {k}", alvos[f"rodada{k}"]["TC"], alvos[f"rodada{k}"]["TCF"]) for k in (1, 2, 3)]
    completo = rodadas_ws(m_a)

    print("Instância do artigo — soluções publicadas × fronteira com o mesmo CO2 (variante C):")
    comparar_com_fronteira(m_c, publicadas)
    print("Rodadas no modelo completo (A): " + "; ".join(f"{r}: TC={tc:,.0f} TCF={tcf:,.0f}" for r, tc, tcf in completo))
    print(f"Extremos C: TC {pontos_c[0]['TC']:,.0f} -> {pontos_c[-1]['TC']:,.0f}; "
          f"TCF {pontos_c[0]['TCF']:,.0f} -> {pontos_c[-1]['TCF']:,.0f}")
    print(f"Extremos A: TC {pontos_a[0]['TC']:,.0f} -> {pontos_a[-1]['TC']:,.0f}; "
          f"TCF {pontos_a[0]['TCF']:,.0f} -> {pontos_a[-1]['TCF']:,.0f}")

    fig1 = fronteira_artigo(pontos_c, pontos_a, publicadas, completo,
                            os.path.join(PASTA_RESULTADOS, "fig_fronteira_artigo.png"))
    print(f"Figura: {fig1}")

    # --- Instância grande ----------------------------------
    inst = gerar(50, 6, 52, 2026)
    m_g = montar_modelo(inst)
    pontos_g = fronteira(m_g, 21)
    salvar_fronteira("e3_fronteira_grande_50x6x52.csv", pontos_g)
    rod_g = rodadas_ws(m_g)
    print(f"Instância grande: {len(pontos_g)} pontos, tempo total {sum(p['tempo_s'] for p in pontos_g):.1f} s")
    comparar_com_fronteira(m_g, rod_g)
    fig2 = fronteira_grande(pontos_g, rod_g, os.path.join(PASTA_RESULTADOS, "fig_fronteira_grande.png"),
                            "Fronteira de Pareto custo × CO$_2$ (instância 50 fábricas × 6 modais × 52 semanas)")
    print(f"Figura: {fig2}")


if __name__ == "__main__":
    qual = sys.argv[1] if len(sys.argv) > 1 else "e1"
    {"e1": e1, "e2": e2, "e3": e3}[qual]()
