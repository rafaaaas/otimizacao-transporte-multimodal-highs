# ============================================================
# GERADOR DE INSTÂNCIAS DE GRANDE PORTE
# ============================================================
#
# Gera instâncias no mesmo formato de data/artigo.json, com dados
# sorteados nas MESMAS FAIXAS do artigo:
#
#   - modais 1–4 (avião, navio, trem, caminhão): custo, CO2 e tempo
#     sorteados em torno dos valores da Tabela 3 (±10%; tempo +0 ou +1);
#   - modais 5–6 (novos, só quando J > 4): navio expresso e
#     intermodal trem+caminhão, com valores intermediários;
#   - produção por fábrica na faixa do artigo (100–150 por semana);
#   - capacidade dos modais proporcional ao número de fábricas
#     (no artigo, B foi dado para 2 fábricas);
#   - demanda com o mesmo perfil do artigo: zero nas 2 primeiras
#     semanas e crescente depois.
#
# A instância é VIÁVEL POR CONSTRUÇÃO: a demanda total é uma fração
# (folga) do que pode ser produzido e entregue até a semana T.
# A semente fixa garante que a mesma instância seja gerada sempre.
# ============================================================

import json
import os
import numpy as np

# Valores-base por modal: (nome, custo, CO2, tempo em semanas, capacidade para 2 fábricas)
# Modais 1–4: médias das duas fábricas da Tabela 3 do artigo.
MODAIS_BASE = [
    ("aviao",           195, 110, 1,  50),
    ("navio",            52,  65, 4, 200),
    ("trem",            145,  45, 2, 200),
    ("caminhao",        105,  28, 3, 100),
    ("navio_expresso",   80,  75, 3, 150),
    ("trem_caminhao",   120,  38, 3, 120),
]


def gerar(I, J, T, semente=2026, folga=0.75):
    """
    Gera uma instância com I fábricas, J modais (máx. 6) e T semanas.

    folga : fração da capacidade máxima de entrega usada como demanda total
            (0.75 = a demanda ocupa 75% do que dá para entregar até T).
    """
    if J > len(MODAIS_BASE):
        raise ValueError(f"J máximo = {len(MODAIS_BASE)}")
    rng = np.random.default_rng(semente)
    modais = MODAIS_BASE[:J]

    def variar(base, n, amplitude=0.10):
        return base * rng.uniform(1 - amplitude, 1 + amplitude, n)

    # --- Parâmetros por fábrica e modal ----------------------

    c = np.zeros((I, J))
    e = np.zeros((I, J))
    tau = np.zeros((I, J), dtype=int)
    for j, (_, custo, co2, tempo, _) in enumerate(modais):
        c[:, j] = np.round(variar(custo, I))
        e[:, j] = np.round(variar(co2, I))
        tau[:, j] = tempo + rng.integers(0, 2, I)    # como no artigo: navio 4 ou 5 semanas

    B = [int(round(cap * I / 2 * rng.uniform(0.9, 1.1))) for (_, _, _, _, cap) in modais]
    h = [int(v) for v in rng.integers(10, 13, I)]   # artigo: 12 e 10
    H, P = 30, 50                                    # iguais ao artigo

    # Produção: cada fábrica tem um nível na faixa 100–150, com ±10% por semana
    nivel = rng.uniform(100, 150, I)
    p = np.round(nivel[:, None] * rng.uniform(0.9, 1.1, (I, T))).astype(int)

    # --- Demanda viável por construção -----------------------
    # Só conta a produção que ainda consegue chegar até T pelo modal mais rápido,
    # e só a capacidade de modal que entrega até T.

    entregavel_producao = sum(
        p[i, t] for i in range(I) for t in range(T) if (t + 1) + tau[i].min() <= T
    )
    entregavel_modais = sum(
        B[j] * max(0, T - int(tau[:, j].min())) for j in range(J)
    )
    total = folga * min(entregavel_producao, entregavel_modais)

    # Perfil do artigo: 0, 0 e depois crescente (com sazonalidade leve)
    semanas = np.arange(1, T + 1)
    perfil = np.where(semanas <= 2, 0.0, 1.0 + 0.5 * (semanas - 3) / max(1, T - 3))
    perfil *= 1 + 0.15 * np.sin(2 * np.pi * semanas / 13)
    perfil *= rng.uniform(0.9, 1.1, T)
    perfil[semanas <= 2] = 0
    D = np.floor(total * perfil / perfil.sum()).astype(int)

    return {
        "descricao": f"Instancia gerada: I={I}, J={J}, T={T}, semente={semente}, folga={folga}. "
                     f"Dados sorteados nas faixas do artigo (Sustainability 2025, 17, 6327).",
        "I": I, "J": J, "T": T,
        "modais": [nome for (nome, *_) in modais],
        "B": B,
        "h": h,
        "H": H,
        "P": P,
        "p": p.tolist(),
        "D": D.tolist(),
        "tau": tau.tolist(),
        "c": c.astype(int).tolist(),
        "e": e.astype(int).tolist(),
        "z0": [0] * I,
        "ahp": None,
        "semente": semente,
    }


def salvar(inst, caminho):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(inst, f, ensure_ascii=False)
    return caminho
