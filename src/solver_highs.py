# ============================================================
# RESOLUÇÃO COM HiGHS
# ============================================================
#
# Adaptação do resolver_lp (versão LINDO) para o HiGHS:
#   - entrada ESPARSA (trios linha / coluna / valor) vinda de modelo.py;
#   - variáveis contínuas (PL) ou inteiras (MIP);
#   - linhas extras opcionais (ex.: ε-restrição  TCF <= ε);
#   - devolve (status, z, solução, tempo) em vez de encerrar o programa.
#
# A troca do LINDO pelo HiGHS foi feita porque a licença Demo do
# LINDO limita o modelo a 300 variáveis, 150 restrições e 30 inteiras.
# ============================================================

import time
import numpy as np
import highspy

INFINITO = highspy.kHighsInf


def resolver(modelo, custo, direcao=1, linhas_extras=None, inteiro=False,
             limite_inf=None, limite_sup=None, tempo_limite=None, mostrar_log=False):
    """
    Resolve o modelo com o vetor de custo informado.

    custo          : vetor denso (tamanho n_variaveis)
    direcao        : 1 = minimizar, -1 = maximizar (mesma convenção do código LINDO)
    linhas_extras  : lista de dicionários {"coef": vetor denso, "tipo": "L"/"G"/"E", "rhs": valor}
    inteiro        : True -> todas as variáveis x[i,j,t] e y[i,t] viram inteiras (contêineres)
    limite_inf/sup : substituem os limites do modelo (ex.: para fixar os envios)

    Retorna (status, z, solucao, tempo), onde status é um texto ("Optimal", "Infeasible", ...).
    """

    n = modelo["n_variaveis"]
    linhas = list(modelo["linhas"])
    colunas = list(modelo["colunas"])
    valores = list(modelo["valores"])
    b = list(modelo["b"])
    tipos = list(modelo["tipos"])

    # --- Acrescentar linhas extras ----------------------------

    for extra in linhas_extras or []:
        r = len(b)
        coef = np.asarray(extra["coef"], dtype=np.double)
        for k in np.nonzero(coef)[0]:
            linhas.append(r)
            colunas.append(int(k))
            valores.append(coef[k])
        b.append(extra["rhs"])
        tipos.append(extra["tipo"])

    m = len(b)

    # --- Tipos L/G/E viram intervalos [inferior, superior] -----

    linha_inf = np.empty(m)
    linha_sup = np.empty(m)
    for r, (tipo, rhs) in enumerate(zip(tipos, b)):
        linha_inf[r] = rhs if tipo in ("G", "E") else -INFINITO
        linha_sup[r] = rhs if tipo in ("L", "E") else INFINITO

    # --- Converter os trios para o formato por colunas ---------
    # (mesmo formato inicio_colunas / coef / linha usado no LINDO)

    ordem = np.lexsort((linhas, colunas))
    colunas_ord = np.asarray(colunas, dtype=np.int32)[ordem]
    linha_coef = np.asarray(linhas, dtype=np.int32)[ordem]
    coef_matriz = np.asarray(valores, dtype=np.double)[ordem]
    inicio_colunas = np.searchsorted(colunas_ord, np.arange(n + 1)).astype(np.int32)

    # --- Montar o modelo do HiGHS -----------------------------

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.sense_ = highspy.ObjSense.kMinimize if direcao == 1 else highspy.ObjSense.kMaximize
    lp.col_cost_ = np.asarray(custo, dtype=np.double)
    li = modelo["limite_inf"] if limite_inf is None else limite_inf
    ls = modelo["limite_sup"] if limite_sup is None else limite_sup
    lp.col_lower_ = np.asarray(li, dtype=np.double)
    lp.col_upper_ = np.where(np.isinf(ls), INFINITO, ls).astype(np.double)
    lp.row_lower_ = linha_inf
    lp.row_upper_ = linha_sup
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = inicio_colunas
    lp.a_matrix_.index_ = linha_coef
    lp.a_matrix_.value_ = coef_matriz

    if inteiro:
        tipo_var = [
            highspy.HighsVarType.kInteger if chave[0] in ("x", "y") else highspy.HighsVarType.kContinuous
            for chave in modelo["nomes"]
        ]
        lp.integrality_ = tipo_var

    solver = highspy.Highs()
    solver.setOptionValue("output_flag", mostrar_log)
    if tempo_limite:
        solver.setOptionValue("time_limit", float(tempo_limite))
    solver.passModel(lp)

    # --- Resolver ----------------------------------------------

    inicio = time.perf_counter()
    solver.run()
    tempo = time.perf_counter() - inicio

    status = solver.modelStatusToString(solver.getModelStatus())
    if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return status, None, None, tempo

    z = solver.getInfo().objective_function_value
    solucao = np.array(solver.getSolution().col_value)
    return status, z, solucao, tempo
