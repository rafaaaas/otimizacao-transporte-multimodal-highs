# Planejamento de transporte multimodal multiobjetivo — PL, Etapa 3

Reprodução e extensão do modelo de

> Kuranovic, V.; Ustinovichius, L.; Nowak, M.; Bazaras, D.; Sokolovskij, E.
> *Improving the Freight Transportation System in the Context of the Country's Economic
> Development.* Sustainability 2025, 17, 6327. https://doi.org/10.3390/su17146327

O problema planeja, semana a semana, quantos contêineres enviar de cada fábrica (China)
para um centro logístico (Alemanha) por avião, navio, trem ou caminhão. Os três objetivos
são custo total (TC), emissão de CO₂ (TCF) e tempo médio de transporte (ATT).
Solver: **HiGHS** (via `highspy`).

## Instalação

Requer Python 3.10 ou mais novo.

```bash
python -m pip install -r requirements.txt
```

## Como rodar

Sempre a partir desta pasta:

```bash
python src/experimentos.py e1    # reprodução do artigo (~1 s)
python src/experimentos.py e2    # instância grande 50 fábricas × 6 modais × 52 semanas (~10 s)
python src/experimentos.py e3    # fronteira de Pareto custo × CO2 pela ε-restrição (~2 min)
```

Se os acentos aparecerem estranhos no terminal do Windows, rode antes, no PowerShell:
`$env:PYTHONIOENCODING="utf-8"`.

Os resultados (CSV com separador `;` e figuras PNG) são gravados em `resultados/`.

| Experimento | O que faz | Saídas |
|---|---|---|
| E1 | Resolve a instância do artigo em 3 variantes (A: modelo completo com z₀ = 0; B: z₀ calibrado; C: sem a restrição de produção) e compara com as Tabelas 4–7 | `e1_reproducao.csv` |
| E2 | Gera a instância 50×6×52 (20 904 variáveis, 2 964 restrições) e aplica o procedimento do artigo | `e2_grande_50x6x52.csv`, `e2_tamanho_50x6x52.csv`, `data/instancias/grande_50x6x52.json` |
| E3 | ε-restrição: fronteira de Pareto TC × TCF, com as soluções do artigo marcadas | `e3_fronteira_*.csv`, `fig_fronteira_artigo.png`, `fig_fronteira_grande.png` |

## Estrutura

```
data/artigo.json        dados da Seção 5 do artigo, pesos do AHP e valores-alvo
data/instancias/        instâncias geradas
src/modelo.py           monta o PPL esparso (variáveis, restrições, 3 vetores de custo)
src/solver_highs.py     resolve com o HiGHS; devolve (status, z, solução, tempo)
src/procedimento.py     método do artigo: tabela payoff + soma ponderada (WS)
src/epsilon.py          método da ε-restrição
src/gerador.py          gerador de instâncias grandes (semente fixa, viável por construção)
src/figuras.py          gráficos
src/experimentos.py     E1, E2, E3
resultados/             CSVs e figuras
ARQUITETURA.md          decisões, modelo, resultados detalhados e análise crítica
GUIA_RELATORIO.md       roteiro para escrever o relatório SBPO
```

## Uso como biblioteca

```python
import sys; sys.path.insert(0, "src")
from modelo import carregar_instancia, montar_modelo, avaliar
from solver_highs import resolver

inst = carregar_instancia("data/artigo.json")
m = montar_modelo(inst)                        # modelo completo
status, z, sol, tempo = resolver(m, m["custos"]["TC"])
print(status, avaliar(m, sol))
```
