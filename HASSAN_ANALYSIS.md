# Análise de Qualidade de Descrições - Estilo Hassan et al.

## Metodologia

Este projeto implementa a análise de qualidade de descrições de ferramentas MCP baseada na metodologia de **Hasan et al. (2026)**, usando uma rubrica de 6 componentes com escala Likert de 5 pontos.

### Rubrica de 6 Componentes

| # | Componente | Descrição | Escala |
|---|-----------|-----------|--------|
| 1 | **Purpose** | Clareza e completude: o que a ferramenta faz? | 1-5 |
| 2 | **Guidelines** | Critérios de decisão (quando usar) + instruções operacionais (como usar) | 1-5 |
| 3 | **Limitations** | Restrições conhecidas, ressalvas, casos de falha | 1-5 |
| 4 | **Parameter Explanation** | Papéis dos parâmetros de entrada além de tipos de dados | 1-5 |
| 5 | **Length & Completeness** | Mínimo 3-4 sentenças substantivas vs fragmentos terse | 1-5 |
| 6 | **Examples** | Exemplos ilustrativos de uso correto/efetivo | 1-5 |

### Escala Likert

```
5 = Ideal: satisfaz completamente sem ambiguidade
4 = Pequena ambiguidade: satisfaz a maioria, pequenas lacunas resolvíveis
3 = Mínimo viável: atinge limite mínimo, incompleto mas usável
2 = Vago: presente mas muito fino ou pouco claro para confiar
1 = Ausente: aspecto completamente ausente
```

---

## Dados de Entrada

### Avaliações dos Juízes

Arquivo: `data/evaluations/{judge_id}.jsonl`

Um JSON por linha, com estrutura:

```json
{
  "tool_uid": "repo::tool_name::file:line",
  "repo": {
    "name_with_owner": "org/repo",
    "stargazer_count": 123,
    "primary_language": "Python"
  },
  "tool": {
    "name": "tool_name",
    "description": "..."
  },
  "scenario": "description_only | with_source",
  "judge": {
    "id": "qwen2.5-14b-instruct",
    "model_id": "gemini-3.1-flash-live-preview"
  },
  "status": "ok | refused | error",
  "scores": {
    "purpose": {"score": 4, "reasoning": "..."},
    "guidelines": {"score": 3, "reasoning": "..."},
    "limitations": {"score": 2, "reasoning": "..."},
    "parameter_explanation": {"score": 4, "reasoning": "..."},
    "length_completeness": {"score": 5, "reasoning": "..."},
    "examples": {"score": 3, "reasoning": "..."}
  },
  "usage": {
    "input_tokens": 1500,
    "output_tokens": 500
  }
}
```

---

## Executar Análise

### Análise Geral (Todos os Juízes + Todos os Dados)

```bash
uv run python scripts/analyze_judges_results.py
```

Saída:
- Resumo geral (OK, recusadas, erros)
- Estatísticas por componente (média, desvio padrão, distribuição)
- Ranking dos componentes
- Comparação de cenários (description_only vs with_source)
- Análise por linguagem
- Análise por juiz

### Análise de Um Juiz Específico

```bash
# Apenas Qwen
uv run python scripts/analyze_judges_results.py --judge qwen2.5-14b-instruct

# Apenas Llama
uv run python scripts/analyze_judges_results.py --judge llama-uncensored

# Apenas Gemini
uv run python scripts/analyze_judges_results.py --judge gemini-3.1-flash-live-preview
```

### Análise com Limite

```bash
# Processar apenas primeiras 100 avaliações
uv run python scripts/analyze_judges_results.py --limit 100

# Qwen + primeiras 50 avaliações
uv run python scripts/analyze_judges_results.py --judge qwen2.5-14b-instruct --limit 50
```

---

## Outputs

### 1. Sumário Executivo (Console)

```
================================================================================
ANÁLISE DE RESULTADOS DOS JUÍZES - ESTILO HASSAN ET AL.
================================================================================

📊 RESUMO GERAL
  Total de avaliações: 24.342
  ✅ OK: 23.450 (96.3%)
  ⚠️  Recusadas (segurança): 650 (2.7%)
  ❌ Erros: 242 (1.0%)

📈 ANÁLISE POR COMPONENTE DA RUBRICA
  Componente                       Média  Desvio Padrão  Min  Max     N
  Purpose                           4.23            0.81    1    5  23450
  Guidelines                        3.45            1.12    1    5  23450
  Limitations                       2.87            1.34    1    5  23450
  Parameter Explanation             3.92            0.95    1    5  23450
  Length & Completeness             4.01            1.02    1    5  23450
  Examples                          2.56            1.45    1    5  23450

🏆 RANKING (por média)
  1. Purpose                           4.23
  2. Parameter Explanation            3.92
  3. Length & Completeness            4.01
  4. Guidelines                       3.45
  5. Limitations                      2.87
  6. Examples                         2.56
```

### 2. CSV com Dados Detalhados

- `data/analysis/judges_results/rubric_components_summary.csv` — Estatísticas por componente
- `data/analysis/judges_results/scenario_comparison.csv` — Comparação description_only vs with_source
- `data/analysis/judges_results/by_language.csv` — Análise por linguagem

### 3. JSON para Processamento

`data/analysis/judges_results/analysis_results.json` — Todos os números em formato estruturado

```json
{
  "summary": {
    "total": 24342,
    "ok": 23450,
    "refused": 650,
    "error": 242
  },
  "rubric_components": {
    "purpose": {
      "count": 23450,
      "mean": 4.23,
      "stdev": 0.81,
      "distribution": {
        "1": 120,
        "2": 450,
        "3": 1200,
        "4": 8900,
        "5": 12780
      }
    },
    ...
  }
}
```

---

## Interpretação dos Resultados

### Métricas Principais

#### 1. Média por Componente
- **Alta (4.0-5.0)**: Componente bem coberto nas descrições
- **Média (3.0-3.9)**: Componente parcialmente coberto
- **Baixa (1.0-2.9)**: Componente negligenciado ou ausente

#### 2. Distribuição Likert
```
Ideal case:   5 | ████████████ | 50%+ das avaliações
Good case:    4 | ████████     | 30-40% das avaliações
Passable:     3 | ███          | 10-20% das avaliações
Poor:       1-2 | █            | <5% das avaliações
```

#### 3. Desvio Padrão
- **Baixo (< 0.8)**: Consenso entre juízes (descrições consistentes)
- **Médio (0.8-1.2)**: Variação moderada
- **Alto (> 1.2)**: Desacordo entre juízes (descrições ambíguas)

### Comparação: Cenários

**description_only** vs **with_source**:
- Se with_source tem média **significativamente maior**: código-fonte é crítico para compreender o propósito/limitações
- Se similar: descrição é auto-suficiente

### Análise por Linguagem

```
Python:     N=5000  Média=4.12  [Melhor documentação]
TypeScript: N=2800  Média=3.95
Java:       N=1200  Média=3.78
Rust:       N=800   Média=3.65  [Pior documentação]
```

Padrão: linguagens mais populares tendem ter descrições melhores.

### Análise por Juiz

Se juízes discordam sistematicamente:
- **Qwen vs Llama média diferente**: Modelos têm critérios diferentes
- **Alta variância em um juiz**: Juiz é menos confiável

---

## Replicar Análise de Hassan

### Etapa 1: Coleta e Extração (Etapas 1-2)
```bash
uv run python -m mcp_pipeline.collection.run_step1
uv run python -m mcp_pipeline.pipeline.run_clone
uv run python -m mcp_pipeline.pipeline.run_step2
uv run python -m mcp_pipeline.schema.assemble_dataset
```

### Etapa 2: Avaliação (Etapa 3)
```bash
# Com juízes locais (Qwen + Llama)
uv run python -m mcp_pipeline.pipeline.run_step3

# Ou incluindo o juiz cloud (Gemini free tier -- ver GEMINI_FREE_TIER_UNAVAILABLE.md)
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --judges qwen2.5-14b-instruct,llama-uncensored,gemini-2.5-flash-lite
```

### Etapa 3: Análise (ESTE SCRIPT)
```bash
# Análise geral
uv run python scripts/analyze_judges_results.py

# Análise por juiz
uv run python scripts/analyze_judges_results.py --judge qwen2.5-14b-instruct
uv run python scripts/analyze_judges_results.py --judge llama-uncensored

# Gerar gráficos e tabelas adicionais
uv run python -m scripts.analysis_report
```

---

## Diferenças vs Hassan et al.

| Aspecto | Hassan et al. | Este Projeto |
|---------|--------------|-------------|
| Modelos avaliadores | GPT-4.1-mini, Claude 3.5 Haiku, Qwen-3-30B | Qwen2.5-14B, Llama, Gemini 2.5 Flash-Lite |
| Rubrica | 6 componentes | 6 componentes (idêntica) |
| Escala Likert | 5 pontos | 5 pontos (idêntica) |
| Cenários | 2 (description_only, with_source) | 2 (idêntica) |
| Tamanho dataset | ~3000 tools | 12.171 tools |
| Juízes em paralelo | Não | Sim (>2x mais rápido) |
| Modelo local | Não | Sim (Qwen + Llama) |
| Open source | Parcial | Total |

---

## Próximos Passos

1. **Executar Etapa 3** (coleta de avaliações):
   ```bash
   uv run python -m mcp_pipeline.pipeline.run_step3 --limit 100
   ```

2. **Rodar análise**:
   ```bash
   uv run python scripts/analyze_judges_results.py
   ```

3. **Interpretar resultados**:
   - Quais componentes estão melhores/piores?
   - Como os cenários diferem?
   - Como as linguagens se comparam?

4. **Publicar resultados**:
   - Tabelas em `data/analysis/judges_results/`
   - JSON estruturado para post-processamento

---

## Referências

- Hasan et al. (2026). *Evaluating Tool Descriptions in Large Language Models*
- Este projeto: [README.md](README.md), [MIGRATION_LOCAL_JUDGES.md](MIGRATION_LOCAL_JUDGES.md)
- Rubrica: [src/mcp_pipeline/evaluation/prompts.py](src/mcp_pipeline/evaluation/prompts.py)
