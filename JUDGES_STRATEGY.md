# Estratégia de Juízes para Etapa 3

## Análise de Capacidade e Custos

Este documento compara as 3 estratégias de juízes (Local, Cloud, Misto) com foco nas limitações do Gemini free tier.

### Dataset

- **Total de ferramentas**: 12.171
- **Cenários por ferramenta**: 2 (description_only, with_source)
- **Total de avaliações necessárias**: 24.342
- **Tokens por avaliação (estimado)**: 3.000 (1.500 input + 1.500 output)
- **Tokens totais necessários**: ~73M

---

## Estratégia 1: 100% Local (Recomendado para Free Tier)

### Juízes
- Prometheus 7B V2.0 (llama-server em 192.168.15.15:8091)
- Llama uncensored (llama-server em 192.168.15.15:8090)

### Capacidade Diária
| Métrica | Valor |
|---------|-------|
| Custo API | $0 |
| Juízes ativos | 2 |
| Requisições/min | Ilimitado (seu hardware) |
| Avaliações/dia com 2 juízes em paralelo | ~1.440 (720 tools × 2 cenários) |
| Dias para dataset completo | ~17 dias |
| Latência/ferramenta | ~30-60s (depende do hardware) |

### Vantagens
✅ Nenhum custo de API  
✅ Nenhum limite de rate  
✅ Dados permanecem locais  
✅ Controle total sobre modelos  
✅ Resumível indefinidamente  

### Desvantagens
❌ Depende do hardware local  
❌ Latência mais alta  
❌ Nenhuma redundância se servidor cair  

### Recomendação
**IDEAL para dataset completo**. Configure assim:

```yaml
# config/judges.yaml
judges:
  - id: prometheus-7b-v2.0
    provider: prometheus
    model_id: prometheus-7b-v2.0
    enabled: true

  - id: llama-uncensored
    provider: llama
    model_id: llama-uncensored
    enabled: true
```

Execução:
```bash
uv run python -m mcp_pipeline.pipeline.run_step3
# ~17 dias para dataset completo
```

---

## Estratégia 2: 100% Cloud (NÃO RECOMENDADO para Free Tier)

### Juízes
- Claude Haiku 4.5 (Anthropic)
- GPT-4.1-mini (OpenAI)
- Gemini 2.5 Flash-Lite (Google)

### Capacidade Diária (Free Tier)

#### Claude Haiku 4.5
- Sem limite de rate público (tier pay-as-you-go)
- Custo: ~$0.08/1M input tokens
- Avaliações/dia: Ilimitado (com custo)
- **Viável**: Sim, mas com custo

#### OpenAI gpt-4.1-mini
- Limite de rate: padrão por tier
- Custo: ~$0.15/1M input tokens
- Avaliações/dia: Ilimitado (com custo)
- **Viável**: Sim, mas com custo

#### Google Gemini (FREE TIER - PROBLEMA!)
| Métrica | Limite |
|---------|--------|
| Rate limit | 15 req/min = 21.6k/dia |
| Quota tokens | 1M/dia (TOTAL, compartilhado) |
| Avaliações/dia com 3k tokens cada | ~333 (sem margem) |
| Dias para dataset completo | ~73 dias |
| **Bloqueador** | Muito lento, provavelmente vai bater limite |

### Custo Estimado (Sem Gemini)
```
73M tokens ÷ 2 juízes = 36.5M tokens cada

Claude:    36.5M × $0.08/1M = $2.92
OpenAI:    36.5M × $0.15/1M = $5.48
---
Total:     ~$8.40 para dataset completo
```

### Vantagens
✅ Modelos mais capazes (flagship)  
✅ Menor latência (~5-15s/ferramenta)  
✅ Redundância (3 juízes)  
✅ Sem dependência de hardware local  

### Desvantagens
❌ **Gemini free tier é inadequado** (~73 dias ou vai bater rate limit)  
❌ Custo de API (se não usar Gemini)  
❌ Dados enviados para cloud  
❌ Dependência de conectividade  
❌ Possíveis delays de API  

### Recomendação
**NÃO use para dataset completo no free tier**. Se usar:

1. Desabilite Gemini
2. Use apenas Claude + OpenAI (com custo)
3. OU: Use Gemini apenas para pequeno subset (10-20 tools)

```yaml
# NÃO faça isso:
judges:
  - id: gemini-2.5-flash-lite
    provider: google
    enabled: true  # ❌ Vai falhar ou levar 73 dias
```

---

## Estratégia 3: Misto (Local + Cloud Selective)

### Recomendação Otimizada

Use **Prometheus + Llama locais** como primários + **Gemini para validação pequena**.

```yaml
# config/judges.yaml - RECOMENDADO
judges:
  # Primários (local) - rápidos, sem limite
  - id: prometheus-7b-v2.0
    provider: prometheus
    enabled: true

  - id: llama-uncensored
    provider: llama
    enabled: true

  # Secundário (validation apenas) - desabilitado por padrão
  - id: gemini-2.5-flash-lite
    provider: google
    enabled: false  # ← Ativar apenas para testes pequenos
```

### Execução

**Fase 1: Dataset completo (local)**
```bash
# Avalia todas as 24.342 avaliações com Prometheus + Llama
# ~17 dias, nenhum custo
uv run python -m mcp_pipeline.pipeline.run_step3
```

**Fase 2: Validation subset (cloud)**
```bash
# Opcional: valida com Gemini apenas ~20 tools
# Usa ~20 × 3k tokens × 2 cenários = 120k tokens (~0.1% da quota)
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --judges gemini-2.5-flash-lite \
  --limit 20 \
  --retry-failed  # Retenta tools que falharam antes
```

### Resultado
| Métrica | Valor |
|---------|-------|
| Custo total | $0 |
| Tempo para completo | ~17 dias |
| Juízes primários | 2 locais |
| Juízes validação | 1 cloud (optional) |
| Capacidade Gemini usada | <1% |

---

## Limitações do Gemini Free Tier - Análise Detalhada

### Problema: Taxa de Avaliação vs. Quota Diária

```
Quota/dia: 1M tokens
Tokens/avaliação: ~3k
Avaliações possíveis/dia: 1M ÷ 3k = 333

Dataset: 24.342 avaliações
Dias necessários: 24.342 ÷ 333 = 73 dias

Com rate limit (15 req/min):
Requisições/dia: 15 × 60 × 24 = 21.600
Avaliações/dia: 21.600 (com ~47 tokens cada = impossível)
```

### Bloqueadores Conhecidos

1. **Quota de tokens**: 1M/dia é extremamente baixa
   - Um prompt médio de rubrica: 2.000+ tokens
   - Limita a ~500 requisições/dia (conservador)

2. **Rate limit**: 15 req/min é apertado para workloads de volume
   - Sem throttling automático = 429 errors após limite
   - Spinner rate limit → retry → novo erro

3. **Structured output**: Gemini requer response_schema
   - Aumenta overhead de tokens (~10-20% mais)

4. **Sem caching livre**: Não há benefits de prompt caching (pago)

### Conclusão sobre Gemini

**Status**: ❌ **Inadequado para dataset completo no free tier**

**Uso recomendado**: 
- ✅ Testes pequenos (5-10 tools)
- ✅ Validação seletiva (10-20 tools)
- ❌ NÃO para 12.171 tools

---

## Matriz de Decisão

| Caso de Uso | Recomendação | Alternativa |
|-------------|--------------|------------|
| Dataset completo (24k tools) | Local (Prometheus + Llama) | Cloud (com custo $8-15) |
| Testes/desenvolvimento | Local (--limit 10-20) | Gemini (--limit 5) |
| Validação seletiva | Local, depois Gemini | Cloud (Claude + OpenAI) |
| Produção/reprodutibilidade | Local | Claude (pago) |
| Zero custo | Local ✅ | Não existe |
| Máxima qualidade | Claude (pago) | Gemini (free, inadequado) |

---

## Plano de Implementação Recomendado

### Passo 1: Setup Local (Dia 1)
```bash
# 1h de setup
uv run python scripts/check_local_llm_servers.py
uv run python scripts/test_local_judges.py --sample-size 20
```

### Passo 2: Execução Primária (Semana 1-3)
```bash
# Roda com Prometheus + Llama
uv run python -m mcp_pipeline.pipeline.run_step3

# Monitorar:
tail -f logs/step3.log
# ~17 dias para 24.342 avaliações
```

### Passo 3: Validação Opcional (Semana 3)
```bash
# Se quiser testar Gemini em subset pequeno
uv run python scripts/check_gemini_free_tier.py  # Avisos sobre limitações
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --judges gemini-2.5-flash-lite \
  --limit 20 \
  --retry-failed

# Usar <1% da quota diária
```

### Passo 4: Análise
```bash
# Quando Etapa 3 terminar (ou mesmo antes)
uv run python -m scripts.analysis_report
# Gera gráficos + tabelas em data/analysis/
```

---

## Conclusão

**Para este projeto com 12.171 ferramentas:**

1. **Use Local (Prometheus + Llama)** como primário
   - ✅ 0% de custo
   - ✅ Nenhuma limitação de rate
   - ✅ ~17 dias de tempo total
   - ✅ Resumível indefinidamente

2. **Mantenha Gemini desabilitado** em config/judges.yaml
   - ❌ Free tier é inadequado
   - ❌ ~73 dias ou vai pegar rate limit
   - ❌ Melhor economizar quota para pequenos testes futuros

3. **Se precisar de validação adicional**
   - ✅ Use Gemini para subset ~20 tools (<1% quota)
   - ✅ Ou upgrade para plan pago
   - ✅ Ou use Claude/OpenAI (com custo ~$8-15)

Veja `MIGRATION_LOCAL_JUDGES.md` para instruções de execução.
