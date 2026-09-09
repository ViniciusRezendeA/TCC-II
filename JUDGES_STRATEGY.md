# Estratégia de Juízes para Etapa 3

## Análise de Capacidade e Custos

Este documento compara as estratégias de juízes (Local, Misto) com foco nas
limitações do Gemini free tier. Claude e GPT não são usados neste projeto —
o único juiz cloud disponível é o Gemini free tier, e ele próprio é inadequado
para o dataset completo (ver análise abaixo).

### Dataset

- **Total de ferramentas**: 12.171
- **Cenários por ferramenta**: 2 (description_only, with_source)
- **Total de avaliações necessárias**: 24.342
- **Tokens por avaliação (estimado)**: 3.000 (1.500 input + 1.500 output)
- **Tokens totais necessários**: ~73M

---

## Estratégia 1: 100% Local (Recomendado)

### Juízes
- Qwen2.5-14B-Instruct (llama-server, endereço em QWEN_LLM_BASE_URL)
- Llama uncensored (llama-server em 192.168.15.15:8090)

### Capacidade Diária
| Métrica | Valor |
|---------|-------|
| Custo API | $0 |
| Juízes ativos | 2 |
| Requisições/min | Ilimitado (seu hardware) |
| Latência/ferramenta | Depende do hardware -- meça com o teste piloto antes de projetar o lote completo |

### Vantagens
✅ Nenhum custo de API
✅ Nenhum limite de rate
✅ Dados permanecem locais
✅ Controle total sobre modelos
✅ Resumível indefinidamente

### Desvantagens
❌ Depende do hardware local
❌ Nenhuma redundância se servidor cair

### Recomendação
**IDEAL para dataset completo**. Configure assim:

```yaml
# config/judges.yaml
judges:
  - id: qwen2.5-14b-instruct
    provider: qwen
    model_id: qwen2.5-14b-instruct
    enabled: true

  - id: llama-uncensored
    provider: llama
    model_id: llama-uncensored
    enabled: true
```

Execução:
```bash
uv run python -m mcp_pipeline.pipeline.run_step3
# meça a duração com o teste de 10-100 tools antes de projetar o dataset completo
```

---

## Estratégia 2: Misto (Local + Gemini para validação seletiva)

### Recomendação Otimizada

Use **Qwen + Llama locais** como primários + **Gemini para validação pequena**
(nunca para o dataset completo -- ver limitações abaixo).

```yaml
# config/judges.yaml - RECOMENDADO
judges:
  # Primários (local) - rápidos, sem limite
  - id: qwen2.5-14b-instruct
    provider: qwen
    enabled: true

  - id: llama-uncensored
    provider: llama
    enabled: true

  # Secundário (validação apenas) - desabilitado por padrão
  - id: gemini-2.5-flash-lite
    provider: google
    enabled: false  # ← Ativar apenas para testes pequenos
```

### Execução

**Fase 1: Dataset completo (local)**
```bash
# Avalia todas as 24.342 avaliações com Qwen + Llama
# tempo depende do hardware do llama-server, nenhum custo
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
| Juízes primários | 2 locais |
| Juízes validação | 1 cloud (opcional) |
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
| Dataset completo (24k tools) | Local (Qwen + Llama) | Não há alternativa cloud viável sem custo |
| Testes/desenvolvimento | Local (--limit 10-20) | Gemini (--limit 5) |
| Validação seletiva | Local, depois Gemini (~20 tools) | — |
| Produção/reprodutibilidade | Local | — |
| Zero custo | Local ✅ | Não existe alternativa cloud gratuita adequada |

---

## Plano de Implementação Recomendado

### Passo 1: Setup Local
```bash
uv run python scripts/check_local_llm_servers.py
uv run python scripts/test_local_judges.py --sample-size 20
```

### Passo 2: Execução Primária
```bash
# Roda com Qwen + Llama
uv run python -m mcp_pipeline.pipeline.run_step3

# Monitorar:
tail -f logs/step3.log
```

### Passo 3: Validação Opcional
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

1. **Use Local (Qwen + Llama)** como primário
   - ✅ 0% de custo
   - ✅ Nenhuma limitação de rate
   - ✅ Resumível indefinidamente

2. **Mantenha Gemini desabilitado** em config/judges.yaml
   - ❌ Free tier é inadequado
   - ❌ ~73 dias ou vai pegar rate limit
   - ❌ Melhor economizar quota para pequenos testes futuros

3. **Se precisar de validação adicional**
   - ✅ Use Gemini para subset ~20 tools (<1% quota)
   - ✅ Ou upgrade para plano pago do Gemini

Veja `MIGRATION_LOCAL_JUDGES.md` para instruções de execução.
