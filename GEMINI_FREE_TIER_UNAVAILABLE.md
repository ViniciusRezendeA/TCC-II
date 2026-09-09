# ⚠️ ACHADO CRÍTICO: Gemini Free Tier Indisponível para Novos Usuários

## Status Atual

Ao tentar usar a API Gemini free tier, **todos os modelos principais retornam**:

```
404 NOT_FOUND
This model models/gemini-2.5-flash-lite is no longer available to new users.
Please update your code...
```

**Modelos testados:**
- ❌ `gemini-1.5-flash` — Não encontrado
- ❌ `gemini-2.5-flash` — Indisponível para novos usuários
- ❌ `gemini-2.5-flash-lite` — Indisponível para novos usuários

**Modelos listados mas indisponíveis:**
- gemini-2.5-computer-use-preview-10-2025
- gemini-2.5-flash-image
- gemini-2.5-flash-native-audio-latest

---

## Implicação para Este Projeto

### ❌ NÃO é possível usar Gemini free tier

| Tentativa | Resultado |
|-----------|-----------|
| Avaliação com Gemini free | ❌ Indisponível (404) |
| Dataset completo | ❌ Impossível |
| Validação de subset | ❌ Impossível |

---

## Alternativas Disponíveis

### ✅ RECOMENDADO: Estratégia Local (FUNCIONANDO)

**Use Prometheus + Llama locais via llama-server:**

```bash
# Setup (1h)
uv run python scripts/check_local_llm_servers.py

# Execução (17 dias)
uv run python -m mcp_pipeline.pipeline.run_step3

# Resultado: 0% de custo, sem limites
```

**Status**: ✅ Totalmente funcional

---

### 💰 ALTERNATIVA: Plano Pago Google (Se Necessário)

Para usar Gemini com plano pago:

1. Upgrade para plano pago em https://aistudio.google.com
2. Diferentes modelos podem estar disponíveis
3. Custos aplicáveis

**Observação**: Mesmo com plano pago, a quota de 1M tokens/dia (free tier) é inadequada. Planos pagos têm limites maiores mas também custos significativos.

---

### 💳 OUTRA ALTERNATIVA: Claude ou OpenAI (Pago)

Se preferir Cloud para redundância:

```bash
# Usar Claude + OpenAI em paralelo com local
# Custo: ~$8-15 para dataset completo
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --judges prometheus-7b-v2.0,llama-uncensored,claude-haiku-4-5,gpt-4.1-mini
```

**Custo estimado**:
- Claude: ~$2.92 (36.5M tokens × $0.08/1M)
- OpenAI: ~$5.48 (36.5M tokens × $0.15/1M)
- **Total**: ~$8.40 para dataset completo

---

## Configuração Recomendada

```yaml
# config/judges.yaml - CONFIGURAÇÃO FINAL

judges:
  # ✅ LOCAL (FUNCIONA, RECOMENDADO)
  - id: prometheus-7b-v2.0
    provider: prometheus
    model_id: prometheus-7b-v2.0
    enabled: true

  - id: llama-uncensored
    provider: llama
    model_id: llama-uncensored
    enabled: true

  # ❌ GEMINI (INDISPONÍVEL para novos usuários)
  - id: gemini-2.5-flash-lite
    provider: google
    model_id: gemini-2.5-flash-lite
    enabled: false  # Não funciona

  # 💳 OPCIONAL (PAGO, se quiser redundância)
  # - id: claude-haiku-4-5
  #   provider: anthropic
  #   enabled: false

  # - id: gpt-4.1-mini
  #   provider: openai
  #   enabled: false
```

**Execução:**
```bash
# Apenas local (0% custo)
uv run python -m mcp_pipeline.pipeline.run_step3

# Com cloud adicional (custo, se habilitado)
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --judges prometheus-7b-v2.0,llama-uncensored,claude-haiku-4-5
```

---

## Conclusão

| Opção | Disponível | Custo | Tempo | Recomendação |
|-------|-----------|-------|-------|--------------|
| **Local (Prometheus + Llama)** | ✅ Sim | $0 | 17 dias | ✅ USE ISTO |
| **Gemini free tier** | ❌ Não (404) | $0 | N/A | ❌ Não funciona |
| **Gemini pago** | ❓ Provável | $$ | ? | ⚠️ Se precisar Cloud |
| **Claude + OpenAI** | ✅ Sim | $8-15 | 1-2 dias | ✅ Se quiser rápido |

---

## Ação Recomendada

1. **Use estratégia LOCAL** (Prometheus + Llama)
   ```bash
   uv run python -m mcp_pipeline.pipeline.run_step3
   ```

2. **Se precisar de redundância**, adicione Claude ou OpenAI (com custo)

3. **NÃO tente usar Gemini free tier** (indisponível)

4. Consulte [MIGRATION_LOCAL_JUDGES.md](MIGRATION_LOCAL_JUDGES.md) para instruções de setup

---

**Data**: 2026-09-08  
**Status**: Confirmado experimentalmente  
**Implicação**: Valida análise anterior que Gemini free tier era inadequado — agora é também indisponível
