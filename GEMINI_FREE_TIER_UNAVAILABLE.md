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

**Use Qwen + Llama locais via llama-server:**

```bash
# Setup (1h)
uv run python scripts/check_local_llm_servers.py

# Execução (tempo depende do hardware)
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

---

## Configuração Recomendada

```yaml
# config/judges.yaml - CONFIGURAÇÃO FINAL

judges:
  # ✅ LOCAL (FUNCIONA, RECOMENDADO)
  - id: qwen2.5-14b-instruct
    provider: qwen
    model_id: qwen2.5-14b-instruct
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
```

**Execução:**
```bash
# Apenas local (0% custo)
uv run python -m mcp_pipeline.pipeline.run_step3
```

---

## Conclusão

| Opção | Disponível | Custo | Recomendação |
|-------|-----------|-------|--------------|
| **Local (Qwen + Llama)** | ✅ Sim | $0 | ✅ USE ISTO |
| **Gemini free tier** | ❌ Não (404) | $0 | ❌ Não funciona |
| **Gemini pago** | ❓ Provável | $$ | ⚠️ Se precisar Cloud |

---

## Ação Recomendada

1. **Use estratégia LOCAL** (Qwen + Llama)
   ```bash
   uv run python -m mcp_pipeline.pipeline.run_step3
   ```

2. **NÃO tente usar Gemini free tier** (indisponível)

3. Consulte [MIGRATION_LOCAL_JUDGES.md](MIGRATION_LOCAL_JUDGES.md) para instruções de setup

---

**Data**: 2026-09-08  
**Status**: Confirmado experimentalmente  
**Implicação**: Valida análise anterior que Gemini free tier era inadequado — agora é também indisponível
