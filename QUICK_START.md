# 🚀 Quick Start: Executar Etapa 3 com Juízes Locais

## Pré-requisitos ✅

```bash
# 1. Credenciais configuradas em .env
cp .env.example .env
# Editar com endpoints/tokens do llama-server

# 2. Servidores locais rodando
uv run python scripts/check_local_llm_servers.py
# ✓ Acesso OK (ambos Prometheus + Llama)
```

---

## Opção 1: Apenas Prometheus (Recomendado para Começar)

**Especializado em LLM-as-a-Judge, mais rápido**

```bash
# Teste rápido (10 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0 --limit 10

# Validação (100 tools, ~2 horas)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0 --limit 100

# Dataset completo (12.171 tools, ~8-10 dias)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0
```

**Monitorar:**
```bash
tail -f logs/step3.log  # Progresso em tempo real
tail -f logs/step3_errors_prometheus-7b-v2.0.jsonl  # Erros (se houver)
```

---

## Opção 2: Apenas Llama (Alternativa)

**Modelo geral, para comparação/diversidade**

```bash
# Teste rápido (10 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 10

# Validação (100 tools, ~2-3 horas)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 100

# Dataset completo (12.171 tools, ~10-12 dias)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored
```

---

## Opção 3: Ambos em Paralelo ⚡ (MAIS RÁPIDO)

**Roda Prometheus + Llama em paralelo, metade do tempo**

```bash
# Teste rápido (10 tools com 2 juízes)
uv run python -m mcp_pipeline.pipeline.run_step3 --limit 10

# Validação (100 tools com 2 juízes, ~1-2 horas)
uv run python -m mcp_pipeline.pipeline.run_step3 --limit 100

# Dataset completo (12.171 tools com 2 juízes, ~5-7 dias) ⚡
uv run python -m mcp_pipeline.pipeline.run_step3
```

**Resultado em 2 arquivos:**
- `data/evaluations/prometheus-7b-v2.0.jsonl`
- `data/evaluations/llama-uncensored.jsonl`

---

## 📊 Comparação Rápida

| Comando | Juiz(es) | Tempo | Casos de uso |
|---------|----------|-------|--------------|
| `--judges prometheus-7b-v2.0` | 1 | 8-10 dias | ✅ Começar aqui |
| `--judges llama-uncensored` | 1 | 10-12 dias | Alternativa |
| (sem --judges) | 2 | 5-7 dias | ⚡ **Recomendado** |

---

## Flags Úteis

```bash
# Limitar quantidade de tools
--limit N                          # Processa apenas N tools

# Escolher juízes
--judges prometheus-7b-v2.0        # Apenas Prometheus
--judges llama-uncensored          # Apenas Llama
--judges prometheus-7b-v2.0,llama  # Ambos (sem --judges usa padrão: ambos)

# Cenários
--scenarios description_only       # Apenas descrição (sem código)
--scenarios with_source            # Apenas com código
--scenarios description_only,with_source  # Ambos (padrão)

# Retry
--retry-failed                     # Retentar tools que falharam antes
```

---

## Exemplos Combinados

```bash
# Teste rápido: Prometheus + apenas descrição + 5 tools
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --judges prometheus-7b-v2.0 \
  --scenarios description_only \
  --limit 5

# Validação: Ambos juízes + ambos cenários + 50 tools
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --limit 50

# Execução completa: Ambos juízes + retentar falhas
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --retry-failed
```

---

## Monitorar Execução

```bash
# Ver logs em tempo real
tail -f logs/step3.log

# Ver erros específicos de um juiz
tail -f logs/step3_errors_prometheus-7b-v2.0.jsonl
tail -f logs/step3_errors_llama-uncensored.jsonl

# Ver resultados (JSON, um por linha)
head -5 data/evaluations/prometheus-7b-v2.0.jsonl | jq .

# Contar avaliações concluídas
wc -l data/evaluations/prometheus-7b-v2.0.jsonl
wc -l data/evaluations/llama-uncensored.jsonl
```

---

## Parar e Retomar

```bash
# Parar execução (Ctrl+C)
# Próxima execução retoma de onde parou
# Nenhuma perda de dados

# Se quiser reprocessar tools que falharam:
uv run python -m mcp_pipeline.pipeline.run_step3 --retry-failed

# Se quiser começar do zero (limpar checkpoints):
rm state/step3_progress.json
```

---

## Pós-Execução

```bash
# Gerar análise/gráficos quando Etapa 3 terminar
uv run python -m scripts.analysis_report

# Outputs em: data/analysis/charts/ e data/analysis/tables/
```

---

## Troubleshooting Rápido

| Problema | Solução |
|----------|---------|
| Conexão recusada (Prometheus/Llama) | Verificar se servidores estão rodando |
| Timeout na chamada | Aumentar timeout em `openai_compatible_judge.py` |
| Taxa muito lenta | Usar 2 juízes em paralelo: `uv run python -m mcp_pipeline.pipeline.run_step3` |
| Muitas refusas de segurança | Normal para MCP servers com ferramentas de pentesting |
| Precisa ver dados de um tool específico | Buscar em `data/evaluations/*.jsonl` com `jq` ou `grep` |

---

## Recomendação

**Para 1ª vez:**
```bash
# Teste rápido: 10 tools com Prometheus
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0 --limit 10

# Se OK, validação: 100 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0 --limit 100

# Se OK, dataset completo com ambos juízes
uv run python -m mcp_pipeline.pipeline.run_step3
```

---

Veja [MIGRATION_LOCAL_JUDGES.md](MIGRATION_LOCAL_JUDGES.md) para detalhes completos.
