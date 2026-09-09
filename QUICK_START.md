# 🚀 Quick Start: Executar Etapa 3 com Juízes Locais

## Pré-requisitos ✅

```bash
# 1. Credenciais configuradas em .env
cp .env.example .env
# Editar com endpoints/tokens do llama-server

# 2. Servidores locais rodando
uv run python scripts/check_local_llm_servers.py
# ✓ Acesso OK (ambos Qwen + Llama)
```

---

## Opção 1: Apenas Qwen (Recomendado para Começar)

**Aderência a instruções superior entre os juízes locais disponíveis — o juiz é
sempre escolhido via `--judges`, nunca hardcoded em script.**

```bash
# Teste rápido (10 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 10

# Validação (100 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 100

# Dataset completo (12.171 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct
```

**Monitorar:**
```bash
tail -f logs/step3.log  # Progresso em tempo real
tail -f logs/step3_errors_qwen2.5-14b-instruct.jsonl  # Erros (se houver)
```

---

## Opção 2: Apenas Llama (Alternativa)

**Modelo geral, para comparação/diversidade**

```bash
# Teste rápido (10 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 10

# Validação (100 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 100

# Dataset completo (12.171 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored
```

---

## Opção 3: Ambos em Paralelo ⚡

**Roda Qwen + Llama em paralelo**

```bash
# Teste rápido (10 tools com 2 juízes)
uv run python -m mcp_pipeline.pipeline.run_step3 --limit 10

# Validação (100 tools com 2 juízes)
uv run python -m mcp_pipeline.pipeline.run_step3 --limit 100

# Dataset completo (12.171 tools com 2 juízes)
uv run python -m mcp_pipeline.pipeline.run_step3
```

**Resultado em 2 arquivos:**
- `data/evaluations/qwen2.5-14b-instruct.jsonl`
- `data/evaluations/llama-uncensored.jsonl`

---

## 📊 Comparação Rápida

| Comando | Juiz(es) | Casos de uso |
|---------|----------|--------------|
| `--judges qwen2.5-14b-instruct` | 1 | ✅ Começar aqui |
| `--judges llama-uncensored` | 1 | Alternativa |
| (sem --judges) | 2 | ⚡ Roda ambos em paralelo |

Tempo por lote depende do hardware que serve cada llama-server — meça com o
teste de 10 tools de cada opção antes de projetar a duração do dataset completo.

---

## Flags Úteis

```bash
# Limitar quantidade de tools
--limit N                          # Processa apenas N tools

# Escolher juízes
--judges qwen2.5-14b-instruct      # Apenas Qwen
--judges llama-uncensored          # Apenas Llama
--judges qwen2.5-14b-instruct,llama-uncensored  # Ambos (sem --judges usa padrão: ambos)

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
# Teste rápido: Qwen + apenas descrição + 5 tools
uv run python -m mcp_pipeline.pipeline.run_step3 \
  --judges qwen2.5-14b-instruct \
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
tail -f logs/step3_errors_qwen2.5-14b-instruct.jsonl
tail -f logs/step3_errors_llama-uncensored.jsonl

# Ver resultados (JSON, um por linha)
head -5 data/evaluations/qwen2.5-14b-instruct.jsonl | jq .

# Contar avaliações concluídas
wc -l data/evaluations/qwen2.5-14b-instruct.jsonl
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
| Conexão recusada (Qwen/Llama) | Verificar se servidores estão rodando |
| Timeout na chamada | `openai_compatible_judge.py` já roda sem limite de timeout por padrão |
| Muitas refusas de segurança | Normal para MCP servers com ferramentas de pentesting |
| Precisa ver dados de um tool específico | Buscar em `data/evaluations/*.jsonl` com `jq` ou `grep` |

---

## Recomendação

**Para 1ª vez:**
```bash
# Teste rápido: 10 tools com Qwen
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 10

# Se OK, validação: 100 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 100

# Se OK, dataset completo com ambos juízes
uv run python -m mcp_pipeline.pipeline.run_step3
```

---

Veja [MIGRATION_LOCAL_JUDGES.md](MIGRATION_LOCAL_JUDGES.md) para detalhes completos.
