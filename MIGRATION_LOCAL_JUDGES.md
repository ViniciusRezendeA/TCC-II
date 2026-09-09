# Migração para Juízes Locais (Prometheus + Llama)

## Resumo da Mudança

O pipeline da Etapa 3 foi expandido para suportar juízes LLM rodando **localmente** em sua rede, usando servidores llama.cpp (llama-server):

- **Prometheus 7B V2.0**: Especializado em avaliação de respostas de LLMs (LLM-as-a-Judge)
- **Llama (llama-uncensored)**: Alternativa geral para diversidade de avaliação

Ambos usam API compatível com OpenAI, rodando em máquinas separadas na rede local.

## Pré-requisitos

1. **Servidores llama.cpp em execução**:
   - Prometheus 7B V2.0 em `http://192.168.15.15:8091/v1`
   - Llama em `http://192.168.15.15:8090/v1`
   - Ambos com autenticação Bearer

2. **Python 3.12+** e **uv** já instalados

3. **Dataset pronto**: `data/dataset.jsonl` com 12.171 tools extraídas

## Passo 1: Configurar Credenciais

```bash
# Copie o template:
cp .env.example .env

# Edite .env com seus dados reais:
PROMETHEUS_LLM_BASE_URL=http://192.168.15.15:8091/v1
PROMETHEUS_LLM_BEARER_TOKEN=seu_token_prometheus_aqui
LOCAL_LLM_BASE_URL=http://192.168.15.15:8090/v1
LOCAL_LLM_BEARER_TOKEN=seu_token_llama_aqui
```

## Passo 2: Validar Conectividade

```bash
# Health check dos servidores locais:
uv run python scripts/check_local_llm_servers.py

# (Opcional) Health check do Gemini free tier:
# Recomendado para entender limitações e capacidade diária
uv run python scripts/check_gemini_free_tier.py
```

**Resultado esperado:**
```
============================================================
RESUMO
============================================================
prometheus          ✓ OK
llama               ✓ OK

✓ Todos os servidores estão acessíveis!
```

Se falhar, verifique:
- Servidores llama.cpp estão rodando em `192.168.15.15`
- Portas 8090 e 8091 estão acessíveis
- Tokens Bearer estão corretos em `.env`

## Passo 3: Teste Piloto (20 Tools)

```bash
# Teste ambos os juízes com 20 tools aleatórios:
uv run python scripts/test_local_judges.py --sample-size 20 --judge both

# Apenas Prometheus:
uv run python scripts/test_local_judges.py --sample-size 20 --judge prometheus

# Apenas Llama:
uv run python scripts/test_local_judges.py --sample-size 20 --judge llama
```

**Esperado:**
- Latência: 5-30s por tool (depende do modelo)
- Taxa de sucesso: 100% (se houver refusas, é normal)
- Log em: `data/test_local_judges_results.jsonl`

Exemplos de saída:
```
✓ prometheus-7b-v2.0 apache__mcp:get_repo_stats latency=12250ms tokens=1823→156
✓ llama-uncensored    apache__mcp:list_files latency=8945ms tokens=1645→142
⚠ prometheus-7b-v2.0 pentesting__mcp:execute_payload RECUSADO: safety
```

## Passo 4: Execução da Etapa 3

Após validar no teste piloto, execute a Etapa 3 completa:

### Apenas Prometheus (recomendado para começar):
```bash
# 100 tools primeiros:
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0 --limit 100

# Dataset completo (12.171 tools):
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0
```

### Apenas Llama:
```bash
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 100
```

### Ambos (mais lento, ~2x tempo):
```bash
uv run python -m mcp_pipeline.pipeline.run_step3
```

## Monitoramento

Durante a execução, monitore:

```bash
# Ver progresso em tempo real:
tail -f logs/step3.log

# Ver erros específicos:
tail -f logs/step3_errors_prometheus-7b-v2.0.jsonl
tail -f logs/step3_errors_llama-uncensored.jsonl

# Ver resultados salvos:
ls -lah data/evaluations/
```

Cada arquivo de saída contém um JSON por line:
```bash
# Ver primeiros 5 resultados do Prometheus:
head -5 data/evaluations/prometheus-7b-v2.0.jsonl | jq .
```

## Resumibilidade

A execução é **resumível** automaticamente. Se interromper (Ctrl+C):

1. Os resultados já gravados **não** são perdidos
2. Na próxima execução, o pipeline retoma a partir de onde parou
3. Use `--retry-failed` para re-tentar ferramentas que deram erro

```bash
# Retomar Prometheus do ponto onde parou:
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0

# Retomar e retentar falhas:
uv run python -m mcp_pipeline.pipeline.run_step3 --judges prometheus-7b-v2.0 --retry-failed
```

## Solução de Problemas

### Erro: "conexão recusada ao prometheus-7b-v2.0"
- Verifique se o servidor em `192.168.15.15:8091` está rodando
- Teste manualmente: `curl -H "Authorization: Bearer TOKEN" http://192.168.15.15:8091/v1/models`

### Erro: "timeout ao chamar prometheus-7b-v2.0"
- Modelo pode estar processando lentamente
- Aumente timeout em `src/mcp_pipeline/evaluation/judges/openai_compatible_judge.py` (padrão: 300s)
- Ou reduza `--limit` para testar com fewer tools

### Erro: "resposta não valida contra RubricScores"
- O modelo respondeu em formato inválido
- Verifique se o prompt de sistema está chegando correto
- Use `--retry-failed` após revisar o modelo

### Muitas refusas de segurança (status="refused")
- Normal para MCP servers com ferramentas de segurança/pentesting
- O resultado "refused" é registrado corretamente, não é erro
- Contagem em logs indica quantas foram refusadas vs ok

## Comparação: Local vs Cloud

| Aspecto | Local (Prometheus + Llama) | Cloud (Claude + GPT + Gemini) |
|---------|---------------------------|-------------------------------|
| Custo | $0 (seu hardware) | $$ por API call |
| Latência | ~10-30s/tool | ~5-15s/tool |
| Modelos | 7B, 7B | Haiku, Mini, Flash |
| Customização | Controle total | Limitado |
| Privacidade | Local | Enviado à API |
| Escala | Limitada pelo hardware | Ilimitada |

## Próximos Passos

1. ✅ Valide conectividade: `python scripts/check_local_llm_servers.py`
2. ✅ Teste piloto: `python scripts/test_local_judges.py --sample-size 20`
3. ✅ Execute Etapa 3: `python -m mcp_pipeline.pipeline.run_step3 --limit 100`
4. ✅ Análise: `python -m scripts.analysis_report` (quando Etapa 3 terminar)

## ⚠️ Sobre Gemini Free Tier

**Importante**: Se está considerando usar Gemini (Google), veja [JUDGES_STRATEGY.md](JUDGES_STRATEGY.md) para análise completa.

**Resumo**: Free tier de Gemini é inadequado para dataset completo (12.171 tools):
- Quota: 1M tokens/dia (levaria ~73 dias)
- Rate limit: 15 req/min (muito lento)
- Recomendação: **Mantenha desabilitado em config/judges.yaml**

Use local (Prometheus + Llama) para dataset completo, sem custos.

Veja: `uv run python scripts/check_gemini_free_tier.py` para análise de capacidade.

## Regressão: Voltar para Cloud Judges

Se precisar voltar aos juízes originais (Claude, OpenAI, Gemini):

```bash
# Editar config/judges.yaml e marcar como enabled:
# - claude-haiku-4-5: enabled: true
# - gpt-4.1-mini: enabled: true
# - prometheus-7b-v2.0: enabled: false
# - llama-uncensored: enabled: false

# Então rodar:
uv run python -m mcp_pipeline.pipeline.run_step3
```

**Aviso**: Veja [JUDGES_STRATEGY.md](JUDGES_STRATEGY.md) sobre custos e limitações de cada estratégia.

## Questões Frequentes

**P: Posso rodar Prometheus e Llama em paralelo?**
R: Sim! Use `--judges prometheus-7b-v2.0,llama-uncensored`. O pipeline fará concorrência por juiz.

**P: Quanto tempo leva?**
R: ~12.171 tools × 30s/tool ÷ threads = ~36-60 horas para dataset completo com 1 juiz.

**P: Posso mudar de modelo?**
R: Sim. Edite `config/judges.yaml` com novo `model_id` e rode novamente.

**P: Os resultados são compatíveis com cloud?**
R: Sim! Mesma estrutura de rubrica (6 componentes, Likert 1-5).

**P: Preciso de GPU?**
R: Depende do servidor llama.cpp. Se já está rodando, seu setup funciona.

---

Para mais detalhes, veja [README.md](README.md#rodando-a-etapa-3).
