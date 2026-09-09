# Migração para Juízes Locais (Qwen + Llama)

## Resumo da Mudança

O pipeline da Etapa 3 foi expandido para suportar juízes LLM rodando **localmente** em sua rede, usando servidores llama.cpp (llama-server):

- **Qwen2.5-14B-Instruct**: Aderência a instruções superior entre os juízes locais testados (ver `config/judges.yaml`)
- **Llama (llama-uncensored)**: Alternativa geral para diversidade de avaliação

Ambos usam API compatível com OpenAI, rodando em máquinas separadas na rede local.

## Pré-requisitos

1. **Servidores llama.cpp em execução**:
   - Qwen2.5-14B-Instruct em `http://<IP>:<PORTA>/v1` (endereço configurado em `QWEN_LLM_BASE_URL`)
   - Llama em `http://192.168.15.15:8090/v1`
   - Ambos com autenticação Bearer

2. **Python 3.12+** e **uv** já instalados

3. **Dataset pronto**: `data/dataset.jsonl` com 12.171 tools extraídas

## Passo 1: Configurar Credenciais

```bash
# Copie o template:
cp .env.example .env

# Edite .env com seus dados reais:
QWEN_LLM_BASE_URL=http://<IP>:<PORTA>/v1
QWEN_LLM_BEARER_TOKEN=seu_token_qwen_aqui
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
llama               ✓ OK
qwen                ✓ OK

✓ Todos os servidores estão acessíveis!
```

Se falhar, verifique:
- Servidores llama.cpp estão rodando nos endereços configurados em `.env`
- As portas estão acessíveis na rede local
- Tokens Bearer estão corretos em `.env`

## Passo 3: Teste Piloto (20 Tools)

```bash
# Teste ambos os juízes com 20 tools aleatórios:
uv run python scripts/test_local_judges.py --sample-size 20 --judge both

# Apenas Qwen:
uv run python scripts/test_local_judges.py --sample-size 20 --judge qwen

# Apenas Llama:
uv run python scripts/test_local_judges.py --sample-size 20 --judge llama
```

**Esperado:**
- Latência: varia com o hardware que serve cada llama-server -- meça no piloto antes de projetar o lote completo
- Taxa de sucesso: 100% (se houver refusas, é normal)
- Log em: `data/test_local_judges_results.jsonl`

Exemplos de saída:
```
✓ qwen2.5-14b-instruct apache__mcp:get_repo_stats latency=18250ms tokens=1823→198
✓ llama-uncensored     apache__mcp:list_files latency=8945ms tokens=1645→142
⚠ qwen2.5-14b-instruct pentesting__mcp:execute_payload RECUSADO: safety
```

## Passo 4: Execução da Etapa 3

Após validar no teste piloto, execute a Etapa 3 completa. O juiz é sempre
escolhido via `--judges <id>` -- nunca hardcoded em script.

### ✅ APENAS QWEN (Recomendado para começar)

```bash
# Teste: 10 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 10

# Validação: 100 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 100

# Dataset completo (12.171 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct
```

**Vantagem**: maior aderência a instruções entre os juízes locais disponíveis
**Tempo estimado**: meça com o teste de 10/100 tools antes de projetar o dataset completo -- depende do hardware que serve o llama-server

### ✅ APENAS LLAMA

```bash
# Teste: 10 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 10

# Validação: 100 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 100

# Dataset completo (12.171 tools)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored
```

**Vantagem**: Alternativa para comparar resultados

### ✅ AMBOS EM PARALELO

```bash
# Teste: 10 tools com ambos juízes
uv run python -m mcp_pipeline.pipeline.run_step3 --limit 10

# Validação: 100 tools com ambos juízes
uv run python -m mcp_pipeline.pipeline.run_step3 --limit 100

# Dataset completo com ambos (12.171 tools)
# Máximo paralelismo, resultado em 2 arquivos (1 por juiz)
uv run python -m mcp_pipeline.pipeline.run_step3
```

**Vantagem**: Roda Qwen + Llama em paralelo -- cada juiz avança seu próprio checkpoint independentemente

---

## Monitoramento

Durante a execução, monitore:

```bash
# Ver progresso em tempo real:
tail -f logs/step3.log

# Ver erros específicos:
tail -f logs/step3_errors_qwen2.5-14b-instruct.jsonl
tail -f logs/step3_errors_llama-uncensored.jsonl

# Ver resultados salvos:
ls -lah data/evaluations/
```

Cada arquivo de saída contém um JSON por linha:
```bash
# Ver primeiros 5 resultados do Qwen:
head -5 data/evaluations/qwen2.5-14b-instruct.jsonl | jq .
```

## Resumibilidade

A execução é **resumível** automaticamente. Se interromper (Ctrl+C):

1. Os resultados já gravados **não** são perdidos
2. Na próxima execução, o pipeline retoma a partir de onde parou
3. Use `--retry-failed` para re-tentar ferramentas que deram erro

```bash
# Retomar Qwen do ponto onde parou:
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct

# Retomar e retentar falhas:
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --retry-failed
```

## Solução de Problemas

### Erro: "conexão recusada ao qwen2.5-14b-instruct"
- Verifique se o servidor no endereço de `QWEN_LLM_BASE_URL` está rodando
- Teste manualmente: `curl -H "Authorization: Bearer TOKEN" <QWEN_LLM_BASE_URL>/models`

### Erro: "timeout ao chamar qwen2.5-14b-instruct"
- `openai_compatible_judge.py` já roda sem limite de timeout por padrão (`timeout_seconds=None`) -- um timeout aqui normalmente indica que o llama-server caiu, não que está lento
- Se quiser reimpor um limite, ajuste `_timeout_seconds` na classe `QwenJudge`

### Erro: "resposta não valida contra RubricScores"
- O modelo respondeu em formato inválido
- Verifique se o prompt de sistema está chegando correto
- Use `--retry-failed` após revisar o modelo

### Muitas refusas de segurança (status="refused")
- Normal para MCP servers com ferramentas de segurança/pentesting
- O resultado "refused" é registrado corretamente, não é erro
- Contagem em logs indica quantas foram refusadas vs ok

## Comparação: Local vs Cloud

| Aspecto | Local (Qwen + Llama) | Cloud (Claude + GPT + Gemini) |
|---------|---------------------------|-------------------------------|
| Custo | $0 (seu hardware) | $$ por API call |
| Modelos | 14B, 7B | Haiku, Mini, Flash |
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

Use local (Qwen + Llama) para dataset completo, sem custos.

Veja: `uv run python scripts/check_gemini_free_tier.py` para análise de capacidade.

## Regressão: Voltar para Cloud Judges

Se precisar voltar aos juízes originais (Claude, OpenAI, Gemini):

```bash
# Editar config/judges.yaml e marcar como enabled:
# - claude-haiku-4-5: enabled: true
# - gpt-4.1-mini: enabled: true
# - qwen2.5-14b-instruct: enabled: false
# - llama-uncensored: enabled: false

# Então rodar:
uv run python -m mcp_pipeline.pipeline.run_step3
```

**Aviso**: Veja [JUDGES_STRATEGY.md](JUDGES_STRATEGY.md) sobre custos e limitações de cada estratégia.

## Questões Frequentes

**P: Posso rodar Qwen e Llama em paralelo?**
R: Sim! Use `--judges qwen2.5-14b-instruct,llama-uncensored`. O pipeline fará concorrência por juiz.

**P: Quanto tempo leva?**
R: Depende do hardware que serve cada llama-server -- rode o teste de 10-100 tools de cada juiz e projete a partir da latência medida, em vez de uma estimativa fixa.

**P: Posso mudar de modelo?**
R: Sim. Edite `config/judges.yaml` com novo `model_id` e rode novamente.

**P: Os resultados são compatíveis com cloud?**
R: Sim! Mesma estrutura de rubrica (6 componentes, Likert 1-5).

**P: Preciso de GPU?**
R: Depende do servidor llama.cpp. Se já está rodando, seu setup funciona.

---

Para mais detalhes, veja [README.md](README.md#rodando-a-etapa-3).
