# Pipeline de Mineração e Extração MCP (TCC)

Implementa as **Etapas 1, 2 e 3** da metodologia do TCC: coleta de
repositórios de servidores MCP no GitHub (via GraphQL), extração das
ferramentas (tools) e seus call graphs de 3 níveis a partir do código-fonte,
e classificação da qualidade das descrições via um júri multi-provedor de
LLMs (Google Gemini, Qwen e Llama locais via llama.cpp).

Ver o plano completo de implementação em
`~/.claude/plans/section-materiais-e-m-todos-label-sec-m-ticklish-panda.md`
para o desenho detalhado, decisões e riscos conhecidos.

## Status

- ✅ **Etapa 1** (mineração via GraphQL): implementada, testada e validada
  contra a API real do GitHub (206 repositórios selecionados).
- ✅ **Clonagem de repositórios**: implementada, testada e validada contra os
  206 repos (0 falhas).
- ✅ **Etapa 2** (detecção de tools + call graph): implementada e validada
  ponta a ponta contra os 206 repositórios reais, cobrindo **todas as 10
  `target_languages`** configuradas (`config/mcp_signals.yaml`): o padrão de
  alto nível de cada SDK oficial — Python (`@mcp.tool()`), TypeScript/
  JavaScript (`.tool()`/`.registerTool()`), Java (Spring AI `@Tool`), C#
  (`[McpServerTool]`), Rust (`#[tool]`/`rmcp`), Dart (`registerTool(Tool(...))`
  do `dart_mcp`), Go (`mcp.AddTool(...)` do `go-sdk`) e Kotlin
  (`addTool(...)` do `kotlin-sdk`) — **e** três padrões adicionais
  genuinamente compartilhados por múltiplos projetos: o padrão de baixo
  nível oficial (Python `@server.list_tools()`, TS/JS `setRequestHandler`),
  o pacote de terceiros `fastmcp` do npm, e a classe `< ...Tool` da gem
  Ruby `fast-mcp`. Go também resolve `Name:`/`Description:` contra
  constantes de pacote (`const NOME = "literal"`) quando não são string
  literal inline, via o mesmo mecanismo de `ValueIndex` já usado nos
  padrões de baixo nível de Python/TS/JS. **2.968 tools extraídas de 85 dos
  206 repositórios** (`data/dataset.jsonl`). 217 testes automatizados (216
  passando; 1 falha pré-existente em `test_gemini_judge.py`, não
  relacionada à Etapa 2).
  - Cobertura por linguagem (repositórios selecionados → com pelo menos 1
    tool detectada → tools extraídas): TypeScript 62→24→436, Python
    51→27→1853, Java 34→9→355, Rust 19→11→123, C# 10→7→125, Dart 8→1→4,
    Ruby 8→1→1, Kotlin 6→0→0, JavaScript 4→3→58, Go 4→2→13. Kotlin ficou em
    0 por decisão consistente com as demais linguagens (mesmo critério do
    JS/TS): exige nome literal (ou resolvível para um literal, no caso do
    Go) no `name = "..."` do `addTool(...)`, e os 6 repositórios reais da
    amostra usam um valor genuinamente dinâmico (campo de um objeto
    construído em loop, resultado de função) em vez de literal ou
    constante — não confirmado como resolvível estaticamente, ver o plano
    de extensão de linguagens.
  - **Fora de escopo, por decisão documentada**: Swift (sem padrão de
    registro único compartilhado — SDK oficial de baixo nível, cada
    repositório liga a implementação ao `Tool(...)` de um jeito diferente),
    Elixir, Lua e PHP (sem candidatos reais suficientes em `target_languages`
    para justificar o investimento — ver o plano de extensão de linguagens
    para a investigação completa), arquiteturas de registro 100%
    customizadas usadas por um único repositório cada (ex: anotação própria
    do `ghidra-mcp`, os 4 esquemas diferentes de C# encontrados), builder
    oficial do Java (nenhum repositório real da amostra o usa), e escanear
    mais de uma linguagem por repositório.
- ✅ **Etapa 3** (classificação via LLM-as-a-Judge): implementada, testada e expandida
  — júri multi-provedor com suporte a um juiz cloud (Google Gemini 2.5 Flash-Lite)
  e juízes locais (Qwen2.5-14B-Instruct, Llama via llama.cpp). Rubrica de 6
  componentes, execução resumível e concorrente por juiz. Pronto para execução
  local e remota — requer configuração de `.env` com a chave do Gemini (cloud)
  ou endpoints locais (llama-server).

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # se `uv` ainda não estiver instalado
cp .env.example .env                               # preencher com um GitHub PAT (somente leitura, repos públicos)
                                                     # e, para a Etapa 3, as chaves de API dos 3 provedores
uv sync
```

## Rodando a Etapa 1

```bash
# Só valida autenticação e rate limit, sem gastar cota de busca:
uv run python -m mcp_pipeline.collection.run_step1 --smoke-test-only

# Rodada completa: busca, deduplica, filtra, rankeia e salva os candidatos.
# Resumível — pode ser interrompida e re-rodada sem perder progresso
# (estado em state/step1_progress.json, páginas brutas em data/raw/search_pages/).
uv run python -m mcp_pipeline.collection.run_step1
```

Saídas em `data/`:
- `candidate_pool.jsonl` — todos os repositórios que passaram nos filtros (auditoria)
- `selected_repos.jsonl` — top N (206 por padrão) por estrelas

### Clonagem

```bash
# Clona todos os repositórios de selected_repos.jsonl para data/repos/.
# Resumível — pula repositórios já clonados; falhas ficam em logs/clone_errors.jsonl
# e não abortam o lote.
uv run python -m mcp_pipeline.pipeline.run_clone [--limit N]
```

## Rodando a Etapa 2

Pré-requisito: repositórios já clonados (ver seção anterior).

```bash
# Detecta tools + constrói call graphs para todo repositório já clonado.
# Resumível — pula repositórios já processados (tools.jsonl existente) ou
# previamente falhados (a menos que --retry-failed seja passado).
uv run python -m mcp_pipeline.pipeline.run_step2 [--limit N] [--retry-failed]

# Concatena a saída de todos os repositórios em um dataset único.
uv run python -m mcp_pipeline.schema.assemble_dataset
```

Saída final: `data/dataset.jsonl`, um registro por tool (descrição + call
graph estruturado de 3 níveis + metadados do repositório). Para gerar o
payload compatível com Hasan et al. (`{name, server_name, description,
SOURCE_CODE}`, usado pela avaliação via LLM-as-a-Judge da Etapa 3), ver
`schema/export_for_evaluation.py`.

## Rodando a Etapa 3

Pré-requisito: `data/dataset.jsonl` já montado (seção anterior).

### Com juiz cloud (Google Gemini)

Preencha `GOOGLE_API_KEY` em `.env` (só é necessário se o juiz Gemini estiver
habilitado em `config/judges.yaml`).

```bash
uv run python -m mcp_pipeline.pipeline.run_step3 [--limit N] [--judges id1,id2] [--scenarios description_only,with_source] [--retry-failed]
```

### Com juízes locais (Qwen2.5-14B-Instruct + Llama via llama.cpp)

**Setup inicial:**

1. Configure `.env` com os endpoints e tokens dos servidores locais:

```bash
cp .env.example .env
# Edite .env com:
# - QWEN_LLM_BASE_URL=http://192.168.15.15:8091/v1
# - QWEN_LLM_BEARER_TOKEN=seu_token_aqui
# - LOCAL_LLM_BASE_URL=http://192.168.15.15:8090/v1
# - LOCAL_LLM_BEARER_TOKEN=seu_token_aqui
```

2. Valide conectividade com os servidores (ambos devem estar rodando via llama-server):

```bash
uv run python scripts/check_local_llm_servers.py
```

3. Teste com um subset pequeno (20 tools):

```bash
uv run python scripts/test_local_judges.py --sample-size 20 --judge both
```

4. Se o teste passar, execute a Etapa 3 com a configuração desejada:

**Apenas Qwen (recomendado para começar) — o juiz é sempre escolhido via `--judges`, nunca hardcoded em script:**
```bash
# Teste: 10 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 10

# Validação: 100 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct --limit 100

# Dataset completo
uv run python -m mcp_pipeline.pipeline.run_step3 --judges qwen2.5-14b-instruct
```

**Apenas Llama:**
```bash
# Teste: 10 tools
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored --limit 10

# Dataset completo (~10-12 dias)
uv run python -m mcp_pipeline.pipeline.run_step3 --judges llama-uncensored
```

**Ambos em paralelo (mais rápido, recomendado para dataset completo):**
```bash
# Teste: 10 tools com ambos juízes
uv run python -m mcp_pipeline.pipeline.run_step3 --limit 10

# Dataset completo -- tempo depende do modelo carregado no llama-server de cada
# endpoint; meça com o teste de 10 tools acima antes de projetar a duração total.
uv run python -m mcp_pipeline.pipeline.run_step3
```

**Resumibilidade:** A execução é resumível por (tool, cenário, juiz) — uma falha
técnica individual não aborta o lote; recusas de segurança do modelo são
gravadas como resultado próprio (status "refused"), não como erro.

Saída: `data/evaluations/{judge_id}.jsonl`, um registro por (tool, cenário,
juiz) com os 6 componentes da rubrica pontuados em escala Likert de 5 pontos.

## Análise e validação

```bash
# Gera gráficos (PNG) e planilhas (XLSX/CSV) comparando repositórios e tools
# por linguagem, médias por servidor, distribuição por sdk_pattern, taxas de
# resolução do call graph, etc. — lê só data/selected_repos.jsonl e
# data/dataset.jsonl, não depende do código de extração.
uv run python -m scripts.analysis_report
```

Saída em `data/analysis/` (`charts/`, `tables/`) — ver a seção do próprio
script para a lista completa de métricas.

## Testes

```bash
uv run pytest        # suíte de testes (não requer token do GitHub nem chaves de LLM)
uv run ruff check .  # lint
```

## Configuração

- `config/mcp_signals.yaml` — tópicos e sinais textuais usados para
  identificar servidores MCP no GitHub, piso de estrelas, linguagens-alvo e N
  alvo. Revisar antes de cada rodada completa (ver
  `scripts/verify_sdk_syntax.py`, que checa se a sintaxe conhecida dos SDKs
  oficiais ainda bate com o que está documentado aqui — os SDKs do MCP
  estavam em reescrita ativa em 2026-08).
- `config/judges.yaml` — registro dos juízes da Etapa 3 (id, provedor, model
  id, habilitado ou não). Editar para trocar de modelo ou adicionar/remover
  um provedor sem tocar em código.
