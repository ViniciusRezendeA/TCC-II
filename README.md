# Pipeline de Mineração e Extração MCP (TCC)

Implementa as **Etapas 1, 2 e 3** da metodologia do TCC: coleta de
repositórios de servidores MCP no GitHub (via GraphQL), extração das
ferramentas (tools) e seus call graphs de 3 níveis a partir do código-fonte,
e classificação da qualidade das descrições via um júri multi-provedor de
LLMs (Claude, OpenAI, Google Gemini).

Ver o plano completo de implementação em
`~/.claude/plans/section-materiais-e-m-todos-label-sec-m-ticklish-panda.md`
para o desenho detalhado, decisões e riscos conhecidos.

## Status

- ✅ **Etapa 1** (mineração via GraphQL): implementada, testada e validada
  contra a API real do GitHub (206 repositórios selecionados).
- ✅ **Clonagem de repositórios**: implementada, testada e validada contra os
  206 repos (0 falhas).
- ✅ **Etapa 2** (detecção de tools + call graph): implementada e validada
  ponta a ponta contra os 206 repositórios reais. Cobre, por linguagem, o
  padrão de alto nível de cada SDK oficial (Python, TypeScript, JavaScript,
  Java, C#) **e** dois padrões adicionais genuinamente compartilhados por
  múltiplos projetos: o padrão de baixo nível oficial (Python
  `@server.list_tools()`, TS/JS `setRequestHandler`) e o pacote de terceiros
  `fastmcp` do npm. **3.535 tools extraídas de 71 dos 206 repositórios**
  (`data/dataset.jsonl`). 162 testes automatizados.
  - **Fora de escopo, por decisão documentada**: arquiteturas de registro
    100% customizadas usadas por um único repositório cada (ex: anotação
    própria do `ghidra-mcp`, os 4 esquemas diferentes de C# encontrados),
    builder oficial do Java (nenhum repositório real da amostra o usa), e
    escanear mais de uma linguagem por repositório — ver o plano para a
    investigação completa e a justificativa.
- ✅ **Etapa 3** (classificação via LLM-as-a-Judge): implementada e testada
  — júri multi-provedor em tier econômico (Claude Haiku 4.5, OpenAI
  gpt-4.1-mini, Google Gemini 2.5 Flash-Lite), rubrica de 6 componentes,
  execução resumível e concorrente por juiz. **Ainda não executada contra a
  API real** — requer preencher as chaves de API em `.env` (ver
  `.env.example`).

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

Pré-requisito: `data/dataset.jsonl` já montado (seção anterior), e as
variáveis `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`GOOGLE_API_KEY` preenchidas
em `.env` (só é necessário preencher a chave dos juízes habilitados em
`config/judges.yaml`).

```bash
# Roda todos os juízes habilitados contra todas as tools, nos 2 cenários
# (com e sem SOURCE_CODE). Resumível por (tool, cenário, juiz) — uma falha
# técnica individual não aborta o lote; recusas de segurança do modelo são
# gravadas como resultado próprio (status "refused"), não como erro.
uv run python -m mcp_pipeline.pipeline.run_step3 [--limit N] [--judges id1,id2] [--scenarios description_only,with_source] [--retry-failed]
```

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
