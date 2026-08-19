# Pipeline de Mineração e Extração MCP (TCC)

Implementa as **Etapas 1 e 2** da metodologia do TCC: coleta de repositórios
de servidores MCP no GitHub (via GraphQL) e extração das ferramentas (tools)
e seus call graphs de 3 níveis a partir do código-fonte.

Ver o plano completo de implementação em
`~/.claude/plans/section-materiais-e-m-todos-label-sec-m-ticklish-panda.md`
para o desenho detalhado, decisões e riscos conhecidos.

## Status

- ✅ **Etapa 1** (mineração via GraphQL): implementada, testada e validada
  contra a API real do GitHub (206 repositórios selecionados).
- ✅ **Clonagem de repositórios**: implementada, testada e validada.
- ✅ **Etapa 2** (detecção de tools + call graph): implementada para o padrão
  de alto nível das 5 linguagens-alvo (Python, TypeScript, JavaScript, Java,
  C#), com orquestração completa (`pipeline/run_step2.py`,
  `schema/assemble_dataset.py`) e validada ponta a ponta contra 20
  repositórios reais. 100 testes automatizados.
- ⬜ **Pendente**: padrões de baixo nível/alternativos por linguagem
  (`@server.list_tools()` em Python, `setRequestHandler` em TS/JS, builder
  oficial em Java, pacote comunitário `fastmcp` em TS/JS) — ver o plano para
  os achados reais de cobertura por padrão.

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # se `uv` ainda não estiver instalado
cp .env.example .env                               # preencher com um GitHub PAT (somente leitura, repos públicos)
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

## Rodando a Etapa 2

Pré-requisito: repositórios já clonados via `clone_manager.clone_all` (ver
`data/repos/<owner>__<name>/repo_meta.json`).

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

## Testes

```bash
uv run pytest        # suíte de testes (não requer token do GitHub)
uv run ruff check .  # lint
```

## Configuração

`config/mcp_signals.yaml` — tópicos e sinais textuais usados para identificar
servidores MCP no GitHub, piso de estrelas, linguagens-alvo e N alvo. Revisar
antes de cada rodada completa (ver `scripts/verify_sdk_syntax.py`, que checa
se a sintaxe conhecida dos SDKs oficiais ainda bate com o que está documentado
aqui — os SDKs do MCP estavam em reescrita ativa em 2026-08).
