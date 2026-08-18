# Pipeline de Mineração e Extração MCP (TCC)

Implementa as **Etapas 1 e 2** da metodologia do TCC: coleta de repositórios
de servidores MCP no GitHub (via GraphQL) e extração das ferramentas (tools)
e seus call graphs de 3 níveis a partir do código-fonte.

Ver o plano completo de implementação em
`~/.claude/plans/section-materiais-e-m-todos-label-sec-m-ticklish-panda.md`
para o desenho detalhado, decisões e riscos conhecidos.

## Status

- ✅ **Etapa 1** (mineração via GraphQL): implementada e testada (29 testes,
  todos com HTTP mockado — ainda não rodada contra a API real do GitHub).
- ✅ **Clonagem de repositórios**: implementada e testada.
- ⬜ **Etapa 2** (detecção de tools + call graph por linguagem): ainda não
  implementada — próximo passo, é um trabalho substancialmente maior
  (5 linguagens × 2 gerações de SDK cada).

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
