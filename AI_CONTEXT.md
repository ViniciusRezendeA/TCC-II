# Contexto Técnico Completo (para IA)

> Este documento existe para dar a uma IA (assistente de código, agente autônomo, revisor)
> o máximo de contexto possível sobre o que este repositório faz, como as peças se conectam
> e por que decisões não óbvias foram tomadas, sem precisar reler todo o código-fonte do
> zero. Não é material de leitura para humanos (para isso, ver README.md e os demais
> `*.md` na raiz) e não substitui a leitura do código quando uma mudança real for feita:
> trate tudo aqui como um mapa, não como a fonte da verdade. Onde este documento e o código
> divergirem, o código está certo — o código muda com mais frequência do que este arquivo é
> atualizado.

## 1. O que este repositório é

Este é o código de um TCC (Trabalho de Conclusão de Curso, PUC Minas) intitulado **"Model
Context Protocol (MCP): Avaliação da Qualidade de Descrições de Tools com Base em Contexto
de Código Fonte"** (autores: Pedro H. P. Rodrigues, Vinicius R. A. de A. Moreira). O texto do
TCC em si vive em `overleaf/` (LaTeX); este repositório é a implementação do experimento que
o TCC descreve.

**Metodologia base**: replica e estende Hasan et al., *"Model Context Protocol (MCP) Tool
Descriptions Are Smelly! Towards Improving AI Agent Efficiency with Augmented MCP Tool
Descriptions"* — um estudo que avalia a qualidade de descrições de *tools* MCP via
LLM-as-a-Judge com uma rubrica de 6 componentes. Este projeto adiciona como extensão
principal: dar ao juiz (LLM avaliador) acesso ao **código-fonte** da tool, não só à sua
descrição em linguagem natural, e comparar os dois cenários.

**As 3 perguntas de pesquisa** (ver `charts/README.md`, a formulação mais direta encontrada
no repo):

- **RQ1**: fornecer o código da tool ao avaliador aumenta a precisão da avaliação da
  qualidade da descrição, comparado a avaliar só a descrição?
- **RQ2**: quais componentes da rubrica são mais afetados pela adição do código?
- **RQ3**: as diferenças observadas entre os cenários (com/sem código) são estatisticamente
  significativas?

Essas 3 perguntas são o que toda a Etapa 3 (avaliação) e a análise estatística em
`scripts/analysis_evaluation_report.py` existem para responder. Praticamente toda decisão de
design a partir da Etapa 3 remonta a uma dessas 3 perguntas.

## 2. Arquitetura em 3 etapas + análise

```
Etapa 1 (coleta)  →  Clonagem  →  Etapa 2 (extração)  →  Etapa 3 (avaliação)  →  Análise/Dashboard
   GitHub GraphQL      git clone      tree-sitter          LLM-as-a-Judge          pandas + HTML
   + REST code search  --depth 1      (10 linguagens)      (rubrica 6 componentes)
```

Cada etapa lê a saída em disco da etapa anterior (arquivos JSONL) e nunca importa código
Python da etapa seguinte — só "para frente". Isso permite rodar qualquer etapa
isoladamente, reprocessar sem repetir trabalho já pago (chamadas de API, clones), e trocar
de máquina no meio do pipeline (os dados são o único acoplamento).

Todo o pacote Python vive em `src/mcp_pipeline/` (instalado via `uv sync`, ver
`pyproject.toml`; nome do pacote: `mcp-pipeline`, Python >=3.12). Scripts de análise/relatório
(não fazem parte do "pipeline" propriamente, são consumidores read-only da sua saída) vivem
em `scripts/` na raiz e são rodados via `python -m scripts.<nome>`.

### Princípios de arquitetura recorrentes (valem para o repo inteiro)

1. **Resumibilidade em toda etapa cara**: qualquer chamada de API (GitHub GraphQL, REST code
   search, judges de LLM) é checkpointada em `state/*.json` antes/depois de cada unidade de
   trabalho, e toda saída é JSONL (append-friendly, uma linha = uma unidade). Interromper com
   Ctrl+C e rodar de novo nunca perde progresso nem repete trabalho já feito (salvo bugs, ver
   §9 "Incidentes já corrigidos").
2. **Um erro nunca aborta o lote**: processar 206 repositórios ou avaliar 24 mil tools
   significa que *algo* vai falhar no meio. Todo loop principal (`clone_all`, `run_step2.main`,
   `run_judge`) captura exceção por unidade, loga em `logs/*_errors.jsonl`, e continua.
3. **"Não fabricar dado"**: em toda a Etapa 2, quando uma tool/valor/chamada não pode ser
   resolvido com confiança (nome não é literal, ambíguo, dinâmico), o código escolhe pular
   ou marcar `ambiguous: true`/`resolved: false` em vez de adivinhar um valor plausível. Essa
   frase aparece literalmente em dezenas de docstrings de `extraction/patterns/*.py`.
4. **Heurística por nome, não resolução de tipos**: a resolução de chamadas (call graph) e de
   valores (constantes/listas) é toda baseada em nome + heurísticas de proximidade de
   diretório, nunca em inferência de tipo real. Isso é uma decisão deliberada e documentada
   (`call_graph_builder.resolve_call`), não uma limitação escondida.
5. **Config como dado, não código**: quais sinais buscar no GitHub (`config/mcp_signals.yaml`)
   e quais juízes de LLM usar (`config/judges.yaml`) são YAML, não Python — trocar de modelo
   ou adicionar um provedor não deveria (e majoritariamente não precisa) tocar em código.
6. **Nada de números hardcoded em relatório**: os scripts de análise recalculam tudo a partir
   dos JSONL brutos a cada execução, inclusive o dashboard HTML (pode ser gerado a qualquer
   momento, até com a Etapa 3 ainda rodando, para ver um snapshot parcial).

## 3. Estrutura de diretórios

```
src/mcp_pipeline/
  config.py              # paths, .env, Signals (mcp_signals.yaml)
  logging_setup.py        # logger padrão (stdout + logs/<name>.log)
  collection/              # Etapa 1: busca no GitHub
    run_step1.py           #   entrypoint (-m mcp_pipeline.collection.run_step1)
    checkpoint.py           #   Checkpoint: JSON em disco, merge-then-write, atômico
    dedupe_rank.py          #   dedupe por id + filtro fork/estrelas/linguagem + ranking
  github/                  # clientes GraphQL/REST + as 3 estratégias de busca
    graphql_client.py, queries.py, models.py (RepoCandidate)
    search_topics.py, search_text.py, search_manifest.py, rest_code_search.py
    search_runner.py        #   loop de paginação genérico (usado por topics/text)
  clone/
    clone_manager.py        # shallow clone --depth 1, RepoMeta, resumível
  extraction/               # Etapa 2: tree-sitter, 10 linguagens
    language_registry.py    #   LanguageSpec por linguagem (gramática tree-sitter, extensões)
    parser_utils.py          #   parse, exclusão de dirs/arquivos de teste, helpers de string literal
    definition_index.py     #   FunctionDef + índice por qualified_name/bare_name
    import_index.py          #   ImportedName + ImportIndex (heurística, não símbolo real)
    value_index.py           #   ValueDef + ValueIndex (para padrões "lowlevel")
    call_graph_builder.py    #   resolve_call (5 passos), build_call_graph (3 níveis), collect_reachable_definitions
    complexity.py             #   McCabe cyclomatic complexity por linguagem
    models.py                 #   ToolRecord, CallGraphNode, SourceLocation (schema serializável)
    tool_detector.py          #   LANGUAGE_ADAPTERS (registry), orquestra tudo por repo
    patterns/*.py              #   1 arquivo por linguagem, detecta o "SDK pattern" de registro de tool
  pipeline/
    run_clone.py, run_step2.py, run_step3.py, run_backfill.py
  schema/
    assemble_dataset.py       # concatena tools.jsonl de cada repo -> data/dataset.jsonl
    export_for_evaluation.py  # payload no formato de Hasan et al. ({name, server_name, description, SOURCE_CODE})
    render_source_view.py      # achata o call_graph em texto anotado (vira SOURCE_CODE)
  evaluation/                # Etapa 3
    prompts.py                 # rubrica de 6 componentes, PROMPT_VERSION (changelog no topo)
    payload.py                  # build_payload: aplica cap de tamanho ao SOURCE_CODE
    judges/
      base.py                    # RubricScores (pydantic), hierarquia de exceções, RateLimiter
      registry.py                 # PROVIDER_CLASSES, load_judges()/build_judge() (lê judges.yaml)
      gemini_judge.py              # Google Gemini via google-genai SDK
      openai_compatible_judge.py   # base HTTP para servidores OpenAI-compatible (llama.cpp)
      qwen_judge.py, llama_judge.py  # subclasses (só resolvem env vars, endpoint local)
      deepseek_judge.py             # idem, cloud pago

scripts/                     # NÃO faz parte do pacote instalável; só lê JSONL, roda via -m
  analysis_report.py           # métricas/gráficos das Etapas 1-2
  analysis_evaluation_report.py  # métricas/gráficos/estatística da Etapa 3 (RQ1-RQ3)
  generate_dashboard.py          # dashboard HTML autocontido (SPA), publicado no GitHub Pages
  generate_narrative_analysis.py # narrativa em prosa via Gemini, sobre os mesmos números do dashboard
  dedupe_evaluations.py, fix_duplicate_evaluations.py, purge_prompt_version.py  # manutenção de dados
  cleanup_disqualified_repos.py  # libera disco de repos abaixo de min_tools
  run_parallel_step3.py, run_sequential_step3.py  # rodízio de N chaves de API do Gemini
  check_local_llm_servers.py, check_gemini_free_tier.py, test_local_judges.py, validate_gemini_keys.py  # smoke tests operacionais
  verify_sdk_syntax.py         # valida se os SDKs oficiais do MCP ainda batem com os padrões hardcoded
  analyze_judges_results.py    # versão MAIS ANTIGA/legada da análise — ver §8.6

config/
  mcp_signals.yaml       # sinais de busca da Etapa 1 (topics, text_signals, manifest_signals, target_languages, min_stars, top_n, min_tools)
  judges.yaml             # registro de juízes da Etapa 3 (id, provider, model_id, enabled, kwargs extras)

data/                    # todo gerado, não versionado no sentido de "editar manualmente"
  candidate_pool.jsonl, selected_repos.jsonl   # saída da Etapa 1
  repos/<owner>__<repo>/  # clones + repo_meta.json + tools.jsonl (saída da Etapa 2)
  dataset.jsonl            # saída consolidada da Etapa 2 (1 linha = 1 tool)
  evaluations/<judge_id>.jsonl  # saída da Etapa 3 (1 linha = 1 tool x cenário x juiz)
  analysis/                # saída dos scripts de análise (charts/, tables/, dashboard.html, narrative_analysis.json)
  raw/                     # cache bruto de páginas de busca (REST/GraphQL), para não regastar rate limit

state/                    # checkpoints JSON (step1_progress.json, step3_progress.json, ...)
logs/                     # logs por etapa + *_errors.jsonl

tests/                    # pytest, 1 arquivo por módulo principal + tests/fixtures/<linguagem>/
overleaf/                 # o texto do TCC em si (LaTeX) — não é código do pipeline
charts/                   # scripts TypeScript soltos, exploração ad-hoc anterior ao pipeline atual em Python (ver §8.7)
anotacoes/                # notas de leitura sobre o paper de Hasan et al. e trabalhos relacionados
```

## 4. Etapa 1 — Mineração de repositórios (`collection/`, `github/`)

**Objetivo**: encontrar candidatos reais a "servidor MCP" no GitHub e selecionar os `top_n`
(206 por padrão) por estrelas, dentre um conjunto de `target_languages`.

**Fontes de sinal** (config `config/mcp_signals.yaml`, cada uma vira 1+ sub-queries,
resultados unidos por `id` do repositório em `dedupe_rank.dedupe`):

1. **Topics** (`search_topics.py`): busca GraphQL por `topic:mcp-server`,
   `topic:model-context-protocol`. O topic genérico `mcp` foi removido deliberadamente
   (dominava o pool com plataformas grandes que só mencionam MCP de passagem).
2. **Text signals** (`search_text.py`): busca GraphQL por strings literais
   (`"@modelcontextprotocol/sdk"`, `"FastMCP"`, etc.) em `readme,description`.
3. **Manifest signals** (`search_manifest.py` + `rest_code_search.py`): busca REST Code
   Search (`/search/code`, 10 req/min, não tem equivalente GraphQL) por um sinal dentro de um
   arquivo de manifesto específico (ex: `"@modelcontextprotocol/sdk" filename:package.json`).
   Cada hit é **rehidratado** via GraphQL (`REPO_BY_NAME_QUERY`) para ganhar
   estrelas/linguagem/fork (a resposta do REST não tem esses campos). Único caminho que
   captura repos que dependem do SDK mas não mencionam isso no README.

`GraphQLClient` (`graphql_client.py`) faz backoff exponencial em 502/503/504, respeita
`retry-after` em rate limit secundário, e pausa proativamente quando o budget de rate limit
fica baixo (lê `rateLimit{remaining,resetAt}` embutido em toda query). `rest_code_search.py`
respeita os 10 req/min do Code Search com sleep fixo entre páginas.

**Resumibilidade**: `Checkpoint` (JSON, merge-then-write atômico — ver §9.1 sobre um bug real
de race condition já corrigido aqui) guarda cursor de paginação por sub-query
(`search::<label>` / `code_search::<label>`) mais um flag `done`. Toda página bruta também é
salva em `data/raw/search_pages/*.json` — se uma sub-query já está `done`, uma nova rodada
**replaya** as páginas do disco em vez de re-buscar (uma sub-query "concluída" ainda precisa
contribuir seus repos toda vez que o pipeline roda de novo, não só na primeira). Se o
`query_string` mudou desde o checkpoint salvo (ex: `min_stars` foi editado no YAML), o
checkpoint é tratado como stale e a sub-query roda do zero.

**Seleção final** (`dedupe_rank.py`):
- `dedupe()`: une por `id` GraphQL (estável mesmo com rename), acumula `matched_signals`.
- `filter_and_rank()`: filtra fork, `stargazer_count >= min_stars`,
  `primary_language in target_languages`; ordena por estrelas desc. Retorna o pool
  completo, não cortado ainda.
- `dedupe_and_rank()`: aplica os dois acima, corta em `top_n`.

**Flags do CLI** (`run_step1.py`): `--smoke-test-only` (só valida token/rate limit),
`--use-cache` (pula toda busca, reusa `data/raw/all_candidates.jsonl` da última rodada real —
útil para iterar em `min_stars`/`top_n` sem gastar rate limit), `--offline` (monta pool só a
partir de páginas REST já em cache, zero rede, sem estrelas/linguagem reais — preview
rápido, não dado real).

**Saídas**: `data/candidate_pool.jsonl` (todo o pool filtrado, para auditoria) e
`data/selected_repos.jsonl` (top N, o que de fato será clonado).

## 5. Clonagem (`clone/clone_manager.py`, `pipeline/run_clone.py`)

Shallow clone (`git clone --depth 1 --single-branch --no-tags`, anônimo via HTTPS) de cada
repo em `selected_repos.jsonl` para `data/repos/<owner>__<repo>/src/`. Grava
`repo_meta.json` (repo + `commit_sha` + `cloned_at`) ao lado — é isso que
`is_already_cloned()`/`clone_all()` checam para pular repos já feitos (resumível). Falha
individual não aborta o lote; erro vai para `logs/clone_errors.jsonl` e o diretório parcial é
removido (git pode deixar working tree quebrado se o clone falhar no meio, ex: disco cheio).

`--depth 1` é proposital, não só otimização: a Etapa 2 é análise estática da árvore de
trabalho, não do histórico.

## 6. Etapa 2 — Extração de tools + call graph (`extraction/`)

Este é o módulo mais complexo do repositório. Objetivo: para cada repo clonado, (a) achar
toda definição de "tool" MCP no código-fonte (nome + descrição + local), e (b) para cada
tool, construir um **call graph de 3 níveis** (a própria implementação + até 2 níveis de
funções chamadas), usado depois para calcular métricas (LOC, complexidade ciclomática) e
para montar o `SOURCE_CODE` que a Etapa 3 manda ao juiz de LLM.

### 6.1. Linguagens suportadas (`language_registry.py`)

10 linguagens, cada uma com sua gramática `tree-sitter` pinada em `pyproject.toml`: Python,
JavaScript, TypeScript, Java, C#, Rust, Ruby, Dart, Go, Kotlin. Cada `LanguageSpec` mapeia
nome (bate com `RepoCandidate.primary_language`) → objeto `Language` do tree-sitter →
extensões de arquivo.

Fora de escopo, por decisão documentada (ver README.md e comentários em `tool_detector.py`):
Swift, Elixir, Lua, PHP, arquiteturas de registro 100% customizadas de um único repo,
builder oficial do Java, e scanning de mais de uma linguagem por repositório.

### 6.2. `parser_utils.py` — o que conta como "código de produção"

Antes de qualquer parsing, `iter_source_files()` exclui:
- **Diretórios** (`EXCLUDED_DIR_NAMES`): `test(s)`, `example(s)`, `docs`, `node_modules`,
  `vendor`, `.venv`, `dist`, `build`, etc. — encontrado empiricamente que sem isso,
  `PrefectHQ/fastmcp` (o próprio framework FastMCP) tinha 89% de suas "tools" detectadas
  dentro de `tests/conftest.py`, e `modelcontextprotocol/kotlin-sdk` tinha 100% das suas
  dentro de fixtures de conformance-test.
- **Arquivos colocalizados de teste** (`EXCLUDED_FILENAME_SUFFIXES` + convenção
  `test_*.py`/`*_test.py`): pega o caso de um arquivo de teste ao lado do código real, sem
  estar sob um diretório `test/` (ex: `FastMCP.test.ts` ao lado de `FastMCP.ts`).

Também fornece: `string_literal_value()` (extrai o conteúdo de um nó de string literal lendo
os filhos de conteúdo da própria gramática, correto para strings triplas/com prefixo — cada
linguagem tem nomes de nó diferentes, verificados manualmente), `template_string_literal_value()`
(trata template string sem interpolação como literal; com interpolação, retorna `None` para
não fabricar um valor incompleto), e `run_query()` (wrapper de `Query`/`QueryCursor`, API do
tree-sitter >=0.25).

### 6.3. Índices repo-wide

- **`DefinitionIndex`** (`definition_index.py`): todo `FunctionDef` (nome qualificado tipo
  `Classe.metodo` ou nome bare) do repo inteiro, indexado por `by_qualified_name` e
  `by_bare_name` (lista, pode colidir). `body_node` é o nó tree-sitter em memória (nunca
  serializado).
- **`ImportIndex`** (`import_index.py`): mapa `alias local -> ImportedName(module,
  original_name)`, por arquivo. Deliberadamente uma heurística de escopo, não resolução real
  de símbolos (o próprio docstring diz isso).
- **`ValueIndex`** (`value_index.py`): igual ao anterior mas para *valores* de nível de
  módulo (ex: `TOOLS = [tool_a, tool_b]` em Python, `const allTools = [...]` em TS/JS,
  `const NOME = "literal"` em Go). Usado só pelos padrões "lowlevel" que precisam resolver
  uma lista de tools construída em outro lugar do código (ver §6.5). `resolve_value()` espelha
  o cascade de `resolve_call()` (mesma-arquivo → import → repo-wide único → desistir).

### 6.4. `call_graph_builder.py` — o coração da resolução de chamadas

`resolve_call()` implementa uma heurística de **5 passos, primeiro que casar ganha**:

1. Receiver `self`/`this` + classe atual conhecida → método da mesma classe.
2. Mesmo nome bare no mesmo arquivo → helper de módulo/irmão (se houver >1 candidato no
   mesmo arquivo, usa `nearest_by_directory` e marca `ambiguous=True`).
3. Receiver casa com um alias de import conhecido → prefere definição cujo arquivo tem o
   stem igual ao módulo importado (ou ao próprio alias, para o caso de import relativo tipo
   `from . import cache_utils`).
4. Busca repo-wide só pelo nome bare (se >1 candidato, `nearest_by_directory` +
   `ambiguous=True`).
5. Não encontrado em lugar nenhum → tratado como chamada externa/dinâmica (`external=True`,
   nó-folha).

`build_call_graph()` constrói a árvore a partir daí, **limitada a `MAX_LEVEL = 3`** (nível 1
= a própria função da tool; nível 3 nunca tem suas próprias chamadas escaneadas). Essa
limitação também evita loop infinito em recursão direta/mútua sem precisar de detecção de
ciclo separada. Chamadas repetidas idênticas (mesmo `raw_text`) no mesmo corpo são
deduplicadas para uma única aresta.

`collect_reachable_definitions()` é uma função **separada e deliberadamente diferente**: para
calcular `loc`/`cyclomatic_complexity` de "a tool completa" (soma da própria implementação +
todo helper transitivamente chamado, **sem** o teto de 3 níveis), usa um fecho transitivo com
detecção de visitado (evita loop e desduplica grafos em diamante). Crucialmente, **só segue
resoluções não-ambíguas** e desativa o passo 4 (fallback repo-wide) especificamente para
chamadas com receiver — porque um fechamento *sem teto de profundidade* que segue um "só tem
uma função com esse nome no repo inteiro" errado explode: um caso real
(`Accenture/mcp-bench`, um repo que empacota dezenas de servidores MCP não relacionados sob
um único `src/`) chegou a inflar `loc=11295`/`cc=1591` para uma tool antes dessa proteção
existir. Chamadas sem receiver continuam usando o passo 4 mesmo aqui, porque é o único jeito
de resolver o padrão CommonJS `const { getCached } = require(...); getCached()` (import não
rastreado, ver `extract_imports` de TS/JS).

`nearest_by_directory()` é genérica (usa `Protocol _HasFile`), reusada por
`value_index.resolve_value()` para o mesmo tipo de desempate.

### 6.5. `tool_detector.py` — orquestração por repositório

`LANGUAGE_ADAPTERS: dict[str, LanguageAdapter]` é o registro central (mesmo idioma de design
de `evaluation/judges/registry.py::PROVIDER_CLASSES`): cada linguagem contribui
`extract_definitions`, `extract_imports_fn`, `extract_calls`, `detect_tools` e,
opcionalmente, `extract_values`/`detect_tools_with_context` (só para os padrões "lowlevel").

`detect_tools_with_call_graphs(repo_src_root, language)` é o pipeline completo por repo:

1. `_effective_language()`: correção pontual para repos cujo `primary_language` do GitHub
   diz "JavaScript" mas o código real é majoritariamente TypeScript (dist/ compilado
   confunde a detecção de linguagem do GitHub) — casos reais confirmados:
   `firecrawl/firecrawl-mcp-server`, `grab/cursor-talk-to-figma-mcp`, etc.
2. `build_definition_index()`: 1 passada pelo repo inteiro.
3. **Fase 1**: 1 passada por arquivo, extrai imports + source bytes + roda todo detector
   "single-file" (não precisa de contexto repo-wide) — os padrões de alto nível
   (`.tool()`, `@mcp.tool()`, `addTool`, etc.).
4. **Fase 2** (só se a linguagem tem `detect_tools_with_context`): roda depois que TODOS os
   arquivos já foram lidos na Fase 1, porque uma referência de valor pode apontar para um
   arquivo que a Fase 1 ainda não alcançou na ordem do `rglob` (caso real:
   `blazickjp/arxiv-mcp-server`).
5. Para cada tool encontrada: busca sua definição no índice, constrói o call graph (3
   níveis), calcula `loc`/`cyclomatic_complexity` via `collect_reachable_definitions` (não
   limitado a 3 níveis, ver acima), e `call_graph_depth` (profundidade real atingida, ≤3).

### 6.6. Padrões por linguagem (`extraction/patterns/*.py`) — o que cada `sdk_pattern` significa

Cada linguagem detecta 1+ "padrões de SDK" (como uma tool é registrada no código-fonte). O
campo `sdk_pattern` no dataset identifica qual:

| Linguagem | `sdk_pattern` | Descrição |
|---|---|---|
| Python | `python.fastmcp_decorator` | `@mcp.tool()` (alto nível, dominante) |
| Python | `python.list_tools_lowlevel` | `@server.list_tools()` (baixo nível, oficial) |
| TS | `typescript.tool_v1` / `typescript.registerTool` | `.tool(name, desc, schema, handler)` / `.registerTool(name, {...}, handler)` |
| TS | `typescript.fastmcp_npm_addtool` | pacote `fastmcp` (npm, terceiro) `.addTool({...})` |
| TS | `typescript.set_request_handler_lowlevel` | `setRequestHandler(ListToolsRequestSchema, ...)` (baixo nível oficial) |
| JS | mesmos acima com prefixo `javascript.` | implementação **compartilhada** com TS via `ecmascript_common.py` |
| Java | `java.springai_annotation` | Spring AI `@Tool(description="...")` (dominante na prática, não o builder oficial) |
| C# | `csharp.mcpservertool_attribute` | `[McpServerTool]` + `[Description("...")]` |
| Rust | `rust.rmcp_tool_attribute` | crate `rmcp`, `#[tool(...)]`/`#[tool]` num impl `#[tool_router]` |
| Ruby | `ruby.fastmcp_tool_class` | gem `fast-mcp`, classe cujo superclass termina em `...Tool` + método `call` |
| Dart | `dart.dart_mcp_registertool` | pacote `dart_mcp`, `registerTool(Tool(name:, description:, ...), handler)` |
| Go | `go.mcp_addtool` | `mcp.AddTool(server, &mcp.Tool{Name:, Description:, ...}, handler)` (única, sempre via contexto) |
| Kotlin | `kotlin.kotlinsdk_addtool` | `addTool(name = "...", description = "...", ...) { ... }` |

**TS e JS compartilham implementação** via `extraction/patterns/ecmascript_common.py`
(gramáticas verificadas idênticas nos construtos usados); `typescript_patterns.py` e
`javascript_patterns.py` são só `functools.partial` do módulo comum com o `Language` de cada
gramática.

**Padrões "lowlevel" (Python `list_tools`, TS/JS `setRequestHandler`) merecem atenção
especial**: um único handler retorna uma **lista** de N tools, então:
- Precisam de resolução de valor cross-file com "hop budget" (`MAX_VALUE_RESOLUTION_HOPS = 4`
  em `value_index.py`, calibrado pelo caso real mais profundo encontrado:
  `chrisryugj/korean-law-mcp`).
- Suportam `.filter()` (sempre seguro, passa através) e `.map()` (só seguro se o callback for
  verificado estruturalmente como "não renomeia `name`", via `_verify_safe_map_callback`) e
  ternário (união dos dois ramos, já que qual ramo executa depende de config em runtime).
- **Compartilham `source_location`** (a localização do handler, não uma por tool) — isso é
  crítico para entender `tool_uid_for()` na Etapa 3 (ver §7.4): sem correção, tools distintas
  colidiriam na mesma chave.
- Trade-off estrutural explícito: o nível 1 do call graph é o handler de metadados
  (`list_tools()`/`setRequestHandler`), não a lógica de execução real da tool (que vive num
  handler separado, tipo `call_tool()`/`CallToolRequestSchema`) — não implementado, ver
  docstrings.

Todos os detectores seguem o princípio "não fabricar": um nome de tool não-literal (ex:
vindo de uma variável de loop) faz o call site inteiro ser **pulado**, nunca vira uma tool
fake com o texto bruto da expressão como nome.

### 6.7. `complexity.py` — Complexidade Ciclomática de McCabe

McCabe clássico (`CC = pontos_de_decisão + 1`), **não** a variante estendida que conta
operadores booleanos de curto-circuito (`&&`/`||`/`and`/`or`) — decisão deliberada, não
default do McCabe original. Tipos de nó de decisão foram verificados por linguagem contra a
gramática real pinada (não assumidos por analogia — ex: o ternário chama-se
`conditional_expression` em Python/C#, mas `ternary_expression` em JS/TS/Java). Para
`switch`/`match`/`when`, distingue `case` de `default`/wildcard (um "senão" não é um ponto de
decisão próprio) via `_is_extra_decision()`, que trata os 5 casos onde a gramática não
distingue estruturalmente os dois. A recursão para de descer em fronteiras de
função/closure/lambda aninhada (`_FUNCTION_BOUNDARY_TYPES`), para não inflar a complexidade
de uma tool com decisões de uma função interna que ela só define, nunca chama.

### 6.8. Modelos serializáveis (`models.py`)

- **`SourceLocation`**: `file` (relativo a `src/`), `start_line`/`end_line` (1-indexed).
- **`ToolRecord`**: `name`, `description`, `description_is_literal`, `sdk_pattern`,
  `source_location`, `qualified_name`, `loc`, `call_graph_depth` (≤3), `cyclomatic_complexity`.
  `loc`/`cyclomatic_complexity` cobrem "a tool completa" (ver §6.4), não limitados a 3 níveis.
- **`CallGraphNode`**: árvore recursiva (`level`, `resolved`, `external`, `ambiguous`,
  `qualified_name | None`, `raw_call_text`, `source_location | None`, `calls: list[...]`).

## 7. Etapa 3 — Avaliação via LLM-as-a-Judge (`evaluation/`, `pipeline/run_step3.py`)

**Objetivo**: para cada tool no `dataset.jsonl`, em 2 cenários (`description_only`,
`with_source`) e para cada juiz habilitado, pedir a um LLM para pontuar a descrição em 6
componentes numa escala Likert 1-5.

### 7.1. A rubrica (`evaluation/prompts.py`)

6 componentes, idênticos aos de Hasan et al.: **Purpose**, **Guidelines**, **Limitations**,
**Parameter Explanation**, **Length & Completeness**, **Examples**. Escala Likert 1-5 (5 =
ideal, sem ambiguidade; 1 = ausente). `RUBRIC_SYSTEM_PROMPT` é o mesmo texto para os dois
cenários (o cenário é definido só pela presença/ausência da chave `SOURCE_CODE` no payload,
não por um prompt diferente — maximiza reuso de cache de prompt no Gemini).

**`PROMPT_VERSION`** (atualmente `"v4"`) é parte da **chave de checkpoint**
(`checkpoint_key()` em `run_step3.py`), então mudar a rubrica nunca sobrescreve resultados
antigos silenciosamente — eles ficam no JSONL etiquetados com sua própria versão, e o
dashboard tem uma aba "Versões do prompt" para comparar v-a-v. O changelog no topo do
arquivo documenta por que cada bump aconteceu:
- v2: mudou de "pode enriquecer com o código" para "deve validar congruência" (o código só
  pode confirmar/contradizer a descrição, não adicionar informação nova à nota).
- v3: a regra de v2 só cobria alguns componentes explicitamente; dados-piloto reais
  (`gemini-3.5-flash-lite`, N=388) mostraram `with_source` pontuando mais alto em **todos os
  6 componentes**, inclusive `length_completeness` — que avalia o tamanho da própria
  descrição, e a `description` é byte-idêntica entre cenários. Isso é um efeito de halo (o
  juiz reage à mera presença/volume de código), não validação genuína. v3 unificou a regra:
  o `SOURCE_CODE` só pode **rebaixar** uma nota (uma contradição/omissão encontrada), nunca
  subir, e a `reasoning` deve nomear o problema específico sempre que uma nota for rebaixada.
- v4: versão atual (ver o arquivo para o texto exato do changelog mais recente).

### 7.2. `payload.py` — o que o juiz recebe

`build_payload()` chama `schema/export_for_evaluation.py::export_tool_for_evaluation()`
(formato de Hasan et al.: `{name, server_name, description, SOURCE_CODE?}`) e aplica um cap
de tamanho: `MAX_SOURCE_CODE_CHARS = 32_000` (calibrado no piloto: mediana ~10.2k chars,
p90 ~47k, máx ~345k — cap na cauda, não validado como limite ótimo). `SOURCE_CODE` vem de
`schema/render_source_view.py`, que achata a árvore do call graph em texto, lendo cada
snippet fresco do disco (não do dataset serializado) e anotando nível/localização/chamada de
cada trecho.

### 7.3. Juízes (`evaluation/judges/`)

`RubricScores` (pydantic, `base.py`) é o schema de saída esperado: 6 campos, cada um
`ComponentScore{score: int(1-5), reasoning: str}`. **Sem campo agregado** — a média por tool
é calculada na análise, nunca emitida pelo juiz. Nota: `Likert` usa `int` com `ge=1,le=5`, não
`Literal[1,2,3,4,5]`, porque o `response_schema` do Gemini só aceita enum de STRING, não de
inteiro — um `Literal[int]` quebraria a validação pydantic antes mesmo de enviar a request.

**Hierarquia de exceções** (`base.py`), cada uma tratada de forma diferente por
`run_step3.py::run_judge`:
- `JudgeRefusal`: recusa de segurança do provedor. Resultado **esperado** (parte do dataset
  vem de servidores de pentest) — grava `status="refused"`, não é erro.
- `JudgeQuotaExhausted`: cota **diária** da conta esgotada — cancela todo o resto do lote
  daquele juiz (não adianta continuar, tudo falharia igual), não grava nada (retry automático
  na próxima rodada, sem flag).
- `JudgeBalanceExhausted`: saldo pré-pago zerado (DeepSeek) — mesmo tratamento de "cancela o
  resto", mas não reseta sozinho (precisa de um humano adicionar crédito).
- `JudgeRateLimited`: 429 que NÃO é a cota diária (ex: limite por minuto) — o fluxo normal
  (single-key) não trata especial, cai no erro genérico; só `run_sequential_step3.py` trata
  essa classe especificamente (rotaciona de chave).
- `JudgeError` genérico: falha técnica (rede, parse) — vai para
  `logs/step3_errors_{judge_id}.jsonl`, não grava JSONL/checkpoint (retry automático).

**`RateLimiter`** (compartilhado, não só Gemini): espaça chamadas uniformemente
(`60/requests_per_minute` segundos entre elas), em vez de deixar rajadas até o limite —
necessário porque `--concurrency` do `ThreadPoolExecutor` sozinho não controla *taxa*, só
quantas chamadas estão em voo ao mesmo tempo; uma rajada de `concurrency` liberada de uma vez
já estourou RPM real (429 confirmado até com `--concurrency 3` no Gemini free tier).

**Implementações**:
- `GeminiJudge`: `google-genai`, `generate_content()` com `response_schema=RubricScores`.
  Distingue cota diária de rate limit por minuto lendo o `QuotaFailure.quotaId` estruturado
  da resposta (não regex no texto do erro). Timeout de 60s explícito (uma conexão morta sem
  timeout já travou uma execução por 7+ horas antes disso existir).
- `OpenAICompatibleJudge`: base HTTP genérica (chamadas `requests` diretas, não SDK OpenAI)
  para qualquer servidor compatível com a API de chat completions da OpenAI — usada por
  `QwenJudge`/`LlamaJudge` (llama.cpp local) e `DeepSeekJudge` (cloud pago). Suporta
  `response_format="json_schema"` (grammar-constrained, usado pelos juízes locais) e
  `"json_object"` (fallback de prompt, usado pelo DeepSeek, que rejeita `json_schema` — tem
  uma extração via regex + `RubricScores.model_validate()` como rede de segurança adicional).
- `QwenJudge`/`LlamaJudge`: lazy-load de credenciais de `QWEN_LLM_BASE_URL`/
  `QWEN_LLM_BEARER_TOKEN` e `LOCAL_LLM_BASE_URL`/`LOCAL_LLM_BEARER_TOKEN` (servidores
  llama-server na rede local do autor). Timeout `None` por padrão (sem limite) — um timeout
  aqui normalmente indica que o llama-server caiu, não que está lento.
- `DeepSeekJudge`: cloud, **pago** (sem free tier confirmado), desabilitado por padrão em
  `judges.yaml` (`enabled: false`), ativado só via `--judges deepseek-flash` explícito.

`registry.py::load_judges(only=None)` lê `config/judges.yaml`: com `only=None`, instancia
toda entrada `enabled: true`; com `only={ids}`, instancia exatamente essas (ignora
`enabled`) — é isso que permite `--judges <id>` num piloto sem editar o YAML. `only_keys`
além de `{id, provider, model_id, enabled}` no YAML são passadas como kwargs extras pro
construtor (ex: `requests_per_minute`).

### 7.4. `pipeline/run_step3.py` — o runner

**`tool_uid_for(row)`**: chave semântica estável = `{repo}::{qualified_name}::{file}:{line}`,
**mais o `tool.name`** quando `sdk_pattern` é um dos padrões "lowlevel"
(`_LOWLEVEL_SHARED_LOCATION_PATTERNS`) — necessário porque essas tools compartilham
`source_location`/`qualified_name` (o handler que as registra), então sem esse sufixo,
tools distintas colidiriam na mesma chave (confirmado: 257 chaves colidindo cobriam ~3.7k
tools, 30% do dataset).

**`checkpoint_key()`** = `{tool_uid}::{scenario}::{judge_id}::{PROMPT_VERSION}`.
**`should_skip()`**: pula se status salvo é `ok`/`refused`; reprocessa `error` só com
`--retry-failed`.

**`shard_for(tool_uid, num_keys)`**: hash SHA-256 determinístico → partição `[0, num_keys)`,
sem gerar arquivo de partição — é isso que permite `run_parallel_step3.py` (N processos, um
por chave de API) filtrarem o mesmo `dataset.jsonl` cada um pra sua fatia disjunta, sem
coordenação além do hash em si.

**`run_judge()`**: monta todos os pares (tool, cenário) pendentes primeiro (I/O de leitura de
snippet é rápido, feito serial), depois roda só `judge.evaluate()` dentro de um
`ThreadPoolExecutor(max_workers=concurrency)`. `checkpoint.set()` só é chamado da thread
principal, **depois** de escrever+flushar a linha no JSONL — ordem importa: se o processo
morre entre os dois, o pior caso é 1 chamada de API redundante no reprocessamento, nunca um
resultado perdido silenciosamente. Uma falha (erro técnico ou cota esgotada) não grava
NADA (nem JSONL nem checkpoint) — reprocessamento automático na próxima rodada, sem flag,
importante para execuções desatendidas/paralelas.

**CLI**: `--limit`, `--judges id1,id2`, `--scenarios description_only,with_source`,
`--concurrency N`, `--retry-failed`, `--num-keys`/`--key-index` (sharding para paralelismo).

**Saída**: `data/evaluations/{judge_id}.jsonl`, append-only, 1 linha por
(tool, cenário, juiz). Ver §10.2 para o schema completo de uma linha.

### 7.5. Rodízio de múltiplas chaves de API (`run_parallel_step3.py`, `run_sequential_step3.py`)

Dois scripts alternativos para acelerar o juiz Gemini usando várias chaves de API
(`GOOGLE_API_KEYS` no `.env`), cada um respondendo a um comportamento real de rate limit
**descoberto empiricamente** (documentado nos próprios docstrings, não é teoria):

- **`run_parallel_step3.py`** (N processos simultâneos, um por chave, via
  `--num-keys`/`--key-index` de `run_step3.py`): a primeira tentativa. Falhou parcialmente:
  mesmo com 20 chaves confirmadas como 20 *projetos* Google distintos, a taxa de 429
  (`GenerateRequestsPerMinutePerProjectPerModel`) escalava com o volume **total** somado
  entre todas as chaves, não com o volume de cada uma isolada — indica throttling agregado
  por **conta** Google, não só por projeto. `--max-parallel` existe para mitigar isso
  (limita quantos processos rodam ao mesmo tempo).
- **`run_sequential_step3.py`** (1 processo, zero simultaneidade, troca de chave só quando a
  atual leva um 429): a correção. Confirmado ao vivo que com `--max-parallel 1` a taxa de
  sucesso vai a 100% — zero simultaneidade elimina o throttling agregado. Honra
  `retry_after_seconds` do provedor (ou um backoff padrão de 5s) antes de tentar de novo, e
  desiste de uma tarefa específica depois de `_MAX_RATE_LIMIT_RETRIES=5` tentativas seguidas
  (fica pendente pra próxima rodada) em vez de rodar pra sempre se a conta inteira estiver
  throttled por mais tempo. `JudgeQuotaExhausted` (cota diária, por chave/projeto) tira a
  chave do rodízio pelo resto da execução; `JudgeRateLimited` (por minuto, agregado na conta)
  só espera e tenta a próxima chave — mas qualquer chave tentada durante uma janela de
  throttle leva o mesmo 429, então trocar não escapa dele por si só, só evita ficar preso
  numa chave específica.

## 8. Scripts de análise e o dashboard

Estes scripts são **consumidores read-only**: nunca importam nada de
`extraction`/`collection`/`github` (exceto `analysis_report.py`, que lê `dataset.jsonl`
diretamente sem reprocessar) e nunca fazem chamada de API de LLM (exceto
`generate_narrative_analysis.py`, isolado de propósito).

### 8.1. `analysis_report.py` — métricas das Etapas 1-2

Lê só `data/selected_repos.jsonl` e `data/dataset.jsonl`. Produz `data/analysis/charts/*.png`
e `data/analysis/tables/resumo_etapas_1_2.xlsx` (+ CSVs individuais). Métricas: repos/tools
por linguagem, funil de atrito da Etapa 1 (bruto → único → filtrado → selecionado), média de
tools por servidor (2 versões: sobre todo o pool selecionado vs. só sobre quem teve ≥1 tool),
distribuição de `sdk_pattern`, taxa de resolução do call graph, distribuição de LOC/CC,
profundidade de call graph, correlação LOC×CC, top repos por número de tools. Fornece também
helpers de gráfico (`_bar_chart`, `_grouped_bar_chart`, `_histogram`, `_scatter_chart`,
`_boxplot`) e `export_tables()`, **reaproveitados** por `analysis_evaluation_report.py` (não
duplicados).

### 8.2. `analysis_evaluation_report.py` — o núcleo estatístico das RQ1-RQ3

Lê só `data/evaluations/{judge_id}.jsonl`. Este é o arquivo que implementa de fato a análise
comparativa do TCC. Funções-chave:

- **`registros_versao_ativa()`**: restringe a registros do `prompt_version` mais recente
  presente nos dados — misturar versões de prompt diferentes compararia respostas a
  instruções distintas como se fossem a mesma coisa.
- **`tool_key_for()`**: **igual em espírito a `tool_uid_for()` da Etapa 3, mas corrige o
  problema retroativamente a partir de dados já coletados** (usa `tool.name`, já gravado em
  todo registro, em vez de precisar reprocessar). Também normaliza tool_uids de "safras"
  diferentes (registros salvos antes/depois da correção em `tool_uid_for` — ver §9.2) para a
  mesma chave, fazendo `strip` do sufixo `::{name}` antes de reanexar.
- **`scores_long()`**: formato longo (1 linha = tool × cenário × juiz × componente), base
  compartilhada por quase tudo abaixo.
- **`comparacao_cenarios()`**: média por componente/juiz em cada cenário + delta pareado
  (`with_source - description_only`), só sobre tools avaliadas nos **dois** cenários pelo
  mesmo juiz.
- **`wilcoxon_por_componente()`**: **o teste de significância central do TCC (RQ3)** — Teste
  de Postos Sinalizados de Wilcoxon, pareado, por componente **e por juiz separadamente**
  (nunca misturando juízes — a metodologia exige poder checar se o efeito se mantém
  consistente entre avaliadores). Descarta pares empatados (`zero_method="wilcox"`, o
  default do scipy) e reporta a proporção de empates. Aplica correção de
  **Benjamini-Hochberg (FDR)** própria (`_benjamini_hochberg()`, implementada manualmente com
  numpy — statsmodels não é dependência do projeto só por isso), corrigindo as 6 comparações
  simultâneas (1 por componente) **dentro de cada juiz** — não entre juízes, porque juízes
  são reportados separadamente por design, não como uma família de testes conjunta.
- **`quartil_notas()`/`migracao_quartil_por_tool()`/`migracao_quartil_por_componente()`**:
  classifica cada nota no seu quartil (rank percentual dentro da distribuição de
  juiz×componente×cenário — não corte bruto Q1/Q2/Q3, porque a escala Likert 1-5 é discreta
  e concentrada demais para isso funcionar) e detecta "divergência" = migração de **pelo
  menos `MUDANCA_MINIMA_QUARTIS = 2` faixas** de quartil entre os dois cenários (calibrado
  contra volume real: qualquer mudança gerava ~5000 linhas de ruído só no DeepSeek; ≥2 faixas
  gera ~1300, mais tratável).
- **`classificar_motivos()`**: junção por palavra-chave (não classificação exclusiva) do texto
  de `reasoning` do juiz contra categorias (`parametro`, `limitacao_erro`, `exemplo`,
  `contradicao`, `omissao`, `escopo_proposito`), com fallback
  `sem_justificativa_especifica` quando nenhuma casa — usado só sobre a reasoning do cenário
  `with_source` (é onde o juiz deveria registrar a inconsistência encontrada, segundo a
  instrução ativa "Handling SOURCE_CODE" do prompt).
- **`veredito_custo_beneficio()`**: cruza significância estatística + direção do efeito
  (`_mediana_diferenca_efetiva_por_componente`, que exclui empates — mesma base que o teste
  de Wilcoxon usa, ao contrário da mediana "bruta" de `wilcoxon_por_componente` que inclui
  empates e fica arrastada pra perto de zero) + proporção de divergências **sem** motivo
  específico + custo extra em tokens/latência de mandar `SOURCE_CODE`. Produz
  `vale_a_pena ∈ {"sim", "não", "inconclusivo"}` por (juiz, componente) — **explicitamente
  não assume que "subir" ou "descer" seja a direção "certa"**: o prompt ativo só diz que a
  nota "pode ser ajustada" a partir de inconsistências, sem prescrever direção.

Todas essas funções são chamadas de novo, com os mesmos dados, por `generate_dashboard.py`
(reuso de código explícito, não duplicação).

### 8.3. `generate_dashboard.py` — dashboard HTML autocontido

Gera um único arquivo HTML (`data/analysis/dashboard.html` por padrão) sem dependência
externa além de fontes do Google Fonts — todo o JS/CSS está inline, os dados vão embutidos
como um bloco JSON (`__DASHBOARD_DATA__` substituído em `render_html()`). Recalcula tudo do
zero a cada execução; pode rodar com a Etapa 3 ainda em andamento (snapshot parcial).

5 abas: **Visão geral** (tiles de status, ranking de componentes, comparação de cenários,
Wilcoxon — com sub-abas por juiz + "Todos"), **Dataset** (métricas das Etapas 1-2, opcional:
só aparece se `selected_repos.jsonl`/`dataset.jsonl` existirem em disco), **Tools** (tabela
pesquisável/ordenável, 1 linha por tool, expandível para ver os 6 componentes por cenário),
**Divergências** (boxplot da distribuição por componente×cenário, tabela de pares que
migraram ≥2 quartis com reasoning lado a lado, resumo de motivos, o veredito
custo-benefício), **Versões do prompt** (quantas avaliações por `prompt_version`, comando
pronto para `purge_prompt_version.py` de versões obsoletas).

`build_dashboard_data()` roda `dedupe_records()` (ver §8.4) antes de tudo, e separa por
`prompt_version` ativa (a mais recente por padrão, ou `--prompt-version` explícito). Um
juiz com `N < MIN_TRUSTWORTHY_N = 30` avaliações ok é sinalizado (`flag`) em vez de mostrado
como se fosse estatisticamente comparável a um com centenas.

**Publicação automática**: `.github/workflows/dashboard.yml` roda em todo push a `main` que
toque `data/evaluations/**`, os scripts de dashboard/narrativa, ou `evaluation/prompts.py`,
publica em GitHub Pages. A chamada ao Gemini para a narrativa (`continue-on-error: true`)
nunca trava o deploy — sem o secret configurado ou numa falha transitória, o dashboard sobe
sem a seção de narrativa, tratado como estado normal.

### 8.4. `dedupe_evaluations.py` / `fix_duplicate_evaluations.py` — manutenção de dados

`run_step3.py` grava em modo append; um retry (`--retry-failed`) soma uma linha nova sem
remover a antiga com `status="error"`. `dedupe_evaluations.py` limpa isso por chave
`(tool_key_for(record), scenario, prompt_version, judge_id)` (inclui `judge_id`
explicitamente — sem isso, avaliações de juízes diferentes para a mesma tool colidiam e uma
apagava a outra), preferindo a linha `ok`/`refused` mais recente, senão a `error` mais
recente. `dedupe_records()` (a função pura, sem I/O) é reusada por
`generate_dashboard.py`/`generate_narrative_analysis.py`; `dedupe_file()` é o CLI que de fato
reescreve o arquivo (com backup `.bak` e checagem de "arquivo mudou de tamanho durante a
leitura" para não pisar em escritas concorrentes de um `run_step3.py` ainda rodando).

`fix_duplicate_evaluations.py` é um reparo **pontual, de um incidente específico** (documentado
no próprio docstring, 2026-09-18): tool_uid colidido retroativamente (antes da correção em
`tool_uid_for`) + avaliações genuinamente duplicadas por 2 processos sem checkpoint
compartilhado. Reconstrói `state/step3_progress.json` do zero a partir dos JSONL já
reparados, em vez de tentar remendar o checkpoint existente.

### 8.5. Outros utilitários de manutenção

- **`purge_prompt_version.py`**: remove permanentemente todas as avaliações de um
  `prompt_version` específico (nunca roda automaticamente — passo manual, explícito, depois
  que uma comparação v-a-v não é mais necessária).
- **`cleanup_disqualified_repos.py`**: libera disco removendo `src/` de repos já processados
  pela Etapa 2 que ficaram abaixo de `min_tools` (cobre repos processados antes dessa limpeza
  existir dentro do próprio `run_step2.process_repo`, ou depois de mudar `min_tools`).
- **`verify_sdk_syntax.py`**: shallow-clone dos 4 SDKs oficiais do MCP (Python/TS/Java/C#) e
  grep por strings-âncora que os padrões em `extraction/patterns/*.py` assumem — smoke check
  de que os SDKs não mudaram de sintaxe desde que os detectores foram escritos (o ecossistema
  MCP estava em reescrita ativa em 2026-08).
- **`check_local_llm_servers.py`/`check_gemini_free_tier.py`/`validate_gemini_keys.py`/
  `test_local_judges.py`**: health checks e pilotos operacionais, não fazem parte do fluxo
  de dados do pipeline em si.

### 8.6. `analyze_judges_results.py` — versão LEGADA

Este script é uma iteração **anterior** de `analysis_evaluation_report.py` (compare os
docstrings: ambos citam a mesma metodologia de Hasan et al., mas este não tem
`tool_key_for()`, teste de Wilcoxon, correção BH, nem o veredito custo-benefício). Ainda
presente no repo e referenciado por `HASSAN_ANALYSIS.md`, mas **`analysis_evaluation_report.py`
+ `generate_dashboard.py` é a análise atualmente ativa e mais completa** — ao editar/estender
a análise estatística, prefira aquele.

### 8.7. `charts/` — toolkit exploratório em TypeScript

Diretório separado com scripts `.ts` (`gerar_graficos.py` também existe, misto) para
exploração ad-hoc das mesmas perguntas de pesquisa (ver `charts/README.md` para a formulação
das RQ1-RQ3 mais direta do repo). Não faz parte do pipeline Python em `src/mcp_pipeline`/
`scripts/`, nem é importado por eles — tratar como um caderno de rascunho anterior/paralelo,
não como a fonte de verdade da análise.

## 9. Incidentes reais já corrigidos (importante para não reintroduzir)

Estes bugs já aconteceram em execução real e foram corrigidos — o código atual já reflete a
correção, mas entender o incidente ajuda a não reintroduzir a mesma classe de erro em código
novo:

1. **`Checkpoint._save()` e a race condition de múltiplos processos** (`collection/checkpoint.py`):
   originalmente, cada `set()` sobrescrevia o arquivo inteiro a partir do `_data` em memória
   daquela instância (carregado uma vez no `__init__`). Rodar 2 processos apontando pro mesmo
   arquivo de checkpoint (ex: Gemini + DeepSeek em paralelo) fazia quem salvasse por último
   apagar tudo que o outro processo tinha adicionado desde que carregou. Incidente real: um
   checkpoint com 9.720 chaves enquanto o JSONL de avaliação já tinha 12.902 pares únicos, e
   ~2.500 pares foram **rebilled** (avaliados de novo, pagando de novo) porque a entrada de
   checkpoint sumida fez `should_skip()` tratar trabalho já pago como pendente. **Correção**:
   `_save()` agora relê o arquivo do disco e faz merge antes de escrever — reduz a janela de
   corrupção para o caso bem mais raro de 2 processos escrevendo a MESMA chave no mesmo
   instante (perde só aquela chave, não o arquivo inteiro).
2. **Colisão de `tool_uid` nos padrões "lowlevel"** (`pipeline/run_step3.py::tool_uid_for`):
   tools que compartilham `source_location` (ver §6.6) colidiam na mesma chave de checkpoint
   e de agrupamento de análise, corrompendo o pareamento `description_only`/`with_source`
   (pivot table promediava tools diferentes juntas) e fazendo `should_skip()` marcar um
   grupo inteiro como "feito" assim que qualquer um de seus membros completasse, descartando
   os outros permanentemente numa retomada. **Correção**: `tool_uid_for` anexa `::{tool.name}`
   para esses padrões; `tool_key_for` (Etapa de análise) normaliza retroativamente dados de
   antes e depois dessa correção para a mesma chave.
3. **Cache de páginas "concluído mas vazio" em `search_runner.py`/`rest_code_search.py`**: se
   o checkpoint sobrevive a uma limpeza de `data/` mas o cache de páginas brutas não, uma
   sub-query marcada `done=True` sem nenhum arquivo de página em disco antes silenciosamente
   contribuía **zero** candidatos, sem erro nem aviso — incluindo, num caso real, o maior
   sinal de manifesto do conjunto inteiro. **Correção**: essa combinação específica agora é
   tratada como cache corrompido (relança a sub-query do zero), nunca como "zero resultados
   legítimo".
4. **Throttling agregado por conta Google, não por projeto** (ver §7.5): rodar N chaves de
   API do Gemini em paralelo, mesmo sendo N projetos Google distintos, ainda leva 429
   proporcional ao volume total somado — descoberto ao vivo, motivou a reescrita de
   `run_parallel_step3.py` (N processos) para `run_sequential_step3.py` (rodízio
   sequencial, zero simultaneidade).
5. **Efeito halo do `SOURCE_CODE` sobre a nota** (ver changelog de `PROMPT_VERSION` em
   §7.1): dados-piloto mostraram `with_source` pontuando mais alto em todos os 6
   componentes, inclusive um que não deveria poder mudar (a descrição é idêntica nos dois
   cenários) — motivou v2→v3 do prompt (SOURCE_CODE só pode rebaixar, nunca subir, uma nota).

## 10. Schemas de dados de referência

### 10.1. Uma linha de `data/dataset.jsonl` (saída da Etapa 2, consumida pela Etapa 3 e pela análise)

```json
{
  "schema_version": "1.0",
  "repo": { "...RepoCandidate.to_dict() + commit_sha + cloned_at..." },
  "tool": { "...ToolRecord.to_dict()..." },
  "call_graph": { "...CallGraphNode.to_dict() (árvore, 3 níveis)..." },
  "extraction_meta": { "extracted_at": "...", "extractor_version": "0.1.0" }
}
```

### 10.2. Uma linha de `data/evaluations/{judge_id}.jsonl` (saída da Etapa 3)

```json
{
  "schema_version": "1.0",
  "prompt_version": "v4",
  "tool_uid": "org/repo::qualified_name::file:line[::tool_name se lowlevel]",
  "repo": { "name_with_owner", "stargazer_count", "primary_language" },
  "tool": { "name", "qualified_name", "description" },
  "scenario": "description_only | with_source",
  "judge": { "id", "provider", "model_id", "raw_model_version" },
  "status": "ok | refused | error",
  "scores": {
    "purpose": { "score": 1-5, "reasoning": "..." },
    "guidelines": { ... }, "limitations": { ... }, "parameter_explanation": { ... },
    "length_completeness": { ... }, "examples": { ... }
  },
  "error_detail": "categoria da recusa, se status=refused",
  "source_code_sha256": "hash do SOURCE_CODE enviado, ou null (description_only)",
  "usage": { "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens" },
  "latency_ms": 1520.4,
  "evaluated_at": "ISO 8601 UTC"
}
```

## 11. Testes e configuração de desenvolvimento

- **`uv sync`** instala tudo (`pyproject.toml`, Python >=3.12, gerenciado por `uv`).
- **`uv run pytest`**: suíte em `tests/`, não depende de token de GitHub nem chave de LLM
  real (testa contra fixtures em `tests/fixtures/<linguagem>/` e mocks). 1 arquivo de teste
  por módulo principal: cada `extraction/patterns/<lang>_patterns.py` tem
  `tests/test_<lang>_patterns.py` correspondente, mais testes de call graph, complexidade,
  checkpoint, clone, dedupe, GraphQL client, registry de juízes, payload, prompts, os 3
  pipelines (`run_step2`, `run_step3`, `run_backfill`), e a análise
  (`test_analysis_evaluation_report.py`, cobre `tool_key_for`, Benjamini-Hochberg, Wilcoxon,
  classificação de motivos, veredito custo-benefício).
- **`uv run ruff check .`**: lint. `pytest.ini_options.testpaths = ["tests"]` existe
  especificamente para o discovery do pytest não tentar importar `test_*.py` de dentro de
  `data/repos/` (repos MCP reais clonados têm seus próprios testes, que colidiam com a
  coleta do pytest deste projeto).
- **`.env`** (não versionado, ver `.env.example`): `GITHUB_TOKEN` (Etapa 1),
  `GOOGLE_API_KEY`/`GOOGLE_API_KEYS`/`GOOGLE_API_KEY_NARRATIVE` (Gemini: juiz normal, rodízio
  paralelo, e narrativa — 3 propósitos deliberadamente separados para não competirem por
  cota), `DEEPSEEK_API_KEY`, `LOCAL_LLM_BASE_URL`/`LOCAL_LLM_BEARER_TOKEN`,
  `QWEN_LLM_BASE_URL`/`QWEN_LLM_BEARER_TOKEN`.

## 12. Perguntas que uma IA provavelmente vai ter (respostas rápidas)

- **"Onde entra uma nova linguagem na Etapa 2?"** → `language_registry.py` (LanguageSpec),
  novo `extraction/patterns/<lang>_patterns.py` (definitions/imports/calls/detect_tools, no
  mínimo), registrar em `LANGUAGE_ADAPTERS` (`tool_detector.py`), `_SIMPLE_DECISION_TYPES`/
  `_FUNCTION_BOUNDARY_TYPES` em `complexity.py`.
- **"Onde entra um novo provedor de juiz LLM?"** → nova classe em
  `evaluation/judges/<provider>_judge.py` (implementa `Judge` protocol: `judge_id`,
  `provider`, `model_id`, `evaluate(payload) -> JudgeEvaluation`), registrar em
  `PROVIDER_CLASSES` (`judges/registry.py`), adicionar entrada em `config/judges.yaml`. Se
  usa API compatível com OpenAI, provavelmente basta subclassear `OpenAICompatibleJudge`
  (ver `qwen_judge.py`/`deepseek_judge.py` como exemplo mínimo).
- **"Como mudar a rubrica de avaliação?"** → editar `RUBRIC_COMPONENTS`/`RUBRIC_SYSTEM_PROMPT`
  em `evaluation/prompts.py` **e** dar bump em `PROMPT_VERSION` (senão resultados antigos e
  novos ficam misturados na análise sob a mesma versão).
- **"Por que uma tool não aparece no dataset?"** → ou o repo caiu abaixo de `min_tools`
  (`config/mcp_signals.yaml`, aplicado em `assemble_dataset.py`), ou a linguagem não está em
  `LANGUAGE_ADAPTERS`, ou o padrão de registro daquele repo não é nenhum dos listados em
  §6.6, ou o nome/descrição da tool não era um literal estático (o extrator pula em vez de
  fabricar).
- **"Por que uma avaliação não aparece no dashboard?"** → checar `prompt_version` (o
  dashboard só mostra a versão ativa por padrão, use `--prompt-version` para ver outra), ou
  `status != "ok"` (refused/error não entram em `scores_long()`), ou duplicata removida por
  `dedupe_records()`.
- **"Isso é uma pipeline de produção ou um experimento acadêmico?"** → experimento acadêmico
  de único uso (dataset fixo de ~206 repos / ~12-24k avaliações), não um serviço contínuo —
  não adicionar abstração/generalização além do que os dados reais do TCC precisam.
