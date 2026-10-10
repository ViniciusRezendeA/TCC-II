# Pipeline de consenso entre juízes e análise qualitativa

Scripts que leem `data/evaluations/{judge_id}.jsonl` (saída da Etapa 3) para identificar,
entre as tools avaliadas, aquelas em que todos os juízes concordam que a nota mudou com a
inclusão do código -- e, a partir disso, caracterizar qualitativamente *o que* o código tende
a revelar. Complementa a análise estatística de `analysis_evaluation_report.py`
(Wilcoxon, por componente da rubrica); aqui o objetivo não é "a nota mudou?", mas "por que
mudou, e que tipo de mudança é essa?".

Rodar nesta ordem:

```bash
uv run python -m scripts.export_consenso_sample          # 1. filtra + anexa código-fonte
uv run python -m scripts.generate_consenso_justifications # 2. 1 justificativa (LLM) por tool
uv run python -m scripts.generate_consenso_categories      # 3. categorias (análise temática indutiva)
uv run python -m scripts.generate_consenso_categorization  # 4. classifica cada tool nas categorias
```

| Script | Entrada | Saída | O que faz |
|---|---|---|---|
| `export_consenso_sample.py` | `data/evaluations/*.jsonl` + `data/dataset.jsonl` | `data/analysis/consenso_amostra.jsonl` | Filtra tools em que **todos** os juízes que a avaliaram mudaram a nota no mesmo componente entre `description_only`/`with_source` (mínimo 2 juízes); anexa o `source_code` real enviado ao juiz. |
| `generate_consenso_justifications.py` | `consenso_amostra.jsonl` | `consenso_justificativas.jsonl` | Por tool, pede a um LLM (Gemini) que sintetize, em inglês, o que o código revelou que a descrição não dizia -- grounded na reasoning dos juízes + código, instruído a não inventar conexão sem evidência. |
| `generate_consenso_categories.py` | `consenso_justificativas.jsonl` (tudo de uma vez) | `consenso_categorias.json` | Análise temática indutiva (Braun & Clarke, 2006): identifica, de baixo para cima, as categorias recorrentes de mecanismo (não de componente da rubrica). |
| `generate_consenso_categorization.py` | `consenso_justificativas.jsonl` + `consenso_categorias.json` | `consenso_categorizacao.jsonl` | Análise de conteúdo dirigida (Hsieh & Shannon, 2005): classifica cada tool contra o conjunto fixo de categorias, uma tool por vez (nunca em lote, para evitar degradação de atenção em entradas/saídas muito longas -- Liu et al., 2024, "Lost in the Middle"). Multi-label, com fallback `"none"`. |

Os três *prompts* usados (geração de justificativa, identificação de categorias,
classificação) estão reproduzidos na íntegra no Apêndice da metodologia do TCC
(`overleaf/sectionsTCCII/Apendice.tex`, seção "Prompts Utilizados na Análise Qualitativa das
Divergências").

Todos usam `GOOGLE_API_KEYS` (rodízio de múltiplas chaves, mesma variável dos juízes reais da
Etapa 3 -- rodam em momentos diferentes, não competem por cota na prática), exceto
`generate_consenso_categories.py`, que por ser uma chamada única (não um lote de milhares)
roda no modelo mais forte disponível no free tier (`gemini-3.6-flash`).

## Limitação conhecida e melhoria futura: concordância entre avaliadores (inter-rater reliability)

As três etapas de análise qualitativa (justificativa, categorias, classificação) são
conduzidas por um único "codificador" automatizado (sempre o mesmo modelo, sem um segundo
avaliador independente rodando em paralelo). Isso significa que não há como quantificar hoje
o quão **reproduzível** é o resultado -- se um segundo avaliador (humano ou outro modelo)
chegaria às mesmas categorias e às mesmas classificações, ou discordaria em pontos
importantes. A validação feita até agora é uma amostragem manual pontual (10 tools
verificadas a mão contra o `reasoning` e o `source_code` reais), não uma verificação
sistemática de concordância.

### Por que isso é mais viável na Etapa 4 (classificação) do que nas Etapas 2 e 3

Das três etapas, só a classificação (`generate_consenso_categorization.py`) produz rótulos
diretamente comparáveis entre dois avaliadores: ambos classificam as *mesmas* tools contra a
*mesma* lista fixa de categorias. Já a geração de justificativa (texto livre) e a
identificação de categorias (categorias emergentes, nomeadas livremente por cada avaliador)
exigiriam uma etapa de reconciliação manual antes de qualquer comparação ser possível --
"Contradictory Parameter Documentation" de um avaliador é a mesma coisa que "Parameter
Mismatch" do outro? Isso não se mede automaticamente sem julgamento humano prévio.

### Proposta de implementação

1. **Segundo avaliador**: usar **DeepSeek** (`deepseek-flash`, já integrado ao projeto como
   juiz real em `config/judges.yaml`), não outra variante do Gemini -- dois modelos da mesma
   empresa/família tendem a compartilhar os mesmos pontos cegos, o que infla artificialmente a
   concordância medida. Não usar `qwen3-14b-ollama`: já observado, durante a validação manual,
   confundindo a tarefa (avaliando a qualidade do código em vez da divergência de nota) --
   baixa concordância ali mediria a fraqueza do modelo, não um desacordo metodológico
   genuíno.
2. **Amostra, não as ~2.700 tools inteiras**: rodar os dois avaliadores sobre um subconjunto
   (ordem de grandeza de 100-200 tools, sorteadas da amostra de consenso), suficiente para
   calcular uma métrica de concordância sem dobrar o custo/tempo do lote completo.
3. **Métrica**: como a classificação é multi-label (uma tool pode casar mais de uma
   categoria), reportar **Cohen's Kappa por categoria** (cada categoria tratada como uma
   pergunta binária independente: "o avaliador A marcou esta categoria para esta tool? e o
   B?") e/ou **Krippendorff's Alpha** multi-label como medida agregada única -- ambas são
   usuais na tradição de análise de conteúdo dirigida (Hsieh & Shannon, 2005) já citada na
   metodologia.
4. **O que precisa ser construído**: (a) rodar `generate_consenso_categorization.py` já
   existente com `--model deepseek-flash` sobre a mesma amostra, salvando em um JSONL
   separado; (b) um novo script de comparação que carrega os dois JSONLs de saída (mesmos
   `tool_uid`) e calcula o Kappa por categoria + a métrica agregada.
