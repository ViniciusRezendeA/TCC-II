#!/usr/bin/env python
"""Gera uma análise narrativa (texto em prosa, em português) dos resultados agregados da
Etapa 3 via Gemini, consumindo o mesmo pacote de dados que alimenta o dashboard
(scripts/generate_dashboard.py::build_dashboard_data()).

Segmentada por modelo de IA como o restante do dashboard: uma análise de "Por componente da
rubrica"/"Comparação de cenários"/"Significância (Wilcoxon)" para CADA breakdown (o
combinado "__all__" -- aba "Todos" -- e um por judge_id), a ser exibida junto da seção
correspondente quando aquela aba estiver selecionada -- não um bloco só, solto no topo da
página. "Por juiz" e "Divergências" não são segmentadas (essas duas seções do dashboard
também não são: a tabela "Por juiz" e a aba "Divergências" já mostram todos os juízes juntos,
sem seletor de aba), então ganham uma análise única cada.

Separado de generate_dashboard.py de propósito: chama uma API externa (rede, custo, rate
limit), então não roda a cada geração do dashboard nem no CI -- rode manualmente quando
quiser atualizar o texto. O resultado fica em cache (default:
data/analysis/narrative_analysis.json); generate_dashboard.py só lê esse arquivo se ele já
existir, nunca chama a API por conta própria.

Usa GOOGLE_API_KEY_NARRATIVE (NÃO GOOGLE_API_KEY/GOOGLE_API_KEYS, usadas pelos juízes reais
da Etapa 3 em config/judges.yaml) -- chave deliberadamente separada para esta chamada avulsa
nunca competir por rate limit/quota diária com as avaliações do experimento.

Saída estruturada (response_schema, mesmo padrão de
evaluation/judges/gemini_judge.py::GeminiJudge): um objeto com `overall_summary`, uma lista
`by_breakdown` (uma entrada por chave em `breakdown_keys` do payload enviado), e
`judges_analysis`/`divergences_analysis` -- nunca texto solto sem se saber a que
seção/breakdown do dashboard ele corresponde.

Uso:
  uv run python -m scripts.generate_narrative_analysis
  uv run python -m scripts.generate_narrative_analysis --model gemini-3.6-flash
  uv run python -m scripts.generate_narrative_analysis --output caminho/custom.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, Field

from mcp_pipeline.config import DATA_DIR
from mcp_pipeline.logging_setup import setup_logging
from scripts.dedupe_evaluations import dedupe_records
from scripts.generate_dashboard import build_dashboard_data, load_records

logger = setup_logging("generate_narrative_analysis")

DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Máximo de pares tool x componente x juiz enviados da aba Divergências -- já vem ordenado
# por |migração de quartil| e depois |diferença de nota| decrescentes (build_divergences_data),
# então truncar mantém os casos mais extremos sem estourar o prompt com centenas de pares
# (dashboard atual: 1000+).
MAX_DIVERGENCES_IN_PROMPT = 20


class BreakdownNarrative(BaseModel):
    key: str = Field(description="A chave deste breakdown, copiada exatamente de um item de 'breakdown_keys' no JSON de entrada ('__all__' ou um judge_id).")
    rubric_components: str = Field(description="Análise da seção 'Por componente da rubrica' PARA ESTE breakdown, em português, 2-4 frases.")
    scenario_comparison: str = Field(description="Análise da seção 'Comparação de cenários' PARA ESTE breakdown, em português, 2-4 frases. String vazia se este breakdown não tiver dados de comparação de cenário no JSON.")
    wilcoxon_significance: str = Field(description="Análise da seção 'Significância (Teste de Wilcoxon)' PARA ESTE breakdown, em português, 2-4 frases. String vazia quando key='__all__' (o teste não é aplicado misturando juízes) ou quando não houver dados de wilcoxon para este breakdown.")


class NarrativeAnalysis(BaseModel):
    overall_summary: str = Field(description="Resumo geral em português, 3-5 frases, conectando os achados entre breakdowns.")
    by_breakdown: list[BreakdownNarrative] = Field(description="Exatamente uma entrada para cada chave em 'breakdown_keys' do JSON de entrada, nesta ordem.")
    judges_analysis: str = Field(description="Análise da tabela 'Por juiz' (compara os juízes entre si -- não segmentada por modelo, é sobre todos ao mesmo tempo), em português, 2-4 frases.")
    divergences_analysis: str = Field(description="Análise da aba 'Divergências' (não segmentada por modelo, lista todos os juízes juntos), em português, 2-4 frases.")


SYSTEM_INSTRUCTION = """Você é um analista de dados auxiliando a redação da seção de resultados de um TCC sobre avaliação automática (LLM-as-a-Judge) da qualidade de descrições de ferramentas MCP (Model Context Protocol).

Você recebe um resumo estruturado (JSON) dos resultados agregados de um experimento comparando duas condições sobre a MESMA descrição em linguagem natural: 'description_only' (o juiz avalia só a descrição) e 'with_source' (o mesmo juiz avalia a mesma descrição, mas com acesso ao código-fonte correspondente). A pergunta de pesquisa central é se o acesso ao código muda a nota atribuída, e em quais dimensões da rubrica (purpose, guidelines, limitations, parameter_explanation, length_completeness, examples) esse efeito é mais forte.

O JSON de entrada traz `breakdown_keys` (a lista fixa de recortes: '__all__' = todos os juízes combinados, e um por judge_id) e `breakdowns` (os dados de cada um desses recortes, na mesma chave). Vários juízes podem discordar entre si -- é esperado, e vale comentar quando o padrão de significância ou o tamanho do efeito diverge de um juiz para outro.

Regras obrigatórias:
- Escreva sempre em português do Brasil, tom acadêmico-descritivo: descreva o que os dados mostram, nunca prescreva o que fazer a respeito.
- Cite números específicos do JSON fornecido (médias, p-valores, contagens, percentuais) em vez de generalidades vagas.
- Nunca invente números que não estejam no JSON. Se os dados de um breakdown/seção estiverem vazios ou ausentes, diga isso explicitamente na análise daquele item em vez de inventar conteúdo.
- p_valor_bh já é corrigido para comparações múltiplas (Benjamini-Hochberg) -- é esse valor que decide significância, não p_valor bruto.
- Para o breakdown '__all__', wilcoxon_significance deve ser string vazia (misturar juízes no mesmo teste pareado não é estatisticamente válido).

Estrutura de saída obrigatória: um objeto com `overall_summary`, `by_breakdown` (exatamente uma entrada por chave em `breakdown_keys`, nesta ordem, cada uma com `key` igual à chave e os 3 campos de análise), `judges_analysis` e `divergences_analysis`."""


def _round_floats(obj, ndigits: int = 3):
    """Trims floats before serializing into the prompt -- a model describing a trend in
    prose doesn't need float64 precision (0.123456789), and shorter numbers mean fewer
    tokens for the same information."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def build_payload(dashboard_data: dict) -> dict:
    """Extrai só o que cada seção da aba Visão Geral (mais Divergências) precisa -- não o
    dataset bruto avaliação a avaliação, que estouraria o prompt sem agregar nada a uma
    análise de tendências. Mesma fonte de dados que o HTML renderiza
    (build_dashboard_data()), então o texto gerado nunca descreve um número que o dashboard
    não mostra também.

    `breakdowns` espelha DATA.breakdowns do dashboard (mesmas chaves, mesma ordem -- inclui
    '__all__') para o modelo devolver uma análise por breakdown, igual ao seletor de aba
    "Todos"/por-juiz que já existe no HTML.
    """
    breakdown_keys = list(dashboard_data["breakdowns"].keys())
    breakdowns_payload = {
        key: {
            "rubric_components": b["rubric_components"],
            "scenario_comparison": b["scenario_comparison"],
            "wilcoxon": b["wilcoxon"],  # [] para "__all__" e para breakdowns sem par suficiente
        }
        for key, b in dashboard_data["breakdowns"].items()
    }
    divergencias = dashboard_data["divergences"][:MAX_DIVERGENCES_IN_PROMPT]
    payload = {
        "breakdown_keys": breakdown_keys,
        "overall": dashboard_data["overall"],
        "breakdowns": breakdowns_payload,
        "por_juiz": [
            {k: v for k, v in j.items() if k != "flag"} | ({"flag": j["flag"]} if j.get("flag") else {})
            for j in dashboard_data["judges"]
        ],
        "maiores_divergencias": [
            {
                "componente": d["componente_label"],
                "tool": d["tool_name"],
                "repo": d["repo"],
                "juiz": d["judge_id"],
                "nota_description_only": d["description_only"]["score"],
                "nota_with_source": d["with_source"]["score"],
                "diferenca": d["diff"],
                "quartil_description_only": d["quartil_description_only"],
                "quartil_with_source": d["quartil_with_source"],
            }
            for d in divergencias
        ],
        "total_divergencias": len(dashboard_data["divergences"]),
        "metodo_divergencia": dashboard_data["meta"]["divergence_method"],
    }
    return _round_floats(payload)


def call_gemini(payload: dict, model: str) -> NarrativeAnalysis:
    api_key = os.environ.get("GOOGLE_API_KEY_NARRATIVE")
    if not api_key:
        raise SystemExit(
            "GOOGLE_API_KEY_NARRATIVE não definida no ambiente/.env -- chave separada de "
            "GOOGLE_API_KEY/GOOGLE_API_KEYS (usadas pelos juízes da Etapa 3), de propósito, "
            "para esta chamada avulsa nunca competir por rate limit/quota diária com as "
            "avaliações reais do experimento. Veja .env.example."
        )
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=model,
            contents=json.dumps(payload, ensure_ascii=False),
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=NarrativeAnalysis,
                max_output_tokens=8192,
            ),
        )
    except genai_errors.APIError as e:
        raise SystemExit(f"chamada ao Gemini falhou: {e.code} {e.message}") from e

    result = response.parsed
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    if result is None:
        raise SystemExit(f"Gemini não retornou saída estruturada válida (finish_reason={finish_reason!r})")
    if not isinstance(result, NarrativeAnalysis):
        result = NarrativeAnalysis.model_validate(result)

    got_keys = {b.key for b in result.by_breakdown}
    missing = [k for k in payload["breakdown_keys"] if k not in got_keys]
    if missing:
        logger.warning("Gemini não devolveu análise para os breakdowns %s -- dashboard mostrará essas em branco.", missing)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera análise narrativa (Gemini) dos resultados agregados da Etapa 3.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"model_id do Gemini (default: {DEFAULT_MODEL}).")
    parser.add_argument("--evaluations-dir", type=Path, default=None, help="Diretório com {judge_id}.jsonl (default: data/evaluations).")
    parser.add_argument("--prompt-version", type=str, default=None, help="Restringe aos dados desta versão (default: a mais recente, ver generate_dashboard.py).")
    parser.add_argument("--output", type=Path, default=None, help="Caminho de saída (default: data/analysis/narrative_analysis.json).")
    args = parser.parse_args()

    eval_dir = args.evaluations_dir or (DATA_DIR / "evaluations")
    records = load_records(eval_dir, None)
    if not records:
        logger.error("nenhuma avaliação encontrada em %s", eval_dir)
        raise SystemExit(1)
    records = dedupe_records(records)

    dashboard_data = build_dashboard_data(records, prompt_version=args.prompt_version)
    payload = build_payload(dashboard_data)

    logger.info(
        "enviando resumo agregado (%s breakdowns: %s, %s de %s divergências) para o modelo %s",
        len(payload["breakdown_keys"]), payload["breakdown_keys"],
        len(payload["maiores_divergencias"]), payload["total_divergencias"], args.model,
    )
    narrative = call_gemini(payload, args.model)

    output_path = args.output or (DATA_DIR / "analysis" / "narrative_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "prompt_version": dashboard_data["meta"]["active_prompt_version"],
        "generated_at": dashboard_data["meta"]["generated_at"],
        **narrative.model_dump(),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("análise narrativa salva em %s -- rode scripts.generate_dashboard de novo para embutir no HTML.", output_path)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        logger.exception("geração da análise narrativa falhou")
        raise SystemExit(1)
