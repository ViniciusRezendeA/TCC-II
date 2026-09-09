#!/usr/bin/env python
"""Health check para Gemini API (free tier).

Valida conectividade, autenticação e avisos sobre limitações da API gratuita.

Modelo testado: gemini-3.6-flash. NÃO usar variantes "-live-preview": elas só suportam
bidiGenerateContent via WebSocket, não generate_content() (usado aqui e em gemini_judge.py) --
retornam 400 INVALID_ARGUMENT nesse método, o que este script antes confundia com chave
de API inválida.

Limitações conhecidas do Gemini free tier:
- Rate limit: 15 requisições por minuto
- Quota: 1M tokens por dia
- Sem cached content support
- Sem fine-tuning

Uso:
  uv run python scripts/check_gemini_free_tier.py
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from mcp_pipeline.evaluation.prompts import RUBRIC_SYSTEM_PROMPT, LIKERT_SCALE

# Free tier rate limits
FREE_TIER_RPM_LIMIT = 15  # Requisições por minuto
FREE_TIER_DAILY_TOKENS = 1_000_000
ESTIMATED_TOKENS_PER_TOOL = 3_000  # Prompt + resposta típica

# Teste precisará de ~3k tokens (estimado para um tool)
TEST_PROMPT = f"""Teste de conectividade com a API gratuita do Gemini.

Você é um avaliador especializado. Responda em JSON com a seguinte estrutura:
{{"purpose": {{"score": 3, "reasoning": "Teste ok"}}}}.

Ignore o resto e responda apenas com o JSON."""


def calculate_daily_capacity() -> dict:
    """Calcula capacidade diária sob limitações do free tier."""
    rppm = FREE_TIER_RPM_LIMIT  # requisições por minuto
    minutes_per_day = 24 * 60
    max_requests_per_day = rppm * minutes_per_day

    estimated_tools_per_day = (FREE_TIER_DAILY_TOKENS // ESTIMATED_TOKENS_PER_TOOL)
    # Assumindo 2 cenários por tool (com/sem source code)
    actual_tools_per_day = estimated_tools_per_day // 2

    return {
        "max_requests_per_day": max_requests_per_day,
        "estimated_tools_per_day": estimated_tools_per_day,
        "actual_tools_per_day_2_cenarios": actual_tools_per_day,
        "tokens_per_day": FREE_TIER_DAILY_TOKENS,
        "estimated_tokens_per_tool": ESTIMATED_TOKENS_PER_TOOL,
    }


def check_connectivity(api_key: str, model_id: str = "gemini-3.6-flash") -> tuple[bool, str, dict]:
    """Testa conectividade com Gemini."""
    print(f"\n{'='*70}")
    print(f"Testando conectividade com {model_id}...")
    print(f"{'='*70}")

    if not api_key:
        return False, "API_KEY não foi definida", {}

    try:
        client = genai.Client(api_key=api_key)

        # Teste 1: Lista de modelos (sem usar tokens)
        print("→ Conectando ao endpoint de modelos...")
        models = list(client.models.list())
        model_names = [m.name.split("/")[-1] for m in models if "gemini" in m.name.lower()]
        print(f"  ✓ Acesso OK. Encontrados {len(model_names)} modelos Gemini disponíveis")
        print(f"    Modelos: {', '.join(sorted(set(model_names))[:5])}...")

        # Verifica se modelo solicitado está disponível
        # Exclui variantes que não aceitam generate_content: "-live"/"-preview" (streaming
        # via WebSocket ou instável), "-image"/"-tts"/"-audio"/"-transcribe" (outras
        # modalidades), "embedding" -- ver bug do modelo -live-preview identificado acima.
        _NON_TEXT_MARKERS = ("-live", "-preview", "-image", "-tts", "-audio", "-transcribe", "embedding")
        text_model_names = [m for m in model_names if not any(marker in m for marker in _NON_TEXT_MARKERS)]

        if model_id not in model_names:
            print(f"  ⚠ Aviso: {model_id} não está disponível")
            # Encontra alternativa (preferência: famílias mais recentes primeiro -- 2.5-flash e
            # anteriores retornam 404 "no longer available to new users" para chaves novas)
            for pattern in ["3.6-flash", "3.5-flash", "3.1-flash-lite", "2.5-flash"]:
                alternatives = [m for m in text_model_names if pattern in m]
                if alternatives:
                    model_id = alternatives[0]
                    print(f"    Usando alternativa: {model_id}")
                    break
            else:
                # Se nenhuma flash disponível, tenta qualquer gemini de texto
                if text_model_names:
                    model_id = next((m for m in text_model_names if "flash" in m or "pro" in m), text_model_names[0])
                    print(f"    Usando: {model_id}")
                else:
                    return False, f"Nenhum modelo Gemini disponível", {}

        # Teste 2: Generate content (usa tokens)
        print(f"\n→ Testando generate_content com {model_id}...")
        started = time.time()
        response = client.models.generate_content(
            model=model_id,
            contents="Responda com apenas 'ok'.",
            config=genai_types.GenerateContentConfig(
                system_instruction="Você é um assistente.",
                max_output_tokens=10,
            ),
        )
        latency_ms = (time.time() - started) * 1000

        if response.usage_metadata:
            usage = response.usage_metadata
            print(f"  ✓ Sucesso!")
            print(f"    Latência: {latency_ms:.0f}ms")
            print(f"    Tokens: {usage.prompt_token_count or 0} entrada → {usage.candidates_token_count or 0} saída")
            return True, "OK", {
                "latency_ms": latency_ms,
                "prompt_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
            }
        else:
            return False, "Sem usage_metadata na resposta", {}

    except genai_errors.ClientError as e:
        # ClientError captura 400, 401, etc. (Authentication, Permission, Invalid)
        # NB: "INVALID_ARGUMENT" por si só NÃO indica chave inválida -- é o status HTTP 400
        # genérico do Google, usado para qualquer erro de validação da requisição (ex.: pedir
        # generate_content de um modelo -live-preview, que só aceita bidiGenerateContent via
        # WebSocket). Só a mensagem "API key not valid" identifica esse problema de fato.
        error_msg = str(e)
        if "API key not valid" in error_msg:
            return False, f"Autenticação falhou: Chave de API inválida ou expirada", {}
        elif "404" in error_msg or "NOT_FOUND" in error_msg or "no longer available" in error_msg:
            return False, f"Modelo indisponível: {error_msg[:150]}", {}
        elif "permission" in error_msg.lower():
            return False, f"Permissão negada: {error_msg[:100]}", {}
        else:
            return False, f"Erro do cliente (400): {error_msg[:150]}", {}
    except genai_errors.ServerError as e:
        error_msg = str(e)
        if "429" in error_msg:
            return False, f"Rate limit atingido (429): {error_msg[:100]}", {}
        elif "503" in error_msg or "500" in error_msg:
            return False, f"Serviço indisponível: {error_msg[:100]}", {}
        else:
            return False, f"Erro do servidor: {error_msg[:100]}", {}
    except genai_errors.APIError as e:
        return False, f"Erro de API: {str(e)[:150]}", {}
    except Exception as e:
        return False, f"Erro inesperado: {type(e).__name__}: {str(e)[:150]}", {}


def check_rubric_compatibility(api_key: str, model_id: str = "gemini-3.6-flash") -> tuple[bool, str]:
    """Testa se o modelo consegue fazer parsing da rubrica com JSON schema."""
    print(f"\n{'='*70}")
    print("Testando compatibilidade com rubrica (JSON schema)...")
    print(f"{'='*70}")

    try:
        from mcp_pipeline.evaluation.judges.base import RubricScores

        client = genai.Client(api_key=api_key)

        # Tentativa de usar response_schema (requer modelo compatível)
        print("→ Testando structured output (response_schema)...")

        # Prompt simplificado para teste
        simple_payload = {
            "name": "test_tool",
            "description": "Uma ferramenta de teste simples.",
        }

        response = client.models.generate_content(
            model=model_id,
            contents=f"Responda com JSON sobre essa ferramenta: {simple_payload}",
            config=genai_types.GenerateContentConfig(
                system_instruction="Você é um avaliador. Responda em JSON válido.",
                response_mime_type="application/json",
                # Tenta usar schema, mas nem todos os modelos suportam
                # response_schema=RubricScores,
                max_output_tokens=100,
            ),
        )

        if response.text:
            print("  ✓ JSON output funciona")
            return True, "OK"
        else:
            return False, "Sem resposta em JSON"

    except genai_errors.ClientError as e:
        error_msg = str(e)
        if "API key not valid" in error_msg:
            return False, "Chave de API inválida"
        elif "response_schema" in error_msg or "json_schema" in error_msg:
            return False, f"Modelo não suporta structured output"
        else:
            return False, f"Erro: {error_msg[:100]}"
    except genai_errors.APIError as e:
        error_msg = str(e)
        if "response_schema" in error_msg or "json_schema" in error_msg:
            return False, f"Modelo não suporta structured output"
        else:
            return False, f"Erro de API: {error_msg[:100]}"
    except Exception as e:
        error_msg = str(e)
        if "response_schema" in error_msg or "json_schema" in error_msg:
            return False, f"Modelo não suporta structured output"
        else:
            return False, f"Erro inesperado: {type(e).__name__}"


def main():
    print("\n" + "=" * 70)
    print("🔍 Health Check: Gemini API Free Tier")
    print("=" * 70)

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("\n✗ Erro: Chave de API não encontrada")
        print("  Configure GOOGLE_API_KEY ou GEMINI_API_KEY em .env")
        sys.exit(1)

    print(f"\nChave de API encontrada: {api_key[:20]}...")

    # Test 1: Conectividade
    success, msg, usage_data = check_connectivity(api_key)

    if not success:
        print(f"\n✗ FALHA na conectividade: {msg}")
        sys.exit(1)

    print(f"\n✓ Conectividade: OK")

    # Test 2: Rubrica
    rubric_ok, rubric_msg = check_rubric_compatibility(api_key)
    print(f"  Rubrica (JSON schema): {'✓ OK' if rubric_ok else '⚠ ' + rubric_msg}")

    # Análise de capacidade
    print(f"\n{'='*70}")
    print("⚠️  LIMITAÇÕES DO FREE TIER (IMPORTANTE)")
    print(f"{'='*70}")

    capacity = calculate_daily_capacity()

    print(f"\n📊 Capacidade Diária:")
    print(f"  Rate limit: {FREE_TIER_RPM_LIMIT} requisições por minuto")
    print(f"  Max requisições/dia: {capacity['max_requests_per_day']:,}")
    print(f"  Quota tokens/dia: {capacity['tokens_per_day']:,}")
    print(f"  Tokens por tool (est.): {capacity['estimated_tokens_per_tool']:,}")

    print(f"\n⚡ Estimativas para este projeto (12.171 tools × 2 cenários):")
    print(f"  Tools que podem avaliar/dia: ~{capacity['actual_tools_per_day_2_cenarios']}")
    print(f"  Dias para dataset completo: ~{(12171 * 2) // max(1, capacity['actual_tools_per_day_2_cenarios'])}")

    print(f"\n🚨 Implicações:")
    print(f"  • A quota de 1M tokens/dia é muito restritiva para 12.171 tools")
    print(f"  • Recomenda-se usar Gemini como SECUNDÁRIO, não primário")
    print(f"  • Melhor opção: Qwen local + Llama local")
    print(f"  • Se usar Gemini: considere --limit pequeno para testes")

    print(f"\n📋 Recomendações:")
    print(f"  1. Mantenha Qwen e Llama como juízes primários (locais)")
    print(f"  2. Use Gemini para subset de validação (~10-20 tools)")
    print(f"  3. Para dataset completo, espere upgrade do plano")
    print(f"  4. Monitore quota diária em console.cloud.google.com")

    # Teste 3: Configuração atual
    print(f"\n{'='*70}")
    print("Verificando configuração em config/judges.yaml...")
    print(f"{'='*70}")

    from mcp_pipeline.evaluation.judges.registry import load_judges

    judges = load_judges()
    gemini_judges = [j for j in judges if j.provider == "google"]

    if gemini_judges:
        print(f"\n✓ Juízes Gemini habilitados em config/judges.yaml:")
        for j in gemini_judges:
            print(f"  - {j.judge_id} ({j.model_id})")
        print(f"\n⚠️  AVISO: Com a quota gratuita, Gemini deve estar DESABILITADO")
        print(f"  Edite config/judges.yaml e marque como: enabled: false")
    else:
        print(f"\n✓ Nenhum juiz Gemini habilitado (correto para free tier)")

    print(f"\n{'='*70}")
    print("✓ Health check concluído")
    print(f"{'='*70}")
    print(f"\nProximo passo:")
    print(f"  1. Revise limitações acima")
    print(f"  2. Mantenha Gemini desabilitado em config/judges.yaml")
    print(f"  3. Execute Etapa 3 com Qwen + Llama locais")
    print(f"  4. Se quiser testar Gemini: use --limit 5 --judges gemini-2.5-flash-lite")


if __name__ == "__main__":
    main()
