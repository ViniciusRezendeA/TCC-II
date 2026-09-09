#!/usr/bin/env python
"""Health check para servidores llama.cpp locais.

Valida conectividade e autenticação com os juízes locais via llama-server.

Uso:
  uv run python scripts/check_local_llm_servers.py
"""

from __future__ import annotations

import json
import os
import sys

import requests

def check_server(name: str, base_url: str, bearer_token: str, model_id: str) -> bool:
    """Testa conectividade e autenticação com um servidor."""
    print(f"\n{'='*60}")
    print(f"Checando: {name}")
    print(f"  URL: {base_url}")
    print(f"  Model: {model_id}")
    print(f"{'='*60}")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer_token}",
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Você é um assistente conciso."},
            {"role": "user", "content": "Responda com apenas '1' se conseguiu ler esta mensagem."},
        ],
        "max_tokens": 10,
    }

    try:
        print(f"→ Conectando a {url}...")
        response = requests.post(url, headers=headers, json=payload, timeout=None)

        print(f"  Status HTTP: {response.status_code}")

        if response.status_code == 401:
            print("  ✗ FALHA: Autenticação rejeitada (401)")
            return False

        if response.status_code == 404:
            print("  ✗ FALHA: Endpoint não encontrado (404)")
            return False

        if response.status_code >= 500:
            print(f"  ✗ FALHA: Erro do servidor ({response.status_code})")
            print(f"    Response: {response.text[:200]}")
            return False

        if response.status_code != 200:
            print(f"  ⚠ Aviso: Status inesperado ({response.status_code})")
            print(f"    Response: {response.text[:200]}")

        try:
            result = response.json()
        except json.JSONDecodeError:
            print(f"  ✗ FALHA: Resposta não é JSON válido")
            print(f"    Response: {response.text[:200]}")
            return False

        if "error" in result:
            error = result.get("error", {})
            print(f"  ✗ FALHA: API retornou erro")
            print(f"    {error.get('message', str(error))}")
            return False

        if "choices" not in result or not result["choices"]:
            print(f"  ✗ FALHA: Resposta sem choices")
            return False

        content = result["choices"][0].get("message", {}).get("content", "")
        print(f"  ✓ Sucesso!")
        print(f"    Modelo respondeu: '{content.strip()}'")

        usage = result.get("usage", {})
        if usage:
            print(f"    Tokens: {usage.get('prompt_tokens', 0)} entrada, {usage.get('completion_tokens', 0)} saída")

        return True

    except requests.ConnectionError as e:
        print(f"  ✗ FALHA: Conexão recusada")
        print(f"    {e}")
        return False
    except requests.Timeout as e:
        print(f"  ✗ FALHA: Timeout na conexão")
        print(f"    {e}")
        return False
    except requests.RequestException as e:
        print(f"  ✗ FALHA: Erro HTTP")
        print(f"    {e}")
        return False
    except Exception as e:
        print(f"  ✗ FALHA: Erro inesperado")
        print(f"    {type(e).__name__}: {e}")
        return False


def main():
    print("\n🔍 Health Check: Servidores LLM Locais")
    print("=" * 60)

    # Lê credenciais de .env
    from dotenv import load_dotenv
    load_dotenv()

    llama_url = os.getenv("LOCAL_LLM_BASE_URL", "http://192.168.15.15:8090/v1")
    llama_token = os.getenv("LOCAL_LLM_BEARER_TOKEN", "")

    qwen_url = os.getenv("QWEN_LLM_BASE_URL", "")
    qwen_token = os.getenv("QWEN_LLM_BEARER_TOKEN", "")

    if not llama_token:
        print("\n✗ Erro: Credenciais não encontradas em .env")
        print("  Certifique-se que LOCAL_LLM_BEARER_TOKEN está definida")
        sys.exit(1)

    results = {}

    results["llama"] = check_server(
        "Llama (llama-uncensored)",
        llama_url,
        llama_token,
        "llama-uncensored",
    )

    if qwen_url and qwen_token:
        results["qwen"] = check_server(
            "Qwen2.5-14B-Instruct",
            qwen_url,
            qwen_token,
            "qwen2.5-14b-instruct",
        )
    else:
        print("\n⏭️  Pulando Qwen2.5-14B-Instruct: QWEN_LLM_BASE_URL/QWEN_LLM_BEARER_TOKEN não configuradas em .env")

    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"{'='*60}")

    for name, success in results.items():
        status = "✓ OK" if success else "✗ FALHA"
        print(f"{name:20} {status}")

    if all(results.values()):
        print("\n✓ Todos os servidores estão acessíveis!")
        print("\nPróximos passos:")
        print("  1. Execute o teste piloto:")
        print("     uv run python scripts/test_local_judges.py --sample-size 20")
        print("  2. Se o teste passar, execute a Etapa 3 completa:")
        print("     uv run python -m mcp_pipeline.pipeline.run_step3 --limit 100")
        sys.exit(0)
    else:
        print("\n✗ Alguns servidores não estão acessíveis")
        print("  Verifique:")
        print("  1. Se os servidores llama.cpp estão rodando")
        print("  2. Se o IP e porta estão corretos")
        print("  3. Se o Bearer token está correto em .env")
        sys.exit(1)


if __name__ == "__main__":
    main()
