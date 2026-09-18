#!/usr/bin/env python
"""Valida cada chave em GOOGLE_API_KEYS individualmente antes de rodar
scripts/run_parallel_step3.py em lote -- separa "chave inválida de verdade" (vai falhar
aqui também) de "só falha sob carga paralela" (passa aqui, mas pode 429 no lote real).

Usa client.models.list() para cada chave: é a chamada mais barata da API (não conta como
generate_content, então não consome a cota de RPM/RPD do free tier -- ver
check_gemini_free_tier.py, de onde essa escolha foi copiada), então validar as 20 chaves
aqui não compete pela cota que o lote real vai precisar depois.

Uso:
  # .env: GOOGLE_API_KEYS=chave1,chave2,chave3,...
  uv run python scripts/validate_gemini_keys.py
"""

from __future__ import annotations

import os
import sys
import time

from google import genai
from google.genai import errors as genai_errors

from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("validate_gemini_keys")

# Espaço entre as chamadas de validação -- não é sobre cota (models.list() não consome RPM
# de generate_content), é só para não abrir 20 conexões simultâneas à toa.
_DELAY_SECONDS = 0.5


def _mask(key: str) -> str:
    return f"...{key[-6:]}" if len(key) > 6 else "..."


def validate_key(key: str) -> tuple[bool, str]:
    try:
        client = genai.Client(api_key=key)
        list(client.models.list())
        return True, "OK"
    except genai_errors.APIError as e:
        return False, f"{e.code}: {e.message}"
    except Exception as e:  # noqa: BLE001 -- uma chave ruim não pode derrubar a validação das outras
        return False, str(e)


def main() -> None:
    keys_raw = os.environ.get("GOOGLE_API_KEYS", "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        logger.error("GOOGLE_API_KEYS não definido (ou vazio) no ambiente/.env.")
        sys.exit(1)

    logger.info("Validando %s chave(s)...", len(keys))
    results: list[tuple[int, str, bool, str]] = []
    for i, key in enumerate(keys):
        ok, detail = validate_key(key)
        results.append((i, _mask(key), ok, detail))
        logger.info("[%s/%s] %s -> %s", i, len(keys), _mask(key), "OK" if ok else f"FALHOU ({detail})")
        if i < len(keys) - 1:
            time.sleep(_DELAY_SECONDS)

    valid = [r for r in results if r[2]]
    invalid = [r for r in results if not r[2]]

    logger.info("Resultado: %s válida(s), %s inválida(s) de %s", len(valid), len(invalid), len(keys))
    if invalid:
        logger.error("Chaves inválidas (índice: motivo):")
        for i, masked, _, detail in invalid:
            logger.error("  [%s] %s: %s", i, masked, detail)
        sys.exit(1)


if __name__ == "__main__":
    main()
