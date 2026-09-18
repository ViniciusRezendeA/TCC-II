#!/usr/bin/env python
"""Etapa 3 em paralelo: roda run_step3.py em N processos, um por chave de API disponível
para o judge escolhido, cada processo cobrindo sua própria partição do dataset (via
--num-keys/--key-index, ver run_step3.py::shard_for -- particionamento por hash estável do
tool_uid, calculado em memória a cada execução, sem gerar nem manter nenhum arquivo de
partição em disco).

Por que processos e não threads: cada SDK de provedor (GeminiJudge) lê sua API key de uma
env var (GOOGLE_API_KEY) no momento da chamada -- é global ao processo, então N threads no
mesmo interpretador compartilhariam a mesma env var e não daria para usar N chaves diferentes
ao mesmo tempo. N subprocessos, cada um com sua própria cópia do ambiente (uma variável
sobrescrita por processo), resolve isso sem qualquer mudança nas classes Judge existentes.

A largada dos processos é escalonada (--stagger-seconds, default 3s): o RateLimiter de cada
Judge só espaça a partir da 2ª chamada *daquele processo* -- a primeira chamada de cada
processo sai sem espera nenhuma. Subir os N processos todos no mesmo instante faz a primeira
chamada de cada um chegar praticamente junto na API (confirmado ao vivo: com 20 processos
simultâneos, ~45% da primeira rodada voltou 429 "resource exhausted", contra ~0% de erro de
autenticação nas mesmas chaves) -- espaçar a largada evita essa rajada sem precisar de
nenhuma coordenação entre os processos (cada um seria uma chave/projeto diferente de
qualquer forma, então não há cota compartilhada para negociar, só o timing da rajada).

Uso:
  # .env: GOOGLE_API_KEYS=chave1,chave2,chave3
  uv run python scripts/run_parallel_step3.py --judge gemini-3.5-flash-lite
  uv run python scripts/run_parallel_step3.py --judge gemini-3.5-flash-lite --limit 100 --concurrency 3
  uv run python scripts/run_parallel_step3.py --judge gemini-3.5-flash-lite --stagger-seconds 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from mcp_pipeline.evaluation.judges.registry import provider_for
from mcp_pipeline.logging_setup import setup_logging

logger = setup_logging("run_parallel_step3")

# provider (config/judges.yaml) -> nome da env var singular que o SDK do provedor lê
# diretamente (ver gemini_judge.py). A versão com "S" no final (ex.: GOOGLE_API_KEYS) é a
# lista separada por vírgula que este script consome. Só provedores cloud com API key
# entram aqui -- judges locais (llama.cpp) não usam nenhuma.
_PROVIDER_API_KEY_ENV = {
    "google": "GOOGLE_API_KEY",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 3 em paralelo, uma chave de API por processo.")
    parser.add_argument("--judge", type=str, required=True, help="Um único judge_id de config/judges.yaml -- o paralelismo aqui é por chave de API de um provedor, não entre judges diferentes.")
    parser.add_argument("--limit", type=int, default=None, help="Repassado a cada run_step3.py.")
    parser.add_argument("--scenarios", type=str, default=None, help="Repassado a cada run_step3.py (default: o default de run_step3.py).")
    parser.add_argument("--concurrency", type=int, default=None, help="Concorrência por processo, repassada a cada run_step3.py.")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dataset", type=str, default=None, help="Repassado a cada run_step3.py.")
    parser.add_argument("--stagger-seconds", type=float, default=3.0, help="Espera entre a largada de cada processo, para a 1ª chamada de cada chave não sair toda junto (ver docstring do módulo). 0 desliga o escalonamento.")
    args = parser.parse_args()

    if args.stagger_seconds < 0:
        parser.error("--stagger-seconds não pode ser negativo")

    provider = provider_for(args.judge)
    env_var = _PROVIDER_API_KEY_ENV.get(provider)
    if env_var is None:
        logger.error(
            "Provedor %r (judge %r) não tem rotação de chave de API por env var configurada "
            "-- este script serve para provedores cloud (google, mistral), não para judges "
            "locais (llama.cpp), que não usam API key.",
            provider, args.judge,
        )
        sys.exit(1)

    keys_env_name = f"{env_var}S"
    keys_raw = os.environ.get(keys_env_name, "")
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        logger.error(
            "%s não definido (ou vazio) no ambiente/.env -- defina uma lista de chaves "
            "separadas por vírgula para rodar em paralelo (ex.: %s=chave1,chave2,chave3).",
            keys_env_name, keys_env_name,
        )
        sys.exit(1)

    num_keys = len(keys)
    logger.info(
        "Judge %s (%s): %s chave(s) em %s -> %s processo(s) em paralelo, largada escalonada a cada %ss",
        args.judge, provider, num_keys, keys_env_name, num_keys, args.stagger_seconds,
    )

    base_cmd = [sys.executable, "-m", "mcp_pipeline.pipeline.run_step3", "--judges", args.judge, "--num-keys", str(num_keys)]
    if args.limit is not None:
        base_cmd += ["--limit", str(args.limit)]
    if args.scenarios is not None:
        base_cmd += ["--scenarios", args.scenarios]
    if args.concurrency is not None:
        base_cmd += ["--concurrency", str(args.concurrency)]
    if args.retry_failed:
        base_cmd += ["--retry-failed"]
    if args.dataset is not None:
        base_cmd += ["--dataset", args.dataset]

    processes: list[subprocess.Popen] = []
    for key_index, key in enumerate(keys):
        if key_index > 0 and args.stagger_seconds > 0:
            time.sleep(args.stagger_seconds)
        env = os.environ.copy()
        env[env_var] = key
        cmd = base_cmd + ["--key-index", str(key_index)]
        logger.info("[chave %s/%s] %s", key_index, num_keys, " ".join(cmd))
        processes.append(subprocess.Popen(cmd, env=env))

    exit_codes = [p.wait() for p in processes]

    failed = [key_index for key_index, code in enumerate(exit_codes) if code != 0]
    if failed:
        logger.error("%s/%s processo(s) terminaram com erro: índice(s) de chave %s", len(failed), num_keys, failed)
        sys.exit(1)

    logger.info("Todos os %s processos concluídos com sucesso.", num_keys)


if __name__ == "__main__":
    main()
