from __future__ import annotations

import datetime
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mcp_pipeline.github.models import RepoCandidate

logger = logging.getLogger("mcp_pipeline.clone_manager")

CLONE_TIMEOUT_SECONDS = 300
META_FILENAME = "repo_meta.json"


class CloneError(RuntimeError):
    pass


@dataclass
class RepoMeta:
    repo: RepoCandidate
    commit_sha: str
    cloned_at: str
    src_path: Path

    def to_dict(self) -> dict:
        d = self.repo.to_dict()
        d["commit_sha"] = self.commit_sha
        d["cloned_at"] = self.cloned_at
        return d

    @classmethod
    def from_meta_file(cls, meta_file: Path) -> RepoMeta:
        """Reconstructs a RepoMeta from a `repo_meta.json` written by
        clone_repo. `src_path` is never serialized (it's always
        `meta_file.parent / "src"` by construction — see repo_dir/clone_repo
        — so storing it would just be a redundant, staleness-prone copy of
        the same path) and is instead derived here from the file's own
        location, which makes this correct even if `dest_root` moved.
        """
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        commit_sha = raw.pop("commit_sha")
        cloned_at = raw.pop("cloned_at")
        repo = RepoCandidate.from_dict(raw)
        return cls(repo=repo, commit_sha=commit_sha, cloned_at=cloned_at, src_path=meta_file.parent / "src")


def slug_for_name_with_owner(name_with_owner: str) -> str:
    owner, name = name_with_owner.split("/", 1)
    return f"{owner}__{name}"


def slug_for(repo: RepoCandidate) -> str:
    return slug_for_name_with_owner(repo.name_with_owner)


def repo_dir(dest_root: Path, repo: RepoCandidate) -> Path:
    return dest_root / slug_for(repo)


def meta_file_path(dest_root: Path, repo: RepoCandidate) -> Path:
    return repo_dir(dest_root, repo) / META_FILENAME


def is_already_cloned(dest_root: Path, repo: RepoCandidate) -> bool:
    return meta_file_path(dest_root, repo).exists()


def clone_repo(repo: RepoCandidate, dest_root: Path) -> RepoMeta:
    """Shallow-clones `repo` (HEAD of the default branch only — Step 2 is
    static analysis of the working tree, not history, so `--depth 1` is
    correct, not just an efficiency shortcut) and records the exact commit
    analyzed. Anonymous HTTPS clone is used; if throttling is observed at
    scale, pass the PAT via `git -c http.extraHeader` instead (not needed for
    206 repos at this volume).
    """
    target = repo_dir(dest_root, repo)
    src_path = target / "src"

    try:
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
    except OSError as e:
        raise CloneError(f"não foi possível preparar o diretório para {repo.name_with_owner}: {e}") from e

    clone_url = f"https://github.com/{repo.name_with_owner}.git"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", "--no-tags", clone_url, str(src_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=CLONE_TIMEOUT_SECONDS,
        )
        rev_parse = subprocess.run(
            ["git", "-C", str(src_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # git can leave a partial working tree behind on failure (e.g. "clone
        # succeeded, but checkout failed" when disk space runs out mid-checkout
        # on a huge repo) -- left alone, that garbage both wastes disk forever
        # and makes a retry's `target.exists()` cleanup redundant at best, so
        # it's removed right here instead of trusting a future caller to do it.
        shutil.rmtree(target, ignore_errors=True)
        raise CloneError(f"git falhou para {repo.name_with_owner}: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(target, ignore_errors=True)
        raise CloneError(f"git clone excedeu o timeout para {repo.name_with_owner}") from e

    meta = RepoMeta(
        repo=repo,
        commit_sha=rev_parse.stdout.strip(),
        cloned_at=datetime.datetime.now(datetime.UTC).isoformat(),
        src_path=src_path,
    )
    (target / META_FILENAME).write_text(
        json.dumps(meta.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def clone_all(
    repos: list[RepoCandidate], dest_root: Path, errors_log: Path
) -> tuple[list[RepoMeta], list[tuple[RepoCandidate, str]]]:
    """Clones every repo, skipping ones already done (resumable) and
    continuing past individual failures instead of aborting the whole batch.

    `successes` always contains every repo currently available on disk
    (both freshly cloned in this call and previously cloned in an earlier
    run) — not just the ones newly cloned in this specific invocation — so
    a caller that drives Etapa 2 extraction from this return value gets the
    complete set on every run, including a resumed one.

    Returns (successes, [(repo, error_message), ...]).
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    errors_log.parent.mkdir(parents=True, exist_ok=True)

    successes: list[RepoMeta] = []
    failures: list[tuple[RepoCandidate, str]] = []

    with logging_redirect_tqdm(loggers=[logger]), tqdm(total=len(repos), desc="Clonando repositórios", unit="repo") as bar:
        for i, repo in enumerate(repos, 1):
            meta_file = meta_file_path(dest_root, repo)
            if meta_file.exists():
                logger.info("[%s/%s] %s já clonado, pulando", i, len(repos), repo.name_with_owner)
                successes.append(RepoMeta.from_meta_file(meta_file))
                bar.set_postfix(ok=len(successes), falhas=len(failures))
                bar.update(1)
                continue
            logger.info("[%s/%s] Clonando %s...", i, len(repos), repo.name_with_owner)
            try:
                meta = clone_repo(repo, dest_root)
                successes.append(meta)
            except CloneError as e:
                logger.warning("Falha ao clonar %s: %s", repo.name_with_owner, e)
                failures.append((repo, str(e)))
                with open(errors_log, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"repo": repo.name_with_owner, "error": str(e)}, ensure_ascii=False
                        )
                        + "\n"
                    )
            bar.set_postfix(ok=len(successes), falhas=len(failures))
            bar.update(1)

    logger.info("Clonagem concluída: %s sucesso(s), %s falha(s)", len(successes), len(failures))
    return successes, failures
