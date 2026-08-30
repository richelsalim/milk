"""Git operations for the research loop (FROZEN). One commit per iteration."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MUTABLE = ["train.py", "recsys"]


def git(*args, check=True) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{proc.stderr}")
    return proc.stdout.strip()


def is_dirty() -> bool:
    return bool(git("status", "--porcelain"))


def current_commit(short: bool = True) -> str:
    return git("rev-parse", "--short" if short else "HEAD", "HEAD") if short \
        else git("rev-parse", "HEAD")


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD")


def branch_exists(name: str) -> bool:
    return bool(git("branch", "--list", name))


def create_branch(name: str, base: str = "main") -> None:
    git("checkout", "-b", name, base)


def checkout_branch(name: str) -> None:
    git("checkout", name)


def diff_since(commit: str, paths=MUTABLE) -> str:
    return git("diff", commit, "--", *paths, check=False)


def restore_mutable(commit: str) -> None:
    git("checkout", commit, "--", *MUTABLE)


def commit_iteration(message: str, extra_paths: list[str]) -> str:
    """Stage the mutable surface plus the ledger paths, commit, return the short hash."""
    git("add", "--", *MUTABLE, *extra_paths)
    git("commit", "-m", message)
    return git("rev-parse", "--short", "HEAD")


def find_iteration_commit(i: int) -> str:
    """Short hash of the commit whose message starts with 'iter <i>:' (newest first)."""
    out = git("log", "--format=%h %s", "-200", check=False)
    for line in out.splitlines():
        h, _, msg = line.partition(" ")
        if msg.startswith(f"iter {i}:"):
            return h
    return ""
