"""environment configuration loading from .env files."""

from pathlib import Path

from dotenv import load_dotenv


def load_env(workdir: Path | None = None) -> None:
    """load environment variables from .env file.

    searches in order: workdir/.env, cwd/.env
    does not override already-set environment variables.
    """
    if workdir and (workdir / ".env").exists():
        load_dotenv(workdir / ".env", override=False)
    else:
        load_dotenv(override=False)
