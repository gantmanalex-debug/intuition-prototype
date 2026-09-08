import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    database_path: Path = Path(".runtime/intuition.sqlite3")

    def __post_init__(self) -> None:
        if self.mode != "demo":
            raise ValueError(
                "Only INTUITION_MODE=demo is supported. "
                "Remote mode requires an explicitly implemented LLM adapter."
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        database_path = env.get("INTUITION_DB_PATH", ".runtime/intuition.sqlite3")
        if not database_path.strip():
            raise ValueError("INTUITION_DB_PATH must not be empty.")
        return cls(
            mode=env.get("INTUITION_MODE", "demo"),
            database_path=Path(database_path),
        )
