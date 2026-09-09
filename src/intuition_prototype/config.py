import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_MODEL_PATH = Path(".runtime/stage4-v1/model.json")


@dataclass(frozen=True)
class Settings:
    mode: str = "demo"
    database_path: Path = Path(".runtime/intuition.sqlite3")
    model_path: Path = DEFAULT_MODEL_PATH

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
        model_path = env.get("INTUITION_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        if not model_path.strip():
            raise ValueError("INTUITION_MODEL_PATH must not be empty.")
        return cls(
            mode=env.get("INTUITION_MODE", "demo"),
            database_path=Path(database_path),
            model_path=Path(model_path),
        )
