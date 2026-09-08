from typing import Protocol

from intuition_prototype.config import Settings


class LLMAdapter(Protocol):
    def generate(self, prompt: str) -> str: ...


class DemoAdapter:
    def generate(self, prompt: str) -> str:
        return (
            "OFFLINE DEMO - deterministic echo, not an LLM response.\n"
            f"Input: {prompt}"
        )


def create_adapter(settings: Settings) -> LLMAdapter:
    if settings.mode == "demo":
        return DemoAdapter()
    raise ValueError(f"No LLM adapter implemented for mode: {settings.mode}")
