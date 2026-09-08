import os
import platform
import sqlite3

from intuition_prototype.config import Settings
from intuition_prototype.llm import create_adapter
from intuition_prototype.storage import count_entries, initialize

# Disable optional Gradio telemetry before its first import.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr


def environment_health(settings: Settings) -> dict[str, str | int]:
    initialize(settings.database_path)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version,
        "gradio": gr.__version__,
        "mode": settings.mode,
        "llm": "Offline deterministic demo; no model or remote API calls",
        "database": str(settings.database_path.resolve()),
        "stored_demo_entries": count_entries(settings.database_path),
        "sqlite_status": "Schema initialization and query succeeded",
    }


def build_ui(settings: Settings | None = None) -> gr.Blocks:
    settings = Settings.from_env() if settings is None else settings
    adapter = create_adapter(settings)
    with gr.Blocks(analytics_enabled=False, title="Intuition Environment") as ui:
        gr.Markdown(
            "# Intuition environment\n"
            "**OFFLINE / DEMO ONLY.** No real LLM, research controller, "
            "or unrestricted tool execution. Inputs are not saved."
        )
        health = gr.JSON(value=environment_health(settings), label="Environment health")
        refresh = gr.Button("Refresh health")
        refresh.click(lambda: environment_health(settings), outputs=health, api_name=False)
        prompt = gr.Textbox(label="Demo input")
        output = gr.Textbox(label="Deterministic demo output", interactive=False)
        run = gr.Button("Echo in offline demo")
        run.click(adapter.generate, inputs=prompt, outputs=output, api_name=False)
    return ui


def main() -> None:
    build_ui().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        enable_monitoring=False,
        run_history=False,
        footer_links=[],
        theme=gr.themes.Default(
            font=["system-ui", "sans-serif"],
            font_mono=["ui-monospace", "monospace"],
        ),
    )


if __name__ == "__main__":
    main()
