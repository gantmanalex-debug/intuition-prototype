import os
import platform
import sqlite3

from intuition_prototype.config import Settings
from intuition_prototype.llm import create_adapter
from intuition_prototype.storage import count_entries, initialize
from intuition_prototype.simulator import SCENARIOS
from intuition_prototype.stage1 import reset_stage1, run_stage1

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
            "**OFFLINE / SCRIPTED ONLY.** No real LLM, learned controller, "
            "or unrestricted tool execution. Echo inputs are not saved. "
            "Stage 1 simulator traces are saved locally in SQLite."
        )
        health = gr.JSON(value=environment_health(settings), label="Environment health")
        refresh = gr.Button("Refresh health")
        refresh.click(lambda: environment_health(settings), outputs=health, api_name=False)
        prompt = gr.Textbox(label="Demo input")
        output = gr.Textbox(label="Deterministic demo output", interactive=False)
        run = gr.Button("Echo in offline demo")
        run.click(adapter.generate, inputs=prompt, outputs=output, api_name=False)
        gr.Markdown(
            "## Stage 1: bounded service simulator\n"
            "Case names are evaluator labels, not inputs to the SCRIPTED inquiry policy. "
            "Every Run starts fresh. No learned intuition or research validation is claimed."
        )
        scenario = gr.Dropdown(
            choices=list(SCENARIOS), value="retry_amplification", label="Evaluator scenario",
        )
        seed = gr.Number(value=7, precision=0, label="Reproducible seed (0..4294967295)")
        budget = gr.Slider(0, 20, value=10, step=1, label="Action budget")
        run_case = gr.Button("Run scripted investigation", variant="primary")
        reset_case = gr.Button("Reset scenario / clear viewer")
        episode_id = gr.Textbox(label="Persisted episode ID", interactive=False)
        trace = gr.Markdown("Choose a case and run the scripted baseline.")

        def run_case_callback(case: str, case_seed: int, case_budget: int) -> tuple[str, str]:
            try:
                return run_stage1(settings.database_path, case, case_seed, case_budget)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        def reset_case_callback(case: str, case_seed: int) -> tuple[str, str]:
            try:
                return reset_stage1(case, case_seed)
            except ValueError as error:
                raise gr.Error(str(error)) from error

        run_case.click(
            run_case_callback, inputs=[scenario, seed, budget],
            outputs=[episode_id, trace], api_name=False,
        )
        reset_case.click(
            reset_case_callback, inputs=[scenario, seed],
            outputs=[episode_id, trace], api_name=False,
        )
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
