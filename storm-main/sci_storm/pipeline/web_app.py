from __future__ import annotations

from pathlib import Path
import os
import sys

import gradio as gr

from ..agents import ExpertManager
from ..config import load_config
from ..engine import BackendAdapter, InferenceEngine
from ..engine.inference import GenerationContext
from ..tools import KISTIMCPClient, LocalRAGClient, TavilySearchClient
from .cli import _load_experts_from_yaml, _load_local_docs, _parse_outline_sections

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _hydrate_engine(config_path: Path, experts: ExpertManager) -> InferenceEngine:
    config = load_config(config_path)
    backend = BackendAdapter(config.backend)
    search_client = TavilySearchClient(
        api_key=config.tavily.api_key, max_results=config.tavily.max_results
    )
    rag_client = LocalRAGClient(config.rag.persist_directory)
    rag_client.ingest(_load_local_docs(Path("./data")))
    mcp_client = KISTIMCPClient(config.mcp)
    return InferenceEngine(
        backend=backend,
        expert_manager=experts,
        search_client=search_client,
        rag_client=rag_client,
        mcp_client=mcp_client,
    )


def _run_session(
    config_path: str,
    experts_path: str,
    goal: str,
    document_style: str,
    structural_requirements: str,
    outline_hint: str,
    output_language: str,
):
    config_path_obj = Path(config_path) if config_path else Path("config.yaml")
    experts = (
        _load_experts_from_yaml(Path(experts_path))
        if experts_path
        else ExpertManager()
    )
    if not experts.experts:
        from .cli import _build_default_experts

        experts = _build_default_experts(goal)

    engine = _hydrate_engine(config_path_obj, experts)
    ctx = GenerationContext(
        goal=goal,
        document_style=document_style,
        structural_requirements=structural_requirements,
        outline_format_hint=outline_hint,
        output_language=output_language,
    )

    outline_response = engine.generate_outline(ctx)
    ctx.outline = outline_response.content

    dialogue = engine.collaborative_dialogue(topic=goal, human_feedback="", turns=1)

    evidence, tavily_result = engine.run_expert_round(goal)
    combined_notes = list(evidence.values()) + dialogue

    sections = _parse_outline_sections(ctx.outline or "")
    drafted_sections = []
    for section_title in sections:
        section = engine.synthesize_section(ctx, section_title, combined_notes)
        drafted_sections.append(f"## {section_title}\n\n{section.content}")

    tool_usage = []
    if tavily_result.sources:
        tool_usage.append(f"Tavily search executed: {tavily_result.query}")
        tool_usage.extend(f"- {s.title}: {s.url}" for s in tavily_result.sources)
    else:
        tool_usage.append(f"Tavily search skipped/failed: {tavily_result.error}")
    tool_usage.append("MCP execution: not invoked in this session.")

    return (
        outline_response.content,
        "\n".join(dialogue) if dialogue else "No dialogue captured.",
        "\n".join(tool_usage),
        "\n\n".join(drafted_sections),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Sci-STORM Web UI") as demo:
        gr.Markdown("# Sci-STORM Web UI")
        with gr.Row():
            config_path = gr.Textbox(label="Config path", value="config.yaml")
            experts_path = gr.Textbox(label="Experts YAML", value="experts.example.yaml")
        goal = gr.Textbox(label="Goal / Research Question")
        document_style = gr.Textbox(label="Document style", value="Report")
        structural_requirements = gr.Textbox(label="Structural requirements", value="IMRaD")
        outline_hint = gr.Textbox(label="Outline format hint", value="Numbered sections")
        output_language = gr.Textbox(label="Output language", value="Korean")
        run_btn = gr.Button("Run Sci-STORM")

        outline_output = gr.Markdown(label="Outline")
        dialogue_output = gr.Markdown(label="Expert Dialogue")
        tool_output = gr.Markdown(label="Tool Usage")
        draft_output = gr.Markdown(label="Draft")

        run_btn.click(
            _run_session,
            inputs=[
                config_path,
                experts_path,
                goal,
                document_style,
                structural_requirements,
                outline_hint,
                output_language,
            ],
            outputs=[outline_output, dialogue_output, tool_output, draft_output],
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
