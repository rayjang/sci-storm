from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from ..agents import ExpertManager
from ..config import AppConfig, load_config
from ..engine import BackendAdapter, InferenceEngine
from ..engine.inference import GenerationContext
from ..tools import KISTIMCPClient, LocalRAGClient, TavilySearchClient


os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

console = Console()
app = typer.Typer(help="Sci-STORM: collaborative scientific authoring agent.")


def _load_local_docs(data_dir: Path) -> list[str]:
    docs = []
    if not data_dir.exists():
        return docs
    for path in data_dir.glob("**/*"):
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            docs.append(path.read_text(encoding="utf-8"))
    return docs


def _build_default_experts(goal: str) -> ExpertManager:
    experts = ExpertManager()
    experts.register(
        name="Literature Reviewer",
        focus="Source recent peer-reviewed findings",
        system_prompt=(
            "Act like a meticulous literature reviewer; summarize peer-reviewed "
            "evidence, key datasets, and state-of-the-art techniques."
        ),
    )
    experts.register(
        name="Methodologist",
        focus="Design experiments",
        system_prompt=(
            "Outline reproducible experimental designs, controls, and evaluation "
            "metrics tailored to the research goal."
        ),
    )
    experts.register(
        name="Data Engineer",
        focus="Implementation constraints",
        system_prompt=(
            "Identify computational constraints, data preprocessing needs, and "
            "implementation pitfalls; surface code sketches when helpful."
        ),
    )
    experts.register(
        name="Policy Analyst",
        focus="Policy and governance impact",
        system_prompt=(
            "Assess regulatory, ethical, and societal impacts of the research "
            "topic, focusing on policy implications and compliance."
        ),
    )
    experts.register(
        name="Systems Architect",
        focus="Scalable system design",
        system_prompt=(
            "Provide a scalable system design perspective, including deployment, "
            "monitoring, and reliability constraints."
        ),
    )
    return experts


def _load_experts_from_yaml(path: Path) -> ExpertManager:
    import yaml

    experts = ExpertManager()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in data.get("experts", []):
        system_prompt = item.get("system_prompt", "")
        if not system_prompt:
            role = item.get("role", "")
            perspective = item.get("perspective", "")
            question_focus = item.get("question_focus", [])
            system_prompt = "\n".join(
                line
                for line in [
                    f"Role: {role}" if role else "",
                    f"Perspective: {perspective}" if perspective else "",
                    "Key questions: " + ", ".join(question_focus)
                    if question_focus
                    else "",
                    "Always respond in the requested output language.",
                ]
                if line
            )
        experts.register(
            name=item.get("name", "Expert"),
            focus=item.get("focus", ""),
            system_prompt=system_prompt,
        )
    return experts


def _parse_outline_sections(outline: str) -> list[str]:
    sections = []
    for line in outline.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                sections.append(title)
    return sections or ["Executive Summary"]


def _hydrate_engine(config: AppConfig, experts: ExpertManager) -> InferenceEngine:
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


@app.command()
def generate(
    config_path: Optional[Path] = typer.Option(None, help="Path to config.yaml"),
    output_path: Optional[Path] = typer.Option(
        Path("sci_storm_output.md"), help="Where to save the generated draft."
    ),
    experts_path: Optional[Path] = typer.Option(
        None, help="Path to experts.yaml defining expert personas."
    ),
):
    """Start an interactive Sci-STORM session."""
    config = load_config(config_path or "config.yaml")
    console.print(Panel.fit("Welcome to Sci-STORM 🚀\nHITL checkpoints will pause for your review."))

    # HITL 1: Research goal analysis
    document_style = Prompt.ask("Document style (any freeform description)", default="Report")
    goal = Prompt.ask("Target objective / research question")
    structural_requirements = Prompt.ask(
        "Structural requirements (e.g., IMRaD, bullet outline)", default="IMRaD"
    )
    output_language = Prompt.ask("Output language", default="Korean")
    outline_format_hint = Prompt.ask(
        "Outline format/template hint (e.g., numbered sections, IMRaD, PRD)", default="Numbered sections"
    )
    console.print(Panel(f"[bold]Goal[/bold]: {goal}\n[bold]Style[/bold]: {document_style}\n"
                        f"[bold]Structure[/bold]: {structural_requirements}\n"
                        f"[bold]Outline hint[/bold]: {outline_format_hint}\n"
                        f"[bold]Language[/bold]: {output_language}",
                        title="Review research setup"))
    if not Confirm.ask("Proceed with this goal?", default=True):
        console.print("Aborting: goal not confirmed.")
        raise typer.Exit(code=1)

    # HITL 2: Expert list confirmation
    if experts_path and experts_path.exists():
        experts = _load_experts_from_yaml(experts_path)
    else:
        experts = _build_default_experts(goal)
    console.print(Panel(Markdown(experts.describe_team()), title="Proposed expert roster"))
    if Confirm.ask("Would you like to add another expert?", default=False):
        name = Prompt.ask("Expert name")
        focus = Prompt.ask("Focus area", default="Custom focus")
        system_prompt = Prompt.ask("System prompt")
        experts.register(name=name, focus=focus, system_prompt=system_prompt)
    console.print(Panel(Markdown(experts.describe_team()), title="Finalized experts"))
    if not Confirm.ask("Confirm experts and continue?", default=True):
        console.print("Aborting: experts not confirmed.")
        raise typer.Exit(code=1)

    ctx = GenerationContext(
        goal=goal,
        document_style=document_style,
        structural_requirements=structural_requirements,
        outline_format_hint=outline_format_hint,
        output_language=output_language,
    )

    engine = _hydrate_engine(config, experts)

    # HITL 3: Knowledge graph / outline approval
    outline_response = engine.generate_outline(ctx)
    console.print(Panel(Markdown(outline_response.content), title="Draft outline"))
    if not Confirm.ask("Approve outline to continue?", default=True):
        console.print("Please rerun after refining the outline requirements.")
        raise typer.Exit(code=1)
    ctx.outline = outline_response.content

    # Collaborative expert dialogue with optional human feedback
    dialogue_rounds = int(Prompt.ask("How many expert dialogue rounds?", default="2"))
    dialogue_notes = []
    for round_idx in range(dialogue_rounds):
        human_note = Prompt.ask(
            f"[Round {round_idx+1}] Optional human guidance for experts", default=""
        )
        dialogue = engine.collaborative_dialogue(
            topic=goal, human_feedback=human_note, turns=1
        )
        dialogue_notes.extend(dialogue)
        console.print(Panel(f"Expert dialogue round {round_idx+1} in progress", title="Live outputs"))
        for line in dialogue:
            console.print(Markdown(line))
        console.print(Panel(Markdown("\n".join(dialogue)), title=f"Dialogue round {round_idx+1} (summary)"))
        if round_idx + 1 < dialogue_rounds and not Confirm.ask("Continue to next dialogue round?", default=True):
            break

    # HITL 4: Final draft review (section synthesis with evidence)
    evidence, tavily_result = engine.run_expert_round(goal)
    combined_notes = list(evidence.values()) + dialogue_notes
    human_final = Prompt.ask("Optional final human feedback before drafting", default="")
    if human_final:
        combined_notes.append(f"Human feedback: {human_final}")

    console.print(Panel("Tool usage report", title="External APIs"))
    if tavily_result.sources:
        console.print(f"Tavily search executed for query: {tavily_result.query}")
        for source in tavily_result.sources:
            console.print(f"- {source.title}: {source.url}")
    else:
        console.print(f"Tavily search skipped/failed: {tavily_result.error}")
    console.print("MCP execution: not invoked in this session.")

    sections = _parse_outline_sections(ctx.outline or "")
    section_outputs = []
    for section_title in sections:
        response = engine.synthesize_section(
            ctx=ctx, section_title=section_title, notes=combined_notes
        )
        section_outputs.append(f"## {section_title}\n\n{response.content}")
        console.print(Panel(Markdown(response.content), title=f"Draft: {section_title}"))

    if output_path:
        transcript = [
            "# Sci-STORM Session",
            "",
            f"## Goal",
            goal,
            "",
            "## Language",
            output_language,
            "",
            "## Outline",
            outline_response.content,
            "",
            "## Expert Dialogues",
            "\n".join(dialogue_notes) if dialogue_notes else "No dialogue rounds were recorded.",
            "",
            "## Tool Usage",
            "### Tavily Sources",
            "\n".join(
                f"- {source.title}: {source.url}"
                for source in tavily_result.sources
            )
            if tavily_result.sources
            else f"Tavily search skipped/failed: {tavily_result.error}",
            "### MCP",
            "Not invoked in this session.",
            "",
            "## Draft",
            "\n\n".join(section_outputs),
        ]
        output_path.write_text("\n".join(transcript), encoding="utf-8")
        console.print(
            f"Draft (outline + dialogue + full sections) saved to [bold]{output_path}[/bold]."
        )

    console.print("Review the draft above. Future steps will iterate section-by-section with the same workflow.")

    typer.echo("Sci-STORM session complete. You can extend this draft iteratively.")


if __name__ == "__main__":
    app()
