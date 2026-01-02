from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from ..agents import ExpertManager, ExpertProfile
from ..config import AppConfig, load_config
from ..engine import BackendAdapter, InferenceEngine
from ..engine.inference import GenerationContext
from ..tools import KISTIMCPClient, LocalRAGClient, TavilySearchClient


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
    return experts


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
def generate(config_path: Optional[Path] = typer.Option(None, help="Path to config.yaml")):
    """Start an interactive Sci-STORM session."""
    config = load_config(config_path or "config.yaml")
    console.print(Panel.fit("Welcome to Sci-STORM 🚀\nHITL checkpoints will pause for your review."))

    # HITL 1: Research goal analysis
    document_style = Prompt.ask(
        "Document style", choices=["Paper", "Report", "Blog"], default="Report"
    )
    goal = Prompt.ask("Target objective / research question")
    structural_requirements = Prompt.ask(
        "Structural requirements (e.g., IMRaD, bullet outline)", default="IMRaD"
    )
    console.print(Panel(f"[bold]Goal[/bold]: {goal}\n[bold]Style[/bold]: {document_style}\n"
                        f"[bold]Structure[/bold]: {structural_requirements}",
                        title="Review research setup"))
    if not Confirm.ask("Proceed with this goal?", default=True):
        console.print("Aborting: goal not confirmed.")
        raise typer.Exit(code=1)

    # HITL 2: Expert list confirmation
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
    )

    engine = _hydrate_engine(config, experts)

    # HITL 3: Knowledge graph / outline approval
    outline_response = engine.generate_outline(ctx)
    console.print(Panel(Markdown(outline_response.content), title="Draft outline"))
    if not Confirm.ask("Approve outline to continue?", default=True):
        console.print("Please rerun after refining the outline requirements.")
        raise typer.Exit(code=1)
    ctx.outline = outline_response.content

    # HITL 4: Final draft review (placeholder section synthesis)
    evidence = engine.run_expert_round(goal)
    section_response = engine.synthesize_section(
        ctx=ctx, section_title="Executive Summary", notes=evidence.values()
    )
    console.print(Panel(Markdown(section_response.content), title="Draft section"))
    console.print("Review the draft above. Future steps will iterate section-by-section.")

    typer.echo("Sci-STORM session complete. You can extend this draft iteratively.")


if __name__ == "__main__":
    app()

