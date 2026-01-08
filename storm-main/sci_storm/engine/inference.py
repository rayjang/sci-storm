from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from ..agents.expert_manager import ExpertManager
from ..tools.kisti_mcp import KISTIMCPClient
from ..tools.rag import LocalRAGClient
from ..tools.tavily import TavilyResult, TavilySearchClient
from .backend import BackendAdapter, BackendResponse


def _messages_from_prompt(system_prompt: str, user_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


@dataclass
class GenerationContext:
    goal: str
    document_style: str
    structural_requirements: str
    outline_format_hint: Optional[str] = None
    output_language: Optional[str] = None
    outline: Optional[str] = None
    shared_notebook_uri: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)


class InferenceEngine:
    """
    Thin orchestration layer that mirrors Co-STORM's collaborative reasoning steps.

    The engine keeps prompts modular so that the ExpertManager can stitch together
    multi-expert perspectives, similar to the round-table logic in
    `CollaborativeStorm.DiscoveryManager`.
    """

    def __init__(
        self,
        backend: BackendAdapter,
        expert_manager: ExpertManager,
        search_client: TavilySearchClient,
        rag_client: LocalRAGClient,
        mcp_client: KISTIMCPClient,
    ):
        self.backend = backend
        self.expert_manager = expert_manager
        self.search_client = search_client
        self.rag_client = rag_client
        self.mcp_client = mcp_client

    def _safe_generate(
        self, messages, fallback: str = "Backend unavailable; skipping."
    ) -> BackendResponse:
        try:
            return self.backend.generate(messages)
        except Exception as exc:  # noqa: BLE001
            return BackendResponse(
                content=f"{fallback} Error: {exc}",
                raw={"error": str(exc)},
            )

    def generate_outline(self, ctx: GenerationContext) -> BackendResponse:
        expert_block = self.expert_manager.describe_team()
        system_prompt = (
            "You are coordinating a collaborative outline session inspired by "
            "Co-STORM's warm-start mind-map builder. Use the expert roster to "
            "propose a draft outline that a human will review."
        )
        user_prompt = (
            f"Goal: {ctx.goal}\n"
            f"Document Style: {ctx.document_style}\n"
            f"Structural Requirements: {ctx.structural_requirements}\n"
            f"Outline Format Hint: {ctx.outline_format_hint or 'None provided'}\n"
            f"Available Experts:\n{expert_block}\n"
            "Return a markdown outline with numbered sections and a short rationale "
            "for each section."
        )
        return self._safe_generate(
            _messages_from_prompt(system_prompt, user_prompt),
            fallback="Outline generation failed.",
        )

    def synthesize_section(
        self, ctx: GenerationContext, section_title: str, notes: Iterable[str]
    ) -> BackendResponse:
        sources = "\n".join(notes)
        language = ctx.output_language or "the requested language"
        system_prompt = (
            "You are a lead author merging expert findings. Ground the response in "
            "the provided notes and cite code execution outputs when available. "
            "Do not invent sources."
        )
        user_prompt = (
            f"Section: {section_title}\nGoal: {ctx.goal}\n"
            f"Document Style: {ctx.document_style}\n"
            f"Output Language: {language}\n"
            f"Outline:\n{ctx.outline or 'N/A'}\n"
            f"Collected Evidence:\n{sources}\n"
            "Write a cohesive draft section in markdown that follows the outline order "
            "and uses a single consistent language throughout. Use bullet points for "
            "experimental results and keep terminology precise. Only cite URLs that "
            "appear in the evidence above."
        )
        return self._safe_generate(
            _messages_from_prompt(system_prompt, user_prompt),
            fallback="Section synthesis failed.",
        )

    def run_expert_round(self, query: str) -> Tuple[Dict[str, str], TavilyResult]:
        """Fan-out a question to the configured experts, Tavily, and the local RAG."""
        evidence: Dict[str, str] = {}
        for expert in self.expert_manager.experts:
            response = self._safe_generate(
                _messages_from_prompt(
                    expert.system_prompt,
                    f"As {expert.name}, analyze: {query}. Return key facts.",
                ),
                fallback=f"{expert.name} unavailable.",
            )
            evidence[expert.name] = response.content

        tavily_result = self.search_client.search(query)
        if tavily_result.sources:
            evidence["tavily"] = "\n".join(
                f"- {item.title} ({item.url})\n{item.content}"
                for item in tavily_result.sources
            )
        else:
            evidence["tavily"] = tavily_result.error or "Tavily search returned no results."
        rag_hits = self.rag_client.query(query)
        evidence["rag"] = "\n".join(rag_hits) if rag_hits else "Local RAG returned no matches."
        return evidence, tavily_result

    def collaborative_dialogue(
        self,
        topic: str,
        human_feedback: str = "",
        turns: int = 2,
    ) -> List[str]:
        """
        Run a round-table style dialogue between experts, optionally seeded with human feedback.
        """
        history: List[str] = []
        seeded_prompt = (
            f"Topic: {topic}\nHuman feedback: {human_feedback or 'None provided.'}\n"
            "Consider prior turns to refine or challenge earlier points. Keep replies concise."
        )
        last_message = seeded_prompt

        for _ in range(turns):
            for expert in self.expert_manager.experts:
                prompt = (
                    f"{expert.system_prompt}\n\n"
                    f"Previous discussion:\n{last_message}\n\n"
                    f"As {expert.name}, add, refine, or correct the discussion."
                )
                response = self._safe_generate(
                    [{"role": "user", "content": prompt}],
                    fallback=f"{expert.name} response unavailable.",
                )
                turn_text = f"{expert.name}: {response.content}"
                history.append(turn_text)
                last_message = turn_text
        return history

    def execute_experiment(self, hypothesis: str, code: str) -> str:
        """Use the KISTI MCP to run code and return the interpreted result."""
        run_info = self.mcp_client.run_experiment(hypothesis, code)
        return self.mcp_client.interpret_result(run_info)
