from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ..agents.expert_manager import ExpertManager
from ..tools.kisti_mcp import KISTIMCPClient
from ..tools.rag import LocalRAGClient
from ..tools.tavily import TavilySearchClient
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
            f"Available Experts:\n{expert_block}\n"
            "Return a markdown outline with numbered sections and a short rationale "
            "for each section."
        )
        return self.backend.generate(_messages_from_prompt(system_prompt, user_prompt))

    def synthesize_section(
        self, ctx: GenerationContext, section_title: str, notes: Iterable[str]
    ) -> BackendResponse:
        sources = "\n".join(notes)
        system_prompt = (
            "You are a lead author merging expert findings. Ground the response in "
            "the provided notes and cite code execution outputs when available."
        )
        user_prompt = (
            f"Section: {section_title}\nGoal: {ctx.goal}\n"
            f"Document Style: {ctx.document_style}\n"
            f"Outline:\n{ctx.outline or 'N/A'}\n"
            f"Collected Evidence:\n{sources}\n"
            "Write a concise draft section in markdown. Use bullet points for "
            "experimental results and keep terminology precise."
        )
        return self.backend.generate(_messages_from_prompt(system_prompt, user_prompt))

    def run_expert_round(self, query: str) -> Dict[str, str]:
        """Fan-out a question to the configured experts, Tavily, and the local RAG."""
        evidence: Dict[str, str] = {}
        for expert in self.expert_manager.experts:
            response = self.backend.generate(
                _messages_from_prompt(
                    expert.system_prompt,
                    f"As {expert.name}, analyze: {query}. Return key facts.",
                )
            )
            evidence[expert.name] = response.content

        evidence["tavily"] = self.search_client.search(query)
        rag_hits = self.rag_client.query(query)
        evidence["rag"] = "\n".join(rag_hits)
        return evidence

    def execute_experiment(self, hypothesis: str, code: str) -> str:
        """Use the KISTI MCP to run code and return the interpreted result."""
        run_info = self.mcp_client.run_experiment(hypothesis, code)
        return self.mcp_client.interpret_result(run_info)

