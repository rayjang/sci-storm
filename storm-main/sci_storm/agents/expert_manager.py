from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExpertProfile:
    name: str
    system_prompt: str
    focus: str = ""


class ExpertManager:
    """Tracks the expert roster and enables dynamic reconfiguration."""

    def __init__(self, experts: List[ExpertProfile] | None = None):
        self.experts: List[ExpertProfile] = experts or []

    def register(self, name: str, system_prompt: str, focus: str = ""):
        self.experts.append(ExpertProfile(name=name, system_prompt=system_prompt, focus=focus))

    def clear(self):
        self.experts = []

    def describe_team(self) -> str:
        descriptions = []
        for expert in self.experts:
            descriptions.append(
                f"- {expert.name}: {expert.focus or 'Generalist'}\n{expert.system_prompt}"
            )
        return "\n".join(descriptions)

    def suggest_outline_hooks(self) -> List[str]:
        """Inspired by Co-STORM's expert rotation, return focus strings."""
        return [expert.focus or expert.name for expert in self.experts]

