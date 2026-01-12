from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import requests


@dataclass
class TavilySource:
    title: str
    url: str
    content: str


@dataclass
class TavilyResult:
    query: str
    sources: List[TavilySource]
    error: Optional[str] = None


class TavilySearchClient:
    """Thin wrapper around the Tavily HTTP API."""

    def __init__(self, api_key: Optional[str], max_results: int = 5):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results

    def search(self, query: str) -> TavilyResult:
        if not self.api_key:
            return TavilyResult(
                query=query,
                sources=[],
                error="Tavily API key not configured; skipping live search.",
            )

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "max_results": self.max_results},
                headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
                timeout=30,
            )
            if not response.ok:
                return TavilyResult(
                    query=query,
                    sources=[],
                    error=f"Tavily search failed: {response.text}",
                )

            data = response.json()
            results = data.get("results", [])
            sources = []
            for item in results:
                sources.append(
                    TavilySource(
                        title=item.get("title", "untitled"),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                    )
                )
            if not sources:
                return TavilyResult(
                    query=query,
                    sources=[],
                    error="Tavily search returned no results.",
                )
            return TavilyResult(query=query, sources=sources)
        except Exception as exc:  # noqa: BLE001
            return TavilyResult(
                query=query,
                sources=[],
                error=f"Tavily search unreachable; continuing without live search. ({exc})",
            )
