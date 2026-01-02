from __future__ import annotations

import os
from typing import Optional

import requests


class TavilySearchClient:
    """Thin wrapper around the Tavily HTTP API."""

    def __init__(self, api_key: Optional[str], max_results: int = 5):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results

    def search(self, query: str) -> str:
        if not self.api_key:
            return "Tavily API key not configured; skipping live search."

        response = requests.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": self.max_results},
            headers={"Content-Type": "application/json", "X-API-Key": self.api_key},
            timeout=30,
        )
        if not response.ok:
            return f"Tavily search failed: {response.text}"

        data = response.json()
        results = data.get("results", [])
        formatted = []
        for item in results:
            title = item.get("title", "untitled")
            url = item.get("url", "")
            content = item.get("content", "")
            formatted.append(f"- {title} ({url})\n{content}")
        return "\n".join(formatted)

