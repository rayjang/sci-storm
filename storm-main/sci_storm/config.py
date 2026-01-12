from __future__ import annotations

import os
from dataclasses import dataclass, field  # Add 'field' here
from pathlib import Path
from typing import Literal, Optional

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class BackendConfig:
    """Configuration for model backends."""

    provider: Literal["ollama", "vllm"] = "ollama"
    model: str = "gpt-oss:20b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    request_timeout: int = 60
    max_retries: int = 3
    retry_backoff: float = 2.0


@dataclass
class TavilyConfig:
    api_key: Optional[str] = None
    max_results: int = 5


@dataclass
class RagConfig:
    provider: Literal["chromadb", "faiss"] = "chromadb"
    persist_directory: Path = Path("./data/index")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class MCPConfig:
    server_url: str = "http://localhost:8000"
    startup_command: str = "kisti-mcp serve --host 0.0.0.0 --port 8000"
    handshake_path: str = "/health"
    max_retries: int = 3
    retry_backoff: float = 2.0


@dataclass
class AppConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return os.path.expandvars(value)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load application configuration, falling back to defaults."""

    config_path = Path(path)
    raw = _load_yaml(config_path)

    backend = raw.get("backend", {})
    tavily = raw.get("tavily", {})
    rag = raw.get("rag", {})
    mcp = raw.get("mcp", {})

    backend_config = BackendConfig(
        provider=backend.get("provider", BackendConfig.provider),
        model=backend.get("model", BackendConfig.model),
        base_url=_resolve_env(backend.get("base_url", BackendConfig.base_url)),
        api_key=_resolve_env(backend.get("api_key", BackendConfig.api_key)),
        request_timeout=backend.get(
            "request_timeout", BackendConfig.request_timeout
        ),
        max_retries=backend.get("max_retries", BackendConfig.max_retries),
        retry_backoff=backend.get("retry_backoff", BackendConfig.retry_backoff),
    )

    tavily_config = TavilyConfig(
        api_key=_resolve_env(tavily.get("api_key", TavilyConfig.api_key)),
        max_results=tavily.get("max_results", TavilyConfig.max_results),
    )

    rag_config = RagConfig(
        provider=rag.get("provider", RagConfig.provider),
        persist_directory=Path(
            rag.get("persist_directory", RagConfig.persist_directory)
        ),
        embedding_model=rag.get("embedding_model", RagConfig.embedding_model),
    )

    mcp_config = MCPConfig(
        server_url=_resolve_env(mcp.get("server_url", MCPConfig.server_url)),
        startup_command=mcp.get("startup_command", MCPConfig.startup_command),
        handshake_path=mcp.get("handshake_path", MCPConfig.handshake_path),
        max_retries=mcp.get("max_retries", MCPConfig.max_retries),
        retry_backoff=mcp.get("retry_backoff", MCPConfig.retry_backoff),
    )

    return AppConfig(
        backend=backend_config,
        tavily=tavily_config,
        rag=rag_config,
        mcp=mcp_config,
    )
