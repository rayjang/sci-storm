from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

import requests
from requests import RequestException

from ..config import BackendConfig


class BackendError(RuntimeError):
    """Raised when a backend returns an error response."""


class BackendRetryableError(RuntimeError):
    """Raised when an operation may be retried."""


@dataclass
class BackendResponse:
    content: str
    raw: Dict[str, Any]


class BaseBackendClient:
    def __init__(self, config: BackendConfig):
        self.config = config

    def _wrap_result(self, fn) -> "BackendResponse":
        try:
            return self._retry_loop(fn)
        except Exception as exc:  # noqa: BLE001
            return BackendResponse(
                content=f"[Backend error] {exc}",
                raw={"error": str(exc)},
            )

    @property
    def headers(self) -> Dict[str, str]:
        if self.config.api_key:
            return {"Authorization": f"Bearer {self.config.api_key}"}
        return {}

    def _retry_loop(self, fn):
        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return fn()
            except BackendRetryableError as exc:  # pragma: no cover - loop only
                last_error = exc
                time.sleep(self.config.retry_backoff * attempt)
            except RequestException as exc:
                last_error = exc
                time.sleep(self.config.retry_backoff * attempt)
        if last_error:
            raise last_error

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> BackendResponse:
        raise NotImplementedError


class OllamaClient(BaseBackendClient):
    """Lightweight adapter for a local Ollama daemon."""

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> BackendResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        payload.update(kwargs)

        def _request():
            response = requests.post(
                f"{self.config.base_url}/api/chat",
                data=json.dumps(payload),
                headers=self.headers,
                timeout=self.config.request_timeout,
            )
            if response.status_code >= 500:
                raise BackendRetryableError(response.text)
            if not response.ok:
                raise BackendError(response.text)
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return BackendResponse(content=content, raw=data)

        return self._wrap_result(_request)


class VLLMClient(BaseBackendClient):
    """OpenAI-compatible client for vLLM deployments."""

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> BackendResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
        }
        payload.update(kwargs)

        def _request():
            response = requests.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=payload,
                headers={
                    **self.headers,
                    "Content-Type": "application/json",
                },
                timeout=self.config.request_timeout,
            )
            if response.status_code >= 500:
                raise BackendRetryableError(response.text)
            if not response.ok:
                raise BackendError(response.text)
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get(
                "content", ""
            )
            return BackendResponse(content=content, raw=data)

        return self._wrap_result(_request)


class BackendAdapter:
    """Unified API across Ollama and vLLM backends."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.clients: Dict[Literal["ollama", "vllm"], BaseBackendClient] = {
            "ollama": OllamaClient(config),
            "vllm": VLLMClient(config),
        }
        if config.provider not in self.clients:
            raise ValueError(f"Unsupported provider: {config.provider}")
        self.active_provider: Literal["ollama", "vllm"] = config.provider

    def switch(self, provider: Literal["ollama", "vllm"], model: Optional[str] = None):
        if provider not in self.clients:
            raise ValueError(f"Unsupported provider: {provider}")
        self.active_provider = provider
        if model:
            self.config.model = model

    @property
    def client(self) -> BaseBackendClient:
        return self.clients[self.active_provider]

    def generate(
        self, messages: Iterable[Dict[str, str]], temperature: float = 0.7, **kwargs
    ) -> BackendResponse:
        payload_messages = list(messages)
        return self.client.generate(payload_messages, temperature=temperature, **kwargs)
