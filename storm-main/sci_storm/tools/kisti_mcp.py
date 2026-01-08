from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Dict, Optional

import requests

from ..config import MCPConfig


class KISTIMCPClient:
    """Adapter for the KISTI MCP server used for experimental execution."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.session = requests.Session()
        self._process: Optional[subprocess.Popen] = None

    def _retry(self, fn):
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                time.sleep(self.config.retry_backoff * attempt)
        if last_error:
            raise last_error

    def _handshake(self) -> bool:
        def _call():
            response = self.session.get(
                f"{self.config.server_url}{self.config.handshake_path}",
                timeout=5,
            )
            response.raise_for_status()
            return True

        try:
            return self._retry(_call)
        except Exception:
            return False

    def ensure_server(self):
        """Start the MCP server if it is not reachable."""
        if self._handshake():
            return

        if self._process is None and self.config.startup_command:
            self._process = subprocess.Popen(
                self.config.startup_command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            time.sleep(1.0)

        if not self._handshake():
            raise ConnectionError("Unable to reach KISTI MCP server after startup.")

    def run_experiment(self, hypothesis: str, code: str) -> Dict[str, Any]:
        """Submit an experiment to the MCP server."""
        self.ensure_server()

        payload = {"hypothesis": hypothesis, "code": code}

        def _call():
            response = self.session.post(
                f"{self.config.server_url}/execute",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if response.status_code >= 500:
                raise RuntimeError(response.text)
            response.raise_for_status()
            return response.json()

        return self._retry(_call)

    def interpret_result(self, result: Dict[str, Any]) -> str:
        """Convert MCP execution output into a concise textual report."""
        logs = result.get("logs") or result.get("stderr") or ""
        output = result.get("stdout") or result.get("result") or ""
        summary = result.get("summary") or ""

        parts = [
            "### MCP Execution Summary",
            summary or "The MCP server did not return an explicit summary.",
            "",
            "### Output",
            output if isinstance(output, str) else json.dumps(output, indent=2),
        ]
        if logs:
            parts.extend(["", "### Logs", logs])
        return "\n".join(parts)

