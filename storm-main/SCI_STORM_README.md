# Sci-STORM Runbook

This guide explains how to stand up and run the **Sci-STORM** CLI for collaborative scientific document generation. It covers prerequisites, configuration, API key setup, KISTI MCP startup, and the interactive workflow.

## 1) Prerequisites

- Python 3.10+ and a POSIX shell
- Recommended: create and activate a virtual environment
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

## 2) Prepare configuration

1. Copy the example config:
   ```bash
   cp config.example.yaml config.yaml
   ```
2. Pick your inference backend in `config.yaml`:
   - `backend.provider`: `ollama` (default) or `vllm`
   - `backend.model`: e.g., `gpt-oss:20b` (Ollama form), `openai/gpt-oss-120b`, `naver-hyperclovax/HyperCLOVAX-SEED-Think-32B`
   - `backend.base_url`: `http://localhost:11434` (Ollama) or your vLLM endpoint

## 3) Configure API keys

- Tavily search: set an API key via env var or directly in `config.yaml`
  ```bash
  export TAVILY_API_KEY="your-tavily-key"
  ```
- If your vLLM endpoint requires auth, add `backend.api_key` in `config.yaml`.

## 4) Start the KISTI MCP server

1. Clone and install the MCP server (https://github.com/ansua79/kisti-mcp):
   ```bash
   git clone https://github.com/ansua79/kisti-mcp.git
   cd kisti-mcp
   pip install -e .
   ```
2. Start the server (default matches `config.example.yaml`):
   ```bash
   kisti-mcp serve --host 0.0.0.0 --port 8000
   ```
   Sci-STORM will probe `/health` and retry before raising a connection error.

## 5) Optional: Seed local RAG

Place any `.md` or `.txt` references under `./data/`. They are auto-ingested into the placeholder local index on startup.

## 6) Run the interactive CLI

From the repo root:
```bash
python -m sci_storm.pipeline.cli generate --config-path config.yaml --output-path sci_storm_output.md
```

The CLI will guide you through human-in-the-loop checkpoints:
1. Confirm document style (freeform), research goal, structural requirements, and outline format hint.
2. Review/extend the expert roster.
3. Approve the collaboratively generated outline.
4. Let experts debate across multiple dialogue rounds, inject human feedback, and synthesize a draft section from expert/Tavily/RAG/MCP evidence.

You can rerun the command anytime after editing `config.yaml` or adding experts. Future iterations can extend section-by-section drafting using the same engine wiring.

## 7) Troubleshooting

- **Backend errors**: ensure the model endpoint in `config.yaml` is reachable; adjust `max_retries` or `retry_backoff` if the local backend is slow to start.
- **MCP connection errors**: verify the server is running and reachable at `mcp.server_url`; increase `mcp.max_retries` if startup is slow.
- **Tavily missing**: set `TAVILY_API_KEY` or expect the CLI to skip live search.
