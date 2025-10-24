# Chatbot

A local chatbot project that combines streaming responses, threading, and persistent memory saved to a local database. The project integrates a set of small tools (calculator, DuckDuckGo search, stock-market API, etc.) which the assistant can call to extend capabilities.

This repository contains multiple frontends (CLI/streaming/Streamlit) and backend components that manage memory, tool access, and external API calls.

## Key features

- Streaming responses: stream partial outputs back to the client for lower-latency UX (see `streaming_frontend.py` and `streamlit_frontend.py`).
- Threading: support for concurrent conversations/requests using a threading-aware frontend (see `threading_frontend.py` and `chatbot_threading_frontend.py`).
- Persistent memory: conversation state and metadata are persisted to a local database so context survives restarts. See `database_frontend.py`, `langgraph_database.py`, and `langgraph_backend_initial.py`.
- Tools integration: small utility tools are available and called when appropriate including:
  - Calculator (local arithmetic/eval)
  - DuckDuckGo search (web search)
  - Stock market API integration (for quotes/history)
  - Other helpers exposed by `tools_backend.py` / `tools_frontend.py`

## Repository layout (important files)

- `streaming_frontend.py` — streaming CLI-style frontend that yields partial responses.
- `streamlit_frontend.py` — Streamlit UI frontend for interactive usage.
- `threading_frontend.py` & `chatbot_threading_frontend.py` — thread-capable frontends and examples for concurrent sessions.
- `database_frontend.py` — interfaces for storing/retrieving persistent memory.
- `langgraph_backend_initial.py` & `langgraph_database.py` — backend logic and DB helpers for the memory layer.
- `tools_backend.py` & `tools_frontend.py` — tool wrappers and the tool registry used by the assistant.

If you add or change file names, update this section to keep the README accurate.

## Quick setup

1. Create and activate a virtual environment (this repo includes `myenv/` but you can create a new one):

```powershell
# create virtualenv (optional if using provided myenv)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install requirements
pip install -r requirements.txt
```

2. Configure API keys and environment variables.

- If you use an external stock market API, set the key (example):

```powershell
$env:STOCK_API_KEY = "your_stock_api_key_here"
```

- If any other external tools or search integration require keys, set them similarly.

Store persistent DB files in the `data/` folder (the code may already create it). By default this project expects a local file-based DB — check `langgraph_database.py` for exact DB path and adjust as needed.

## Run examples

Streaming frontend (simple):

```powershell
python streaming_frontend.py
```

Threading / concurrency demo:

```powershell
python chatbot_threading_frontend.py
```

Streamlit UI (if you prefer a browser UI):

```powershell
streamlit run streamlit_frontend.py
```

Notes:
- Some frontends accept CLI args or configuration files — check the top of each file for available flags.

## How the memory works

- Conversations (or summarized embeddings) are persisted to the DB via the `database_frontend.py` and `langgraph_database.py` modules. The assistant can re-load prior context on startup and use it to answer queries with contextual awareness.
- Memory typically stores: conversation turns, optional vector embeddings (if enabled), and metadata such as timestamps or session IDs.

## Tools and how they're used

- Tools are registered and invoked by the assistant when appropriate. The project splits tool logic between `tools_backend.py` (implementations) and `tools_frontend.py` (wrappers/registrations).
- Example tools included or referenced:
  - Calculator: evaluate arithmetic or lightweight expressions locally.
  - DuckDuckGo search: quick web searches for retrieving facts.
  - Stock market API: fetch quotes, historical prices, or summaries.

When the assistant decides to call a tool, it will run the tool, capture the structured result, and include it in the response flow (often combined with streaming so the user sees updates quickly).

## Configuration and tuning

- Logging: enable or increase logging to debug tool calls, DB operations, or threading issues.
- Concurrency: adjust thread pool / worker counts in `threading_frontend.py` if you hit CPU or I/O constraints.
- Database: review `langgraph_database.py` for connection strings, file paths, and retention policies. Back up the `data/` directory before large changes.

## Security and API keys

- Never commit API keys to source control. Use environment variables or a local `.env` file (and add it to `.gitignore`).
- Validate and sanitize any external input used with tools that run code (like the calculator) to avoid injection risks.

## Tests and validation

This repository currently focuses on demos and examples. Adding unit tests for tools, DB operations, and the threading layer is recommended. Suggested next steps:

- Add a small pytest suite that verifies:
  - DB read/write roundtrip
  - Tool wrapper outputs (mock external APIs)
  - Thread-safety for concurrent conversation writes

## Troubleshooting

- If streaming responses hang: check network calls in tool implementations and ensure they have timeouts.
- If memory persistence fails: verify write permissions for the `data/` folder and check DB connection settings in `langgraph_database.py`.

## Next steps I can do for you

- Create a `.env.example` with suggested environment variables.
- Add a minimal pytest file for DB read/write validation.
- Wire a small Streamlit demo showing memory loaded into the UI.

Tell me which (if any) you'd like me to implement next, and I'll add it.
