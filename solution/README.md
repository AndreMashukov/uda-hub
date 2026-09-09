# UDA-Hub — Universal Decision Agent

LangGraph multi-agent support system for the CultPass account. This folder is the
graded `solution/` tree. Do not import anything from outside `solution/`.

Python version: **3.11+** (developed on 3.11).

## What it does

- Accepts a natural-language support ticket (plus optional user id / email)
- Supervisor loads CultPass and UDA-Hub context
- Classifier labels the issue and applies routing rules
- Resolver retrieves CultPass knowledge with TF-IDF RAG and answers
- Escalation summarizes for a human (blocked accounts, refunds, low confidence)
- Short-term memory: LangGraph `thread_id` + `MemorySaver`
- Long-term memory: SQLite store with lexical recall of preferences/resolutions

## Setup

```bash
cd solution
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # optional; add OPENAI_API_KEY or VOCAREUM_API_KEY
python setup_external_db.py
python setup_core_db.py
```

You can also run the notebooks `01_external_db_setup.ipynb` and `02_core_db_setup.ipynb`.
They call the same setup functions.

An OpenAI/Vocareum key is optional. Without it, classification and answers use
documented heuristics plus RAG. With a key, the classifier and resolver call
`gpt-4o-mini`.

## Run

```bash
python 03_agentic_app.py
```

Or open `03_agentic_app.ipynb` and run all cells.

Example prompts:

- `user_id f556c0: How do I reserve an experience?`
- `user_id a4ab87: I can't log in` (blocked account → escalation)
- `user_id f556c0: I want a refund for last month`

Type `quit` to exit.

Optional MCP path (stdio servers wrapping the same DB tools):

```bash
USE_MCP=1 python -c "from agentic.tools.mcp_client import get_tools; print([t.name for t in get_tools(True)])"
```

The graph uses the shared Python ops by default so tests and local runs do not
depend on subprocess MCP. The MCP servers live in `agentic/tools/mcp/` and call
those same ops (absolute DB paths via `SOLUTION_ROOT`).

## Tests

```bash
cd solution
pytest -q
```

## Packages

See `requirements.txt`. Notable versions requested by the starter:

- langgraph>=0.5.4
- langchain>=0.3.27
- langchain-openai>=0.3.28
- fastmcp>=2.10.6
- langchain-mcp-adapters>=0.1.9
- sqlalchemy>=2.0.41
- pytest>=8.3.0 (added for the required test suite)

## Design

Architecture, routing rules, RAG, and a full code walkthrough:
`agentic/design/` (start with `CODE_WALKTHROUGH.md`).
