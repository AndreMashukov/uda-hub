# UDA-Hub code walkthrough

This document explains the **decision system**, not every helper file. It quotes the
code that actually routes tickets, retrieves knowledge, and talks to CultPass.

Related shorter notes: [ARCHITECTURE.md](ARCHITECTURE.md), [RAG.md](RAG.md).

---

## 1. What the system is doing

A customer support ticket is a blob of natural language plus optional metadata
(CultPass `user_id` like `f556c0`, email, channel). UDA-Hub must **decide how to
automate**:

1. Understand the ticket
2. Load the member and any existing UDA-Hub ticket
3. Classify the issue
4. Either **resolve** from the knowledge base or **escalate** to a human
5. Remember the session (short-term) and the outcome (long-term)

That is a graph, not a single chatbot. The starter file `workflow.py` used
`create_react_agent`. This project **does not**. Nodes are declared and wired
explicitly.

```
Incoming message
      │
      ▼
 Supervisor  ── load CultPass user + UDA-Hub ticket
      │
      ▼
 Classifier  ── issue_type, urgency, confidence, blocked?, refund?
      │
      ├── blocked / refund / low confidence ──► Escalation ──► END
      │
      └── otherwise ──► Resolver (RAG + account tools)
                              │
                              ├── still unsure ──► Escalation ──► END
                              └── answer ──► END
```

---

## 2. Shared state (`agentic/agents/state.py`)

Every node reads and writes the same `AgentState`. LangGraph merges node
outputs into this dict.

```python
class Classification(BaseModel):
    issue_type: IssueType = "unknown"
    urgency: Literal["low", "medium", "high"] = "medium"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    needs_refund: bool = False
    user_blocked: bool = False
    rationale: str = ""


class TicketContext(BaseModel):
    ticket_id: str = ""
    account_id: str = "cultpass"
    channel: str = "chat"
    external_user_id: str = ""
    user_email: str = ""
    user_name: str = ""
    raw_ticket: str = ""


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    context: dict[str, Any]
    classification: dict[str, Any]
    route: str
    retrieved: list[dict[str, Any]]
    resolution: str
    escalation_summary: str
    tool_notes: list[str]
```

**Why this shape**

| Field | Role |
| --- | --- |
| `messages` | Chat transcript. `add_messages` **appends** instead of replacing, so Supervisor + Classifier + Resolver all leave a trail. |
| `context` | Structured ticket/member identity. Nodes pass dicts; Pydantic models validate at the edges. |
| `classification` | Output of the Classifier; Resolver and Escalation read it. |
| `route` | The decision string (`resolver`, `escalation`, `end`) used by conditional edges. |
| `retrieved` | RAG hits, useful for debugging and tests. |

`Classification` is also the **structured-output schema** for the optional LLM
classifier (`with_structured_output(Classification)`), so the heuristic path and
the LLM path produce the same object.

---

## 3. The graph (`agentic/workflow.py`)

This is the file the spec told us to write from scratch.

```python
def route_after_classifier(state: AgentState) -> str:
    return "escalation" if state.get("route") == "escalation" else "resolver"


def route_after_resolver(state: AgentState) -> str:
    return "escalation" if state.get("route") == "escalation" else "end"


def build_orchestrator(checkpointer: MemorySaver | None = None):
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("resolver", resolver_node)
    graph.add_node("escalation", escalation_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "classifier")
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"resolver": "resolver", "escalation": "escalation"},
    )
    graph.add_conditional_edges(
        "resolver",
        route_after_resolver,
        {"escalation": "escalation", "end": END},
    )
    graph.add_edge("escalation", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


orchestrator = build_orchestrator()
```

**How to read it**

- `add_node` registers a Python function `state -> partial state`.
- `add_edge(START, "supervisor")` is always the first hop.
- Supervisor always goes to Classifier (no branch yet). Policy lives **after**
  classification, not inside a giant ReAct prompt.
- `route_after_classifier` looks at `state["route"]` that the Classifier wrote.
  The mapping dict tells LangGraph which node name that string means.
- Resolver is allowed a **second** branch: if RAG confidence is poor it sets
  `route = "escalation"` and the conditional edge sends the ticket on.
- `compile(checkpointer=MemorySaver())` is **short-term memory**. The same
  `thread_id` in `config["configurable"]["thread_id"]` reloads prior state.

`03_agentic_app.py` and `03_agentic_app.ipynb` import this `orchestrator`.

---

## 4. Supervisor — intake (`agentic/agents/supervisor.py`)

The Supervisor does not answer the customer. It **grounds** the graph.

### 4.1 Pull the latest human text

```python
def _latest_human(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""
```

The chat loop may prepend a `SystemMessage` with the thread id. Walking **backwards**
avoids treating that as the ticket.

### 4.2 Extract CultPass identity from free text

```python
def _extract_ids(text: str) -> tuple[str, str]:
    email_match = _EMAIL.search(text)
    email = email_match.group(0) if email_match else ""
    user_id = ""
    id_match = re.search(r"user[_ ]id[:\s]+([a-f0-9]{6})", text, re.I)
    if id_match:
        user_id = id_match.group(1)
    elif _USER_ID.search(text) and "ticket" not in text.lower():
        user_id = _USER_ID.search(text).group(0)
    return user_id, email
```

Demo members use 6-character hex ids (`f556c0`, `a4ab87`). The explicit
`user_id f556c0:` prefix is the reliable form. A bare hex token is a fallback.

### 4.3 Load databases before classifying

```python
    if context.external_user_id or context.user_email:
        lookup = lookup_user(user_id=context.external_user_id, email=context.user_email)
        ...
    if context.external_user_id or context.ticket_id:
        bundle = get_ticket_bundle(
            ticket_id=context.ticket_id,
            external_user_id=context.external_user_id,
        )
```

`lookup_user` hits **CultPass** (`data/external/cultpass.db`): name, email,
`is_blocked`. `get_ticket_bundle` hits **UDA-Hub** (`data/core/udahub.db`):
ticket metadata and prior messages.

The node returns:

```python
    return {
        "messages": [ack],
        "context": context.model_dump(),
        "tool_notes": notes,
    }
```

`ack` is an `AIMessage` so the transcript shows that intake happened. Downstream
nodes use `context`, not the ack text.

---

## 5. Classifier — decide the path (`agentic/agents/classifier.py`)

Two stages: **label**, then **route**. Labeling can be keywords or an LLM.
Routing is always deterministic Python. That is intentional: refunds and blocked
accounts must not depend on a model mood.

### 5.1 Keyword rules

Order matters. Refund is checked **before** billing so “charged twice, I want a
refund” does not land as a generic billing FAQ.

```python
_RULES: list[tuple[str, list[str]]] = [
    ("refund", ["refund", "money back", "charged twice", "duplicate charge"]),
    ("account_blocked", ["blocked", "ban", "suspended"]),
    ("login", ["log in", "login", "password", "sign in", "can't access", "cannot access"]),
    ("billing", ["billing", "payment", "card", "charged", "invoice"]),
    ("reservation", ["reserv", "booking", "qr", "event", "experience", "venue"]),
    ("subscription", ["subscription", "cancel", "pause", "quota", "plan", "included"]),
    ("catalog", ["catalog", "what's on", "available experience", "paddleboard", "museum"]),
]

def heuristic_classify(text: str) -> Classification:
    lowered = text.lower()
    for issue_type, keywords in _RULES:
        if any(keyword in lowered for keyword in keywords):
            urgency = "high" if issue_type in {"refund", "account_blocked", "login"} else "medium"
            return Classification(
                issue_type=issue_type,
                urgency=urgency,
                confidence=0.82,
                needs_refund=issue_type == "refund" or "refund" in lowered,
                rationale=f"Matched keywords for {issue_type}",
            )
    return Classification(issue_type="general", urgency="low", confidence=0.6, ...)
```

If `OPENAI_API_KEY` / `VOCAREUM_API_KEY` is set and `UDA_USE_HEURISTIC` is not
`1`, `_llm_classify` runs first with `with_structured_output(Classification)`.
On any failure it falls back to heuristics. Tests force the heuristic path.

### 5.2 Ground “blocked” in the database, not the wording

```python
        parsed = json.loads(lookup_user(user_id=user_id, email=email))
        if parsed.get("is_blocked"):
            classification.user_blocked = True
            classification.issue_type = "account_blocked"
            classification.urgency = "high"
```

Alice (`a4ab87`) can type a normal login question. CultPass still has
`is_blocked=true`, so the Classifier **overrides** the label. That is the
policy the KB article “Blocked Account Policy” describes: do not auto-resolve.

### 5.3 Routing function

```python
CONFIDENCE_RESOLVE = 0.55

def decide_route(classification: Classification) -> str:
    if classification.user_blocked or classification.issue_type == "account_blocked":
        return "escalation"
    if classification.needs_refund or classification.issue_type == "refund":
        return "escalation"
    if classification.confidence < CONFIDENCE_RESOLVE or classification.issue_type == "unknown":
        return "escalation"
    return "resolver"
```

First-pass login still goes to Resolver (password-reset article). Persistent
login is handled later: the login KB says “escalate if it keeps failing”, and
the Resolver drops confidence when the ticket contains `"persistent"`.

The node writes `classification` and `route` into state so the graph edges can
branch without calling the LLM again.

---

## 6. Resolver — retrieve and answer (`agentic/agents/resolver.py`)

Resolver is the “operational brain” for tickets that policy allows us to close.

### 6.1 Gather tools, then RAG

```python
    if user_id:
        extras.append(lookup_user(user_id=user_id))
        extras.append(get_subscription(user_id))
        extras.append(list_reservations(user_id))
        extras.append(recall(user_id, query=text, k=3))
    if classification.issue_type == "catalog":
        extras.append(list_experiences(query=text))

    articles = retrieve_knowledge(text, k=3)
    draft, confidence = _draft_from_articles(text, articles, extras[:4])
    polished = _llm_polish(text, draft, articles)
    reply = polished or draft
```

Account JSON is **context**, not the answer. The answer comes from the top
knowledge article’s “Suggested phrasing” block. Optional `_llm_polish` rewrites
that draft using the retrieved sources; it is forbidden to invent refunds or
unblocks.

### 6.2 Confidence from retrieval, not vibes

```python
    confidence = min(0.93, 0.55 + float(top.get("score") or 0) * 0.5)
    if "escalate to human" in top["content"].lower() and "persistent" in text.lower():
        confidence = 0.4
```

`score` is TF-IDF cosine similarity. If nothing retrieves, confidence is `0.2`.

```python
RESOLVE_THRESHOLD = 0.62

    if confidence < RESOLVE_THRESHOLD:
        return { ..., "route": "escalation" }
```

That is how Resolver **hands back** to Escalation without the Classifier
running again.

### 6.3 Side effects on success

```python
        update_ticket(..., status="resolved", main_issue_type=classification.issue_type, ...)
        remember(user_id, "resolution", f"{classification.issue_type}: {reply[:400]}")
```

UDA-Hub ticket metadata is updated so a later session sees `resolved`. Long-term
memory stores a short resolution string keyed by CultPass `user_id`.

---

## 7. Escalation — human packet (`agentic/agents/escalation.py`)

Escalation never pretends the bot solved the case. It builds a specialist brief.

```python
    if classification.needs_refund or classification.issue_type == "refund":
        refund_line = issue_refund(
            user_id=user_id,
            amount_cents=1000,
            reason=ticket_text[:200] or "customer refund request",
        )
```

`issue_refund` is the spec’s “optional internal tool”. It **records** a request;
if the member is blocked it returns `pending_review` instead of looking like
cash already moved.

Then the ticket is marked `escalated` and a memory row of kind `escalation` is
stored. The `AIMessage` the customer sees is that summary (issue type, urgency,
confidence, reason, original text, human hours).

---

## 8. RAG (`agentic/tools/rag.py`)

Corpus: `udahub.knowledge` rows (from `cultpass_articles.jsonl`, 16 articles).
If the core DB is empty, the JSONL file is the fallback so retrieval still works.

```python
def retrieve_knowledge(query: str, k: int = 3) -> list[dict[str, Any]]:
    articles = _load_articles()
    docs = [tokenize(f"{a['title']} {a['tags']} {a['content']}") for a in articles]
    df: Counter[str] = Counter()
    for tokens in docs:
        df.update(set(tokens))
    n = len(docs)
    idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    query_vec = _tfidf_vector(tokenize(query), idf)
    scored = []
    for article, tokens in zip(articles, docs):
        score = _cosine(query_vec, _tfidf_vector(tokens, idf))
        scored.append((score, article))
    scored.sort(key=lambda item: item[0], reverse=True)
    ...
```

**Mechanics**

1. Tokenize with `[a-z0-9]+` (lowercase).
2. Term frequency = count / doc length.
3. IDF = `log((N+1)/(df+1)) + 1` (smoothed so rare words still count and
   missing terms are safe).
4. Cosine similarity between the query vector and each article vector.
5. Drop non-positive scores; return top `k` with a `score` field.

This is lexical RAG: “password reset” ranks the login article; “reserve a spot”
ranks the reservation article. No embedding API, no large vector file to submit.

The Resolver uses the **title** of the top hit as the citation (`Source: ...`).

---

## 9. Memory: two stores, two lifetimes

### Short-term (same chat session)

: 5 - 
config = {"configurable": {"thread_id": ticket_id}}
result = agent.invoke({ "messages": [...] }, config=config)
```

`ticket_id` here is the **session key**, not necessarily the SQL ticket primary
key. Repeating `invoke` with the same `thread_id` reloads `messages` and
`context`. That is LangGraph’s checkpointer, not our SQLite.

### Long-term (across sessions)

```python
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,   -- preference | resolution | escalation
    text TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
```

`remember` inserts. `recall` filters by `user_id`, then ranks with token overlap
against the current ticket (same tokenizer as RAG). Resolver calls `recall`
before drafting so a returning member can get prior resolutions in `extras`.

---

## 10. Tools, paths, and MCP

### Why absolute paths

Notebooks and MCP child processes do not share a working directory. Every DB
path is resolved from `SOLUTION_ROOT` (or `CULTPASS_DB` / `UDAHUB_DB` /
`MEMORY_DB` env vars) in `agentic/tools/paths.py`.

### One implementation, two facades

Business logic is plain functions in `cultpass_ops.py` / `udahub_ops.py`.
Example: look up a member without caring who called you.

```python
def lookup_user(user_id: str = "", email: str = "") -> str:
    session = _session_factory()
    try:
        query = session.query(cultpass.User)
        if user_id:
            user = query.filter_by(user_id=user_id).first()
        elif email:
            user = query.filter_by(email=email).first()
        ...
        return json.dumps(payload, default=str)
    finally:
        session.close()
```

JSON strings are the tool contract (MCP and LangChain both like strings).

**Facade 1 — graph nodes** import these functions directly (tests stay
subprocess-free).

**Facade 2 — FastMCP stdio** wraps the same functions:

```python
mcp = FastMCP("cultpass")

@mcp.tool()
def lookup_user(user_id: str = "", email: str = "") -> str:
    """Look up a CultPass member by user_id or email."""
    return cultpass_ops.lookup_user(user_id=user_id, email=email)
```

`agentic/tools/mcp/udahub_server.py` does the same for tickets, RAG, and memory.

**Facade 3 — optional LangChain MCP client**

```python
client = MultiServerMCPClient({
    "cultpass": {
        "command": sys.executable,
        "args": [_server_path("cultpass_server.py")],
        "transport": "stdio",
        "env": {**os.environ, "SOLUTION_ROOT": str(solution_root())},
    },
    ...
})
return asyncio.run(client.get_tools())
```

`USE_MCP=1` loads tools from those servers; otherwise `get_local_tools()` wraps
the ops with `StructuredTool`. Agents currently call ops in-process so pytest
does not spawn MCP. The servers exist so DB access is not glued to
`os.getcwd()`.

---

## 11. Entry points

### CLI (`03_agentic_app.py`)

```python
def main() -> None:
    ensure_databases()
    thread_id = os.environ.get("UDA_THREAD_ID", "demo-ticket")
    chat_interface(orchestrator, thread_id)
```

If the SQLite files are missing, it runs `setup_external_db` then
`setup_core_db` (same functions the notebooks call).

### Chat loop (`utils.py`)

Each turn is one `orchestrator.invoke`. The printed assistant line is
**the last message**, which is Resolver’s answer or Escalation’s summary.
Classification is printed in parentheses so you can see the decision.

### Data seed

| Script / notebook | Database | Contents |
| --- | --- | --- |
| `setup_external_db.py` / `01_*.ipynb` | `data/external/cultpass.db` | Users, subscriptions, experiences, reservations |
| `setup_core_db.py` / `02_*.ipynb` | `data/core/udahub.db` | CultPass account, ≥14 knowledge articles, sample ticket |

Subscriptions are **deterministic** (not `random.choice`) so tests know Bob is
active and Alice is blocked.

---

## 12. Worked examples (what the code does on real prompts)

Assume DBs are seeded. `thread_id` is any string.

### A. Bob reserves — Resolver

Prompt: `user_id f556c0: How do I reserve a spot for a CultPass experience?`

1. Supervisor extracts `f556c0`, loads Bob (`is_blocked=false`).
2. Classifier matches `"reserv"` → `issue_type=reservation`, confidence `0.82`
   → `decide_route` → `resolver`.
3. Resolver RAG ranks “How to Reserve a Spot for an Event”.
4. Confidence from the TF-IDF score is above `0.62`.
5. Ticket (if any) marked `resolved`; memory row `kind=resolution`.
6. Customer sees the article’s suggested phrasing plus account JSON.

### B. Alice cannot log in — Escalation

Prompt: `user_id a4ab87: I can't log in to my Cultpass account.`

1. Classifier would have said `login`, but `lookup_user` shows blocked.
2. Override to `account_blocked` → `decide_route` → `escalation`.
3. Resolver never runs. Refund tool is not called.
4. Customer sees a specialist summary; ticket status `escalated`.

### C. Refund — Escalation + refund tool

Prompt: `user_id f556c0: I want a refund, you charged me twice`

1. `_RULES` hits `refund` first.
2. `needs_refund=True` → Escalation.
3. `issue_refund(f556c0, 1000, ...)` is recorded on the packet.

---

## 13. File map (what to open first)

| File | Read it for |
| --- | --- |
| `agentic/workflow.py` | Graph topology |
| `agentic/agents/state.py` | Data that flows between nodes |
| `agentic/agents/classifier.py` | Decision policy |
| `agentic/agents/resolver.py` | RAG + tools + second-chance escalate |
| `agentic/agents/escalation.py` | Human handoff + refund tool |
| `agentic/agents/supervisor.py` | Identity and DB intake |
| `agentic/tools/rag.py` | How articles are ranked |
| `agentic/tools/memory.py` | Long-term store |
| `agentic/tools/cultpass_ops.py` | CultPass actions |
| `agentic/tools/mcp/*.py` | MCP wrappers of those actions |
| `tests/test_udahub.py` | Executable specification of the paths above |

Everything a reviewer needs is under `solution/`. Do not import `examples/` or
the repo root from this package.
