# LangGraph Supervisor Skill

A **generic**, config-driven multi-agent supervisor pattern using LangGraph.

The supervisor **analyses the full user input**, identifies **all specialist agents
needed**, extracts a **focused sub-query for each agent**, then calls each one
**in sequence** — each agent answers only its own relevant part.

---

## Quick Start

```python
from supervisor_skill import AgentConfig, run_supervisor
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """Does something useful."""
    return f"Result for {query}"

agents = [
    AgentConfig(
        name="research",
        route_key="research",
        description="research questions, fact lookup, information retrieval",
        system_prompt="You are a research agent. Use tools to answer questions.",
        tools=[my_tool],
    ),
    AgentConfig(
        name="finance",
        route_key="finance",
        description="financial questions, budgets, investments, banking",
        system_prompt="You are a finance expert.",
        tools=[],
    ),
    AgentConfig(                     # last item = fallback
        name="general",
        route_key="general",
        description="anything else",
        system_prompt="You are a general assistant.",
        tools=[],
    ),
]

# Single-topic query  → one agent called
result = run_supervisor("What is the capital of France?", agents)

# Multi-topic query   → multiple agents called in sequence
result = run_supervisor(
    "Explain index funds and how are they taxed?",
    agents,
)
print(result["final_response"])
```

---

## AgentConfig Fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Unique node name inside the graph (e.g. `"FinanceAgent"`) |
| `route_key` | `str` | Single lowercase word the supervisor outputs to reach this agent |
| `description` | `str` | One-line hint shown to the supervisor so it knows when to route here |
| `system_prompt` | `str` | Full system prompt passed to this agent at runtime |
| `tools` | `list` | `@tool`-decorated functions the agent can call (empty list = no tools) |

---

## run_supervisor API

```python
run_supervisor(user_input: str, agents: List[AgentConfig]) -> dict
```

**Raises** (before returning state):

| Exception | When |
|---|---|
| `ValueError` | `agents` list is empty, or `user_input` is blank |

**Returns** the full state dict:

| Key | Type | Description |
|---|---|---|
| `final_response` | `str` | Combined formatted output shown to the user |
| `routes` | `List[str]` | All route_keys identified by the supervisor |
| `agents_used` | `List[str]` | Names of agents that actually ran, in order |
| `agent_results` | `List[str]` | Per-agent response strings, in order |
| `agent_result` | `str` | All agent results joined (backward-compat alias) |
| `handoffs` | `List[str]` | Full handoff chain including tool calls |
| `aborted` | `bool` | `True` when the run was stopped by any guard or error |
| `abort_reason` | `str` | `"injection"` \| `"timeout"` \| `"api_error"` \| `"network_error"` \| `"runtime_error"` \| `"llm_error"` \| `""` |

---

## Multi-Agent Detection & Sequential Execution

### How it works

1. **Supervisor analyses the query** — identifies every topic and how many agents are needed.
   Returns one `route_key: focused sub-query` line per agent.
2. **Single agent?** → the **full original query** is passed to that agent unchanged.
   No splitting, no sub-query extraction — the agent answers the complete question.
3. **Multiple different agents?** → the query is **split into focused sub-queries**.
   Each agent receives only the part of the query relevant to it, preventing any
   agent from answering topics outside its domain.
4. **Deduplication** — if the supervisor returns the same route_key twice, the
   duplicate is logged and skipped. No agent is ever called twice for the same query.
5. **Iteration only when needed** — `[1/2]`, `[2/2]` handoff logs only appear when
   multiple different agents are invoked. Single-agent queries show a simple handoff.
6. **Abort on failure** — if any agent fails, the run aborts immediately with an
   `[ABORTED]` response identifying the failing agent.

### Decision logic

```
query received
    │
    ▼
supervisor analyses → how many agents needed?
    │
    ├── 1 agent → pass FULL original query to that agent (no split)
    │
    └── 2+ agents → split into focused sub-queries
                    → pass each agent only its relevant part
                    → iterate in sequence
```

### Example — single-agent query (full query passed, no split)

```
User: "What is the best way to save tax on my salary?"

Supervisor: 1 agent needed — tax
→ full query passed as-is to TaxAgent

INFO | Supervisor identified 1 agent needed: tax — full query passed as-is.
INFO | Handoff: supervisor -> TaxAgent
```

### Example — multi-agent query (query split into focused sub-queries)

```
User: "I want to plan a trip to Paris and also save tax on travel expenses"

Supervisor: 2 agents needed
  travel: I want to plan a trip to Paris
  tax: save tax on travel expenses

INFO | Supervisor identified 2 different agents needed: travel, tax
INFO |   [travel] sub-query: I want to plan a trip to Paris
INFO |   [tax] sub-query: save tax on travel expenses
INFO | Handoff [1/2]: supervisor -> TravelAgent | query: I want to plan a trip to Paris
INFO | Handoff [2/2]: supervisor -> TaxAgent    | query: save tax on travel expenses
```

Output — each agent answers only its own part:
```
[TRAVELAGENT]
Handoffs : supervisor -> TravelAgent

<answer about Paris trip only>

[TAXAGENT]
Handoffs : supervisor -> TravelAgent -> TaxAgent

<answer about tax on travel only>
```

### Duplicate route handling

```
Supervisor returns duplicate: finance, finance
→ INFO | Duplicate route 'finance' — skipping, no repeated agent call.
→ Only one FinanceAgent call is made, with the full original query.
```

### Inspecting which agents ran

```python
result = run_supervisor(user_input, agents)
print("Routes identified :", result["routes"])       # ["travel", "tax"]
print("Agents used       :", result["agents_used"])  # ["TravelAgent", "TaxAgent"]
print("Per-agent results :", result["agent_results"])
```

---

## Built-in Safety Features

### Prompt Injection Guard
- Runs **before** any agent is invoked
- 11 regex patterns: ignore-instructions, jailbreak, DAN mode, override, bypass, etc.
- Each pattern is wrapped in `try/except re.error` — a malformed pattern is logged
  and skipped, never crashes the run
- On match → aborts immediately with:
  `"do not try to hack the system, it is illegal"`
- Add custom patterns: `supervisor_skill.INJECTION_PATTERNS.append(r"your_pattern")`

### Timeout Guard
- Every LLM call and every tool call is timed
- Default limit: **30 seconds** (`MAX_AGENT_SECONDS = 30.0`)
- On breach → aborts with: `"please try with smaller query"`
- Override globally: `supervisor_skill.MAX_AGENT_SECONDS = 60.0`

---

## Error Handling

### LLM Initialisation Errors
Caught at import time — the program exits immediately with a clear message.

| Cause | Message shown |
|---|---|
| `langchain-openai` not installed | `langchain-openai is not installed. Run: pip install langchain-openai` |
| Missing / invalid API key | `LLM initialisation failed. Check that OPENAI_API_KEY is set in your .env file.` |

### LLM Call Errors (`_timed_llm_call`)
Every LLM call maps raw API exceptions to typed Python exceptions,
then to user-friendly `[ABORTED]` responses:

| Raw error | Mapped to | `abort_reason` | Message shown |
|---|---|---|---|
| `timeout` / `timed out` | `TimeoutError` | `"timeout"` | `"please try with smaller query"` |
| `401` / `authentication` / `invalid api key` | `PermissionError` | `"api_error"` | `"Invalid or missing API key..."` |
| `429` / `rate limit` / `quota` | `PermissionError` | `"api_error"` | `"API rate limit or quota exceeded..."` |
| `connection` / `network` / `unreachable` | `ConnectionError` | `"network_error"` | `"Cannot reach the API..."` |
| anything else | `RuntimeError` | `"llm_error"` | `"Supervisor failed to classify query: <detail>"` |

### Agent Errors (during sequential iteration)
The first agent that fails aborts the entire run. The `[ABORTED]` response
identifies which agent failed:

```
[ABORTED]
Handoffs : supervisor -> FinanceAgent -> TaxAgent
Message  : Agent 'TaxAgent' encountered an error: <detail>
```

### Tool Call Errors (`_timed_tool_call`)
Any exception raised inside a tool is caught, logged, and re-raised as
`RuntimeError` → `abort_reason = "runtime_error"`.

### Tool Not Found
If the LLM requests a tool name not in the agent's `tools` list, the call is
**skipped with a warning** — the loop continues rather than crashing:
```
WARNING | [BillingAgent] Requested tool 'unknown_tool' not found in tool_map — skipping.
```

### Empty LLM Response
If the LLM returns empty content (no text, no tool calls), a warning is logged
and a fallback message is substituted so `final_response` is never blank.

### Checking `abort_reason` in caller code

```python
result = run_supervisor(user_input, agents)

if result["aborted"]:
    reason = result["abort_reason"]
    if reason == "injection":
        pass  # log security event
    elif reason in ("api_error", "network_error"):
        pass  # retry or alert ops
    elif reason == "timeout":
        pass  # ask user to shorten query
    else:
        pass  # generic runtime error
```

---

## Observability

Every LLM call and tool call is logged, including each agent's focused sub-query:
```
INFO | Supervisor identified 2 different agents needed: travel, tax
INFO |   [travel] sub-query: I want to plan a trip to Paris
INFO |   [tax] sub-query: how to save tax on travel expenses
INFO | Handoff [1/2]: supervisor -> TravelAgent | query: I want to plan a trip to Paris
INFO | [TravelAgent call-1] time: 1.40s | tokens: {'input': 67, 'output': 22, 'total': 89}
INFO | Handoff [2/2]: supervisor -> TaxAgent | query: how to save tax on travel expenses
INFO | [TaxAgent call-1] time: 2.09s | tokens: {'input': 72, 'output': 54, 'total': 126}
```

Single-agent query log (no iteration notation):
```
INFO | Supervisor identified 1 agent needed: tax
INFO | Handoff: supervisor -> TaxAgent
INFO | [TaxAgent call-1] time: 1.43s | tokens: {'input': 74, 'output': 50, 'total': 124}
```

---

## Agent Ordering & Fallback

- **The last agent in the list is the fallback** — used when the supervisor outputs
  a route_key that matches no agent, or when the supervisor returns an empty list
- Each route_key is used **at most once** per call (deduplication is automatic)

```python
agents = [
    AgentConfig(name="FinanceAgent", route_key="finance", ...),
    AgentConfig(name="TaxAgent",     route_key="tax",     ...),
    AgentConfig(name="GeneralAgent", route_key="general", ...),  # <- fallback
]
```

---

## Customisation Recipes

### Change the LLM
```python
import supervisor_skill
from langchain_openai import ChatOpenAI
supervisor_skill.llm = ChatOpenAI(model="gpt-4o")
```

### Change timeout
```python
import supervisor_skill
supervisor_skill.MAX_AGENT_SECONDS = 60.0
```

### Add injection patterns
```python
import supervisor_skill
supervisor_skill.INJECTION_PATTERNS.append(r"my_custom_pattern")
```

### Sequential agent pipeline (chain agents)
Define agents whose `system_prompt` references prior context and pass the
`agent_result` from one `run_supervisor` call as input to the next.
