# Multi-Agent Supervisor System

A LangGraph-based agentic application where a **supervisor** analyses user input, identifies the required specialist agents, and routes each query (or sub-query) to the right agent — with full logging, error handling, and prompt-injection protection.

---

## Architecture

```
User Input
    │
    ▼
Supervisor (LLM)
    │
    ├── 1 agent needed  → full query passed to that agent
    │
    └── 2+ agents needed → query split into focused sub-queries
                           → each agent receives only its relevant part
                           → agents called in sequence (iteration)
```

### Available Agents

| Agent | Route Key | Handles |
|---|---|---|
| **FinanceAgent** | `finance` | Budgets, investments, banking, loans, money management |
| **TravelAgent** | `travel` | Trip planning, flights, hotels, itineraries, visas |
| **ShipmentAgent** | `shipment` | Shipping, logistics, package tracking, courier services |
| **SearchAgent** | `search` | Research, fact lookup, information retrieval |
| **TaxAgent** | `tax` | Income tax, GST, TDS, deductions, tax filing, tax planning |
| **GeneralAgent** | `general` | Anything not covered by the above *(fallback)* |

---

## Project Structure

```
lang_agents/
├── app.py                  # Main entry point — run this
├── supervisor_skill.py     # Reusable supervisor skill (AgentConfig + run_supervisor)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── plan.txt                # Original project plan
└── .claude/
    └── commands/
        └── langgraph-supervisor.md   # Skill documentation
```

---

## Setup

### 1. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install langchain langchain-openai langchain-core langgraph python-dotenv
```

### 3. Configure API keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## Running the App

```bash
python app.py
```

You will see:
```
=================================================================
  Multi-Agent Supervisor System (LangGraph)
  Agents: Finance | Travel | Shipment | Search | Tax | General
=================================================================

Enter input:
```

Type any query and press Enter.

---

## Example Queries

### Single-agent query
```
Enter input: How do I file my income tax return?
```
```
INFO | Supervisor identified 1 agent needed: tax — full query passed as-is.
INFO | Handoff: supervisor -> TaxAgent

[TAXAGENT]
Handoffs : supervisor -> TaxAgent

Filing ITR requires choosing the correct form based on your income type...
```

### Multi-agent query
```
Enter input: I want to plan a trip to Paris and also save tax on travel expenses
```
```
INFO | Supervisor identified 2 different agents needed: travel, tax
INFO |   [travel] sub-query: I want to plan a trip to Paris
INFO |   [tax] sub-query: save tax on travel expenses
INFO | Handoff [1/2]: supervisor -> TravelAgent
INFO | Handoff [2/2]: supervisor -> TaxAgent

[TRAVELAGENT]
Handoffs : supervisor -> TravelAgent

Book flights 6–8 weeks in advance for best fares...

[TAXAGENT]
Handoffs : supervisor -> TaxAgent

Business travel expenses are deductible under Section 37...
```

### Multiple questions (numbered or newline-separated)
```
Enter input:
1. What are index funds?
2. How do I track a shipment?
```
Each question is processed independently through the supervisor.

---

## Key Features

- **Automatic agent detection** — supervisor identifies how many agents are needed
- **Focused sub-queries** — for multi-agent queries, each agent receives only its relevant portion (no duplication)
- **Deduplication** — same agent is never called twice for the same query
- **Concise responses** — every agent is instructed to reply in 4 lines or fewer
- **Prompt injection protection** — 11 regex patterns block malicious input before any agent runs
- **Timeout guard** — every LLM and tool call is timed (default 30s limit)
- **Full error handling** — auth errors, rate limits, network failures, empty responses all handled gracefully
- **Observability** — every handoff, LLM call time, and token count is logged

---

## Supervisor Skill (`supervisor_skill.py`)

The supervisor logic is packaged as a reusable skill. Use it in any project:

```python
from supervisor_skill import AgentConfig, run_supervisor

agents = [
    AgentConfig(
        name="MyAgent",
        route_key="myagent",
        description="handles my domain",
        system_prompt="You are an expert in my domain.",
        tools=[],
    ),
    AgentConfig(  # fallback — must be last
        name="GeneralAgent",
        route_key="general",
        description="anything else",
        system_prompt="You are a general assistant.",
        tools=[],
    ),
]

result = run_supervisor("my query", agents)
print(result["final_response"])
```

### Return value keys

| Key | Description |
|---|---|
| `final_response` | Formatted output for each agent |
| `routes` | List of route_keys identified |
| `agents_used` | Names of agents that ran |
| `agent_results` | Per-agent response strings |
| `aborted` | `True` if run was stopped by a guard or error |
| `abort_reason` | `"injection"` / `"timeout"` / `"api_error"` / `"network_error"` / `"runtime_error"` |

---

## Configuration

Override globals in `supervisor_skill.py` as needed:

```python
import supervisor_skill

# Change LLM
from langchain_openai import ChatOpenAI
supervisor_skill.llm = ChatOpenAI(model="gpt-4o")

# Change timeout (seconds)
supervisor_skill.MAX_AGENT_SECONDS = 60.0

# Add custom injection patterns
supervisor_skill.INJECTION_PATTERNS.append(r"my_pattern")
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `langchain` | LLM framework |
| `langchain-openai` | OpenAI LLM integration |
| `langchain-core` | Messages, tools, base types |
| `langgraph` | Agent graph orchestration |
| `python-dotenv` | Load `.env` API keys |
