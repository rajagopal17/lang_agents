"""
supervisor_skill.py — Generic config-driven multi-agent supervisor using LangGraph.

Supervisor analyses the user input, identifies ALL required agents, then calls
each one in sequence — collecting and combining their responses.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global settings (override from outside if needed)
# ---------------------------------------------------------------------------
MAX_AGENT_SECONDS: float = 30.0

INJECTION_PATTERNS: List[str] = [
    r"ignore (all |previous |prior )?instructions",
    r"disregard (all |previous |prior )?instructions",
    r"forget (all |previous |prior )?instructions",
    r"you are now",
    r"jailbreak",
    r"dan mode",
    r"pretend (you are|to be)",
    r"override",
    r"bypass",
    r"do anything now",
    r"roleplay as",
]

# ---------------------------------------------------------------------------
# LLM initialisation — fail fast with a clear message if key is missing
# ---------------------------------------------------------------------------
try:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini")
    logger.info("LLM initialised: gpt-4o-mini (OpenAI)")
except ImportError:
    logger.critical("langchain-openai is not installed. Run: pip install langchain-openai")
    raise
except Exception as e:
    logger.critical("Failed to initialise LLM: %s", e)
    raise RuntimeError(
        "LLM initialisation failed. Check that OPENAI_API_KEY is set in your .env file."
    ) from e


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    name: str          # unique node name, e.g. "FinanceAgent"
    route_key: str     # single lowercase word the supervisor outputs, e.g. "finance"
    description: str   # one-line hint shown to supervisor
    system_prompt: str # full system prompt for this agent
    tools: list = field(default_factory=list)  # @tool-decorated functions


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    user_input: str
    routes: List[str]           # all route_keys identified by supervisor
    agent_results: List[str]    # per-agent result strings
    agent_result: str           # combined result (backward compat)
    handoffs: List[str]         # full handoff chain
    final_response: str
    aborted: bool
    abort_reason: str
    agents_used: List[str]      # names of agents that ran


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_injection(text: str) -> bool:
    """Return True if text matches any known prompt-injection pattern."""
    for pattern in INJECTION_PATTERNS:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        except re.error as e:
            logger.error("Invalid injection regex pattern '%s': %s", pattern, e)
    return False


def _timed_llm_call(node_name: str, call_num: int, messages: list):
    """
    Invoke the global LLM, log time + tokens.
    Raises:
        TimeoutError    — call exceeded MAX_AGENT_SECONDS
        PermissionError — authentication / quota error
        ConnectionError — network / API unreachable
        RuntimeError    — any other LLM error
    """
    start = time.time()
    try:
        result = llm.invoke(messages)
    except Exception as e:
        elapsed = time.time() - start
        err_str = str(e).lower()

        if any(k in err_str for k in ("timeout", "timed out", "read timeout")):
            logger.error("[%s call-%d] LLM timed out after %.2fs", node_name, call_num, elapsed)
            raise TimeoutError(f"LLM call by '{node_name}' timed out") from e

        if any(k in err_str for k in ("401", "authentication", "invalid api key", "incorrect api key")):
            logger.error("[%s call-%d] Authentication error: %s", node_name, call_num, e)
            raise PermissionError(
                "Invalid or missing API key. Check OPENAI_API_KEY in your .env file."
            ) from e

        if any(k in err_str for k in ("429", "rate limit", "quota", "resource_exhausted")):
            logger.error("[%s call-%d] Rate limit / quota exceeded: %s", node_name, call_num, e)
            raise PermissionError(
                "API rate limit or quota exceeded. Please wait a moment and try again."
            ) from e

        if any(k in err_str for k in ("connection", "network", "unreachable", "name or service not known")):
            logger.error("[%s call-%d] Network error: %s", node_name, call_num, e)
            raise ConnectionError(
                "Cannot reach the API. Check your internet connection."
            ) from e

        logger.error("[%s call-%d] Unexpected LLM error: %s", node_name, call_num, e)
        raise RuntimeError(f"LLM error in '{node_name}': {e}") from e

    elapsed = time.time() - start
    tokens = {}
    if hasattr(result, "usage_metadata") and result.usage_metadata:
        um = result.usage_metadata
        tokens = {
            "input": um.get("input_tokens", 0),
            "output": um.get("output_tokens", 0),
            "total": um.get("total_tokens", 0),
        }
    logger.info("[%s call-%d] time: %.2fs | tokens: %s", node_name, call_num, elapsed, tokens)

    if elapsed > MAX_AGENT_SECONDS:
        raise TimeoutError(f"LLM call by '{node_name}' exceeded {MAX_AGENT_SECONDS}s")

    if not getattr(result, "content", None) and not getattr(result, "tool_calls", None):
        logger.warning("[%s call-%d] LLM returned an empty response.", node_name, call_num)

    return result


def _timed_tool_call(node_name: str, tool, args: dict):
    """
    Invoke a tool, log time.
    Raises:
        TimeoutError — tool exceeded MAX_AGENT_SECONDS
        RuntimeError — tool raised an unexpected exception
    """
    start = time.time()
    try:
        result = tool.invoke(args)
    except Exception as e:
        elapsed = time.time() - start
        logger.error("[%s] tool '%s' failed after %.2fs: %s", node_name, tool.name, elapsed, e)
        raise RuntimeError(f"Tool '{tool.name}' failed: {e}") from e

    elapsed = time.time() - start
    logger.info("[%s] tool '%s' | time: %.2fs", node_name, tool.name, elapsed)

    if elapsed > MAX_AGENT_SECONDS:
        raise TimeoutError(f"Tool '{tool.name}' exceeded {MAX_AGENT_SECONDS}s")

    return result


def _abort_state(state: GraphState, reason: str, message: str) -> GraphState:
    state["aborted"] = True
    state["abort_reason"] = reason
    handoff_str = " -> ".join(state["handoffs"])
    state["final_response"] = (
        f"[ABORTED]\n"
        f"Handoffs : {handoff_str}\n"
        f"Message  : {message}"
    )
    return state


def _run_agent(
    agent: AgentConfig,
    user_input: str,
    state: GraphState,
) -> Tuple[Optional[str], Optional[GraphState]]:
    """
    Run a single agent against user_input.

    Returns:
        (result_str, None)   — on success
        (None, aborted_state) — on any error
    """
    messages = [
        SystemMessage(
            content=agent.system_prompt
            + "\n\nIMPORTANT: Your reply must be no more than 4 lines — be very short and concise."
        ),
        HumanMessage(content=user_input),
    ]

    try:
        if agent.tools:
            agent_llm = llm.bind_tools(agent.tools)
            tool_map = {t.name: t for t in agent.tools}
            call_num = 1

            while True:
                start = time.time()
                response = agent_llm.invoke(messages)
                elapsed = time.time() - start
                tokens = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    um = response.usage_metadata
                    tokens = {
                        "input": um.get("input_tokens", 0),
                        "output": um.get("output_tokens", 0),
                        "total": um.get("total_tokens", 0),
                    }
                logger.info(
                    "[%s call-%d] time: %.2fs | tokens: %s",
                    agent.name, call_num, elapsed, tokens,
                )
                if elapsed > MAX_AGENT_SECONDS:
                    raise TimeoutError(f"Agent '{agent.name}' exceeded {MAX_AGENT_SECONDS}s")

                if not response.tool_calls:
                    result = response.content or ""
                    if not result:
                        logger.warning("[%s] Agent returned empty content.", agent.name)
                        result = "Agent returned no response."
                    return result, None

                messages.append(response)
                for tc in response.tool_calls:
                    tool = tool_map.get(tc["name"])
                    if tool is None:
                        logger.warning(
                            "[%s] Requested tool '%s' not found in tool_map — skipping.",
                            agent.name, tc["name"],
                        )
                        continue
                    tool_result = _timed_tool_call(agent.name, tool, tc["args"])
                    state["handoffs"].append(f"tool:{tc['name']}")
                    logger.info("Handoff: %s -> tool:%s", agent.name, tc["name"])
                    messages.append(
                        ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                    )
                call_num += 1

        else:
            response = _timed_llm_call(agent.name, 1, messages)
            result = response.content or ""
            if not result:
                logger.warning("[%s] Agent returned empty content.", agent.name)
                result = "Agent returned no response."
            return result, None

    except TimeoutError:
        return None, _abort_state(state, "timeout", "please try with smaller query")
    except PermissionError as e:
        logger.error("[%s] Auth/quota error: %s", agent.name, e)
        return None, _abort_state(state, "api_error", str(e))
    except ConnectionError as e:
        logger.error("[%s] Network error: %s", agent.name, e)
        return None, _abort_state(state, "network_error", str(e))
    except RuntimeError as e:
        logger.error("[%s] Tool or LLM error: %s", agent.name, e)
        return None, _abort_state(
            state, "runtime_error",
            f"Agent '{agent.name}' encountered an error: {e}"
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_supervisor(user_input: str, agents: List[AgentConfig]) -> GraphState:
    """
    Supervisor analyses user_input, identifies ALL required agents, then calls
    each one in sequence. Returns the combined state dict.

    Returns keys: final_response, handoffs, aborted, abort_reason,
                  routes, agents_used, agent_results, agent_result.
    """
    if not agents:
        raise ValueError("The agents list must not be empty.")
    if not isinstance(user_input, str) or not user_input.strip():
        raise ValueError("user_input must be a non-empty string.")

    state: GraphState = {
        "user_input": user_input,
        "routes": [],
        "agent_results": [],
        "agent_result": "",
        "handoffs": ["supervisor"],
        "final_response": "",
        "aborted": False,
        "abort_reason": "",
        "agents_used": [],
    }

    # ── Injection guard ────────────────────────────────────────────────────
    if _check_injection(user_input):
        logger.warning("Prompt injection detected in input.")
        state["handoffs"].append("supervisor (BLOCKED: injection detected)")
        return _abort_state(state, "injection", "do not try to hack the system, it is illegal")

    # ── Build routing prompt (identify agents + extract focused sub-query) ──
    valid_keys = [a.route_key for a in agents]
    agent_descriptions = "\n".join(
        f"  {a.route_key}: {a.description}" for a in agents
    )
    routing_prompt = (
        "You are a supervisor that analyses user queries and routes each topic to the right specialist.\n"
        "Available agents (route_key: description):\n"
        f"{agent_descriptions}\n\n"
        "Instructions:\n"
        "- Read the user query carefully.\n"
        "- Identify every topic that needs a specialist agent.\n"
        "- For each needed agent, extract ONLY the part of the query relevant to that agent.\n"
        "- Reply with one line per agent in this exact format:\n"
        "    route_key: focused question for that agent only\n"
        "- Use each route_key at most once.\n"
        f"- Only use keys from this list: {', '.join(valid_keys)}\n"
        "- No extra explanation, no blank lines.\n\n"
        "Example output:\n"
        "finance: What are index funds?\n"
        "tax: How are capital gains from index funds taxed?"
    )

    # ── Classify routes and extract sub-queries ────────────────────────────
    logger.info("Supervisor analysing query: %s", user_input[:80])
    try:
        route_resp = _timed_llm_call(
            "supervisor", 1,
            [SystemMessage(content=routing_prompt), HumanMessage(content=user_input)],
        )
        raw_routes = (route_resp.content or "").strip()
    except TimeoutError:
        return _abort_state(state, "timeout", "please try with smaller query")
    except PermissionError as e:
        logger.error("Supervisor auth/quota error: %s", e)
        return _abort_state(state, "api_error", str(e))
    except ConnectionError as e:
        logger.error("Supervisor network error: %s", e)
        return _abort_state(state, "network_error", str(e))
    except RuntimeError as e:
        logger.error("Supervisor LLM error: %s", e)
        return _abort_state(state, "llm_error", f"Supervisor failed to classify query: {e}")

    # Parse "route_key: sub_query" lines — deduplicate while preserving order
    seen = set()
    route_keys: List[str] = []
    sub_queries: dict = {}          # route_key -> focused question for that agent

    for line in raw_routes.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, sub_q = line.partition(":")
        key = key.strip().lower()
        sub_q = sub_q.strip()
        if not key or key not in valid_keys:
            continue
        if key in seen:
            logger.info(
                "Duplicate route '%s' returned by supervisor — skipping, no repeated agent call.", key
            )
            continue
        seen.add(key)
        route_keys.append(key)
        sub_queries[key] = sub_q if sub_q else user_input

    if not route_keys:
        logger.warning("Supervisor returned no routes — falling back to last agent.")
        route_keys = [agents[-1].route_key]
        sub_queries[agents[-1].route_key] = user_input

    state["routes"] = route_keys
    multi_agent = len(route_keys) > 1

    if not multi_agent:
        # Single agent — pass the full original query, no splitting required
        sub_queries[route_keys[0]] = user_input
        logger.info(
            "Supervisor identified 1 agent needed: %s — full query passed as-is.",
            route_keys[0],
        )
    else:
        # Multiple different agents — each receives only its focused sub-query
        logger.info(
            "Supervisor identified %d different agents needed: %s",
            len(route_keys), ", ".join(route_keys),
        )
        for k, q in sub_queries.items():
            logger.info("  [%s] sub-query: %s", k, q)

    # ── Build agent lookup map ─────────────────────────────────────────────
    agent_map = {a.route_key: a for a in agents}
    fallback_agent = agents[-1]

    # ── Call each unique agent with its focused sub-query ─────────────────
    # Iteration only happens when multiple different agents are identified.
    sections: List[str] = []

    for idx, route_key in enumerate(route_keys, start=1):
        matched_agent = agent_map.get(route_key)
        if matched_agent is None:
            logger.warning(
                "Route '%s' has no matching agent — using fallback '%s'.",
                route_key, fallback_agent.name,
            )
            matched_agent = fallback_agent

        agent_query = sub_queries.get(route_key, user_input)

        state["handoffs"].append(matched_agent.name)
        state["agents_used"].append(matched_agent.name)
        if multi_agent:
            logger.info(
                "Handoff [%d/%d]: supervisor -> %s | query: %s",
                idx, len(route_keys), matched_agent.name, agent_query[:60],
            )
        else:
            logger.info("Handoff: supervisor -> %s", matched_agent.name)

        result, err_state = _run_agent(matched_agent, agent_query, state)
        if err_state is not None:
            return err_state  # abort on first agent failure

        state["agent_results"].append(result)

        handoff_str = " -> ".join(state["handoffs"])
        sections.append(
            f"[{matched_agent.name.upper()}]\n"
            f"Handoffs : {handoff_str}\n\n"
            f"{result}"
        )

    # ── Combine all agent responses ────────────────────────────────────────
    state["agent_result"] = "\n\n".join(state["agent_results"])
    state["final_response"] = ("\n\n" + "=" * 60 + "\n\n").join(sections)
    return state
