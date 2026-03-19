"""
app.py — Multi-Agent Supervisor System
Agents: FinanceAgent | TravelAgent | ShipmentAgent | SearchAgent | TaxAgent | GeneralAgent

Run:
    python app.py
"""

import re
import sys
import logging

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from supervisor_skill import AgentConfig, run_supervisor
except RuntimeError as e:
    print(f"[STARTUP ERROR] {e}")
    sys.exit(1)
except ImportError as e:
    print(f"[IMPORT ERROR] Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------
AGENTS = [
    AgentConfig(
        name="FinanceAgent",
        route_key="finance",
        description="financial questions, budgets, investments, banking, money management, loans",
        system_prompt=(
            "You are an expert financial advisor. Help users with budgeting, investments, "
            "banking, loans, savings, and money management. Provide clear, actionable advice."
        ),
        tools=[],
    ),
    AgentConfig(
        name="TravelAgent",
        route_key="travel",
        description="travel planning, flights, hotels, destinations, itineraries, visas, tourism",
        system_prompt=(
            "You are an expert travel consultant. Help users plan trips, discover destinations, "
            "arrange flights and hotels, create itineraries, and navigate visa requirements."
        ),
        tools=[],
    ),
    AgentConfig(
        name="ShipmentAgent",
        route_key="shipment",
        description="shipping, logistics, package tracking, delivery, freight, courier services",
        system_prompt=(
            "You are a logistics and shipment expert. Help users with shipping options, "
            "package tracking, delivery timelines, freight services, and courier comparisons."
        ),
        tools=[],
    ),
    AgentConfig(
        name="SearchAgent",
        route_key="search",
        description="search for information, research topics, look up facts, find data or news",
        system_prompt=(
            "You are a research and information specialist. Help users find accurate information, "
            "research topics thoroughly, and present well-sourced facts and data."
        ),
        tools=[],
    ),
    AgentConfig(
        name="TaxAgent",
        route_key="tax",
        description="taxes, tax filing, deductions, GST, income tax, TDS, tax planning, returns",
        system_prompt=(
            "You are a certified tax expert. Help users understand tax obligations, file returns, "
            "maximize deductions, plan tax-saving strategies, and navigate GST, TDS, and income tax."
        ),
        tools=[],
    ),
    AgentConfig(  # fallback (must be last)
        name="GeneralAgent",
        route_key="general",
        description="anything else not covered by other specialist agents",
        system_prompt=(
            "You are a knowledgeable general assistant. Answer any question the user has "
            "clearly, helpfully, and concisely."
        ),
        tools=[],
    ),
]


# ---------------------------------------------------------------------------
# Multi-question splitter
# ---------------------------------------------------------------------------
def split_questions(text: str) -> list[str]:
    """
    Split a block of text into individual questions/tasks.
    Handles:
      - Numbered lists  (1. ... 2. ...)
      - Newline-separated items

    Note: does NOT split on '?' — the supervisor already handles
    multiple topics within a single query via multi-agent detection.
    """
    try:
        # Numbered list: "1. foo 2. bar" or "1) foo 2) bar"
        numbered = re.split(r"\b\d+[.)]\s+", text)
        numbered = [q.strip() for q in numbered if q.strip()]
        if len(numbered) > 1:
            return numbered

        # Newline-separated items
        by_newline = [line.strip() for line in text.splitlines() if line.strip()]
        if len(by_newline) > 1:
            return by_newline

    except Exception as e:
        logger.warning("split_questions failed unexpectedly: %s — treating as single query.", e)

    return [text]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 65)
    print("  Multi-Agent Supervisor System (LangGraph)")
    print("  Agents: Finance | Travel | Shipment | Search | Tax | General")
    print("=" * 65)

    # ── Read input ─────────────────────────────────────────────────────────
    try:
        user_input = input("\nEnter input: ").strip()
    except EOFError:
        print("\n[INPUT ERROR] No input received (stdin closed). Exiting.")
        return
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting.")
        return

    if not user_input:
        print("[ERROR] No input provided. Please enter a question or query.")
        return

    # ── Split into individual questions ────────────────────────────────────
    questions = split_questions(user_input)

    if len(questions) > 1:
        logger.info("Detected %d question(s). Processing each separately.", len(questions))
        print(f"\nDetected {len(questions)} question(s). Processing each...\n")

    # ── Process each question ──────────────────────────────────────────────
    for idx, question in enumerate(questions, start=1):
        if len(questions) > 1:
            print(f"\n{'-' * 65}")
            print(f"  Question {idx}: {question}")
            print(f"{'-' * 65}")
            logger.info("Processing question %d: %s", idx, question)

        try:
            result = run_supervisor(question, AGENTS)
        except ValueError as e:
            logger.error("Invalid input for question %d: %s", idx, e)
            print(f"\n[INPUT ERROR] {e}")
            continue
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Stopping.")
            break
        except Exception as e:
            logger.error("Unexpected error processing question %d: %s", idx, e)
            print(f"\n[ERROR] An unexpected error occurred: {e}")
            print("        Please try again or rephrase your question.")
            continue

        if result["aborted"]:
            print()
            print(result["final_response"])
        else:
            for agent_name, agent_reply in zip(result["agents_used"], result["agent_results"]):
                print()
                print(f"[{agent_name.upper()}]")
                print(f"Handoffs : supervisor -> {agent_name}")
                print()
                print(agent_reply)
                print()

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
