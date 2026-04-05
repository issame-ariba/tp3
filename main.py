"""
CLI entrypoint: loads .env, builds the agent, runs an interactive chat loop with thread memory.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when executed as a script
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.agent import build_agent_graph
from agent.memory import sync_buffer_from_lc_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


def _extract_reply(messages: list) -> str:
    from langchain_core.messages import AIMessage

    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if m.tool_calls or getattr(m, "invalid_tool_calls", None):
                continue
            c = m.content
            return c if isinstance(c, str) else str(c)
    return "(no assistant text)"


def main() -> None:
    load_dotenv(_ROOT / ".env", override=True)
    groq = (os.getenv("GROQ_API_KEY") or "").strip()
    oai = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not groq and not oai:
        logger.error("Renseignez GROQ_API_KEY ou OPENAI_API_KEY dans .env")
        sys.exit(1)

    data_file = _ROOT / "data" / "sample.txt"
    graph, buffer_memory = build_agent_graph(data_path=str(data_file))

    thread_id = os.getenv("AGENT_THREAD_ID", "cli-session-1")
    config = {"configurable": {"thread_id": thread_id}}

    print("Agent ready. Commands: /quit to exit, /new to reset thread id suffix.\n")
    n = 0
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("/quit", "/exit"):
            break
        if user.lower() == "/new":
            n += 1
            thread_id = f"cli-session-{n}"
            config = {"configurable": {"thread_id": thread_id}}
            print(f"New thread: {thread_id}\n")
            continue

        result = graph.invoke({"messages": [("user", user)]}, config=config)
        messages = result.get("messages", [])
        reply = _extract_reply(messages)
        print(f"Agent: {reply}\n")
        try:
            sync_buffer_from_lc_messages(buffer_memory, messages)
        except Exception as e:  # noqa: BLE001
            logger.debug("Buffer sync skipped: %s", e)


if __name__ == "__main__":
    main()
