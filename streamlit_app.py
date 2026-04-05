"""
Streamlit chat UI for the agent (bonus). Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit_app")


@st.cache_resource
def _get_graph_and_memory():
    from agent.agent import build_agent_graph

    data_file = _ROOT / "data" / "sample.txt"
    return build_agent_graph(data_path=str(data_file))


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
    st.set_page_config(page_title="Agentic AI Agent", layout="wide")
    st.title("Agentic AI — LangChain `create_agent` + RAG tool")

    groq = (os.getenv("GROQ_API_KEY") or "").strip()
    oai = (os.getenv("OPENAI_API_KEY") or "").strip()
    groq_ok = bool(groq) and not groq.lower().startswith("your_")
    oai_ok = bool(oai) and not oai.lower().startswith("your_")
    if not groq_ok and not oai_ok:
        st.error("Renseignez **GROQ_API_KEY** (recommandé) ou **OPENAI_API_KEY** dans `.env`, puis enregistrez.")
        st.stop()
    if groq_ok:
        st.sidebar.success("LLM : **Groq** (prioritaire si les deux clés sont présentes)")
    else:
        st.sidebar.info("LLM : **OpenAI**")

    auto = st.sidebar.checkbox(
        "Auto-approve sensitive tools (sets AGENT_AUTO_APPROVE for this session)",
        value=os.getenv("AGENT_AUTO_APPROVE", "0") == "1",
    )
    if auto:
        os.environ["AGENT_AUTO_APPROVE"] = "1"
    else:
        os.environ["AGENT_AUTO_APPROVE"] = "0"

    thread_id = st.sidebar.text_input("Thread ID (conversation memory)", value="streamlit-user-1")

    graph, _buffer = _get_graph_and_memory()
    config = {"configurable": {"thread_id": thread_id}}

    if "messages_ui" not in st.session_state:
        st.session_state.messages_ui = []

    for role, text in st.session_state.messages_ui:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("Ask something (try RAG: 'What is Agentic RAG in the course doc?')")
    if not prompt:
        return

    st.session_state.messages_ui.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = graph.invoke({"messages": [("user", prompt)]}, config=config)
                messages = result.get("messages", [])
                reply = _extract_reply(messages)
            except Exception as e:  # noqa: BLE001
                logger.exception("invoke failed")
                reply = f"Error: {e!s}"
        st.markdown(reply)

    st.session_state.messages_ui.append(("assistant", reply))


if __name__ == "__main__":
    main()
