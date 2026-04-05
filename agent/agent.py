"""
Assemble the LangChain 1.x agent graph (`create_agent`) with tools, memory, and middleware.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from agent.llm_config import create_chat_model, default_llm_model_name
from agent.memory import build_checkpointer, build_conversation_buffer_memory
from agent.middleware import build_middleware_stack
from agent.tools import build_all_tools
from rag.rag_pipeline import build_rag_retriever_tool

logger = logging.getLogger(__name__)

BASE_SYSTEM_PROMPT = (
    "You are an intelligent agent that uses tools when necessary and explains reasoning. "
    "Follow a ReAct-style loop: think step by step, pick the right tool, interpret the "
    "observation, and continue until you can answer the user clearly."
)


def build_agent_graph(
    data_path: str | None = None,
) -> tuple[CompiledStateGraph, Any]:
    """
    Build the compiled agent graph and a ConversationBufferMemory instance (syllabus).

    Returns:
        graph: invoke with `graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "..."}})`
        buffer_memory: optional classic memory object for inspection / demos.
    """
    from pathlib import Path

    path = Path(data_path) if data_path else None
    rag_tool = build_rag_retriever_tool(data_path=path) if path else build_rag_retriever_tool()
    tools = build_all_tools(rag_tool)

    model = create_chat_model(default_llm_model_name(), temperature=0.2)

    graph = create_agent(
        model=model,
        tools=tools,
        system_prompt=BASE_SYSTEM_PROMPT,
        middleware=build_middleware_stack(),
        checkpointer=build_checkpointer(),
        debug=os.getenv("LANGCHAIN_AGENT_DEBUG", "").strip() in ("1", "true", "yes"),
    )
    buffer_memory = build_conversation_buffer_memory()
    logger.info("Agent graph compiled with %d tools.", len(tools))
    return graph, buffer_memory
