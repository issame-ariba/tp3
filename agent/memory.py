"""
Conversation memory helpers.

`create_agent` persists multi-turn state via a LangGraph checkpointer (thread_id).
We use MemorySaver as the modern equivalent of buffering full message history.

`ConversationBufferMemory` from langchain_classic is kept for coursework alignment
and optional inspection of the last turns outside the graph.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_classic.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


def build_checkpointer() -> MemorySaver:
    """In-memory checkpoint store: pass to `create_agent(..., checkpointer=...)`."""
    logger.info("Using MemorySaver checkpointer for conversation threads.")
    return MemorySaver()


def build_conversation_buffer_memory() -> ConversationBufferMemory:
    """
    Classic buffer memory (return_messages=True) for syllabus compliance.
    Call `sync_buffer_from_lc_messages` after each agent turn if you want it updated.
    """
    return ConversationBufferMemory(return_messages=True)


def sync_buffer_from_lc_messages(
    memory: ConversationBufferMemory,
    messages: list[BaseMessage],
) -> None:
    """Append the last human/ai pair from LangChain messages into the buffer (best-effort)."""
    if len(messages) < 2:
        return
    last = messages[-2:]
    human: HumanMessage | None = None
    ai: AIMessage | None = None
    for m in last:
        if isinstance(m, HumanMessage):
            human = m
        elif isinstance(m, AIMessage) and not (m.tool_calls or getattr(m, "invalid_tool_calls", None)):
            ai = m
    if human and ai:
        memory.save_context({"input": str(human.content)}, {"output": str(ai.content)})


def messages_preview(messages: list[BaseMessage], max_items: int = 8) -> list[dict[str, Any]]:
    """Lightweight log helper for debugging chat history."""
    out: list[dict[str, Any]] = []
    for m in messages[-max_items:]:
        role = m.__class__.__name__
        content = getattr(m, "content", "") or ""
        if isinstance(content, list):
            content = str(content)
        out.append({"role": role, "content": content[:500]})
    return out
