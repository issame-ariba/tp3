"""
AgentMiddleware implementations: logging, guardrails, PII, HITL, dynamic prompt, dynamic model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from agent.llm_config import chat_model_label, coding_llm_model_name, create_chat_model, default_llm_model_name
from utils.guardrails import SAFE_REFUSAL, contains_blocked_content
from utils.pii import mask_pii

logger = logging.getLogger(__name__)

# Tools that can have side effects or run arbitrary code — require confirmation.
SENSITIVE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "python_repl",
        "tavily_search",
        "duckduckgo_search",
        "knowledge_base_search",
    }
)

# Pluggable confirmation (CLI uses input(); Streamlit overrides via set_hitl_confirm).
_hitl_confirm: Callable[[str], bool] | None = None


def set_hitl_confirm(fn: Callable[[str], bool] | None) -> None:
    """Register a yes/no confirmation callback for sensitive tools (or None for default)."""
    global _hitl_confirm
    _hitl_confirm = fn


def _default_confirm(message: str) -> bool:
    if os.getenv("AGENT_AUTO_APPROVE", "").strip() in ("1", "true", "yes"):
        return True
    try:
        answer = input(f"{message} Type 'yes' or 'no': ").strip().lower()
    except EOFError:
        return False
    return answer in ("yes", "y")


def _confirm(message: str) -> bool:
    fn = _hitl_confirm or _default_confirm
    return fn(message)


def _message_text(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(c)


def _last_human_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _message_text(m)
    return ""


def _is_coding_query(text: str) -> bool:
    t = text.lower()
    keywords = (
        "def ",
        "class ",
        "import ",
        "function",
        "refactor",
        "debug",
        "typescript",
        "javascript",
        "python",
        "bug",
        "stack trace",
        "leetcode",
        "async ",
        "pytest",
    )
    return any(k in t for k in keywords)


def _prompt_addon_for_query(text: str) -> str:
    """Extra system instructions based on the latest user message."""
    t = text.lower()
    bits: list[str] = []
    if _is_coding_query(text):
        bits.append(
            "Coding task: give concise, correct code; cite language in fenced blocks; "
            "explain reasoning briefly after non-trivial steps."
        )
    if any(w in t for w in ("search", "news", "today", "latest", "current")):
        bits.append("For fresh facts or news, prefer web search tools (Tavily or DuckDuckGo).")
    if any(w in t for w in ("course", "knowledge", "document", "sample.txt", "rag", "middleware")):
        bits.append("For project/course document facts, prefer `knowledge_base_search`.")
    if not bits:
        bits.append("Use tools only when they add verifiable value; otherwise answer directly.")
    return " ".join(bits)


class LoggingMiddleware(AgentMiddleware):
    """Logs lifecycle hooks before and after the agent run."""

    name = "logging_middleware"

    def before_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        logger.info("--- before_agent: starting turn ---")
        msgs = state.get("messages") or []
        logger.debug("message_count=%d", len(msgs))
        return None

    def after_agent(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        logger.info("--- after_agent: turn complete ---")
        return None


class GuardrailsMiddleware(AgentMiddleware):
    """Blocks disallowed topics (see utils.guardrails)."""

    name = "guardrails_middleware"

    def wrap_model_call(self, request: ModelRequest, handler: Callable[..., Any]) -> Any:
        text = _last_human_text(request.messages)
        if contains_blocked_content(text):
            logger.warning("Guardrails blocked model call.")
            return AIMessage(content=SAFE_REFUSAL)
        return handler(request)


class PIIMiddleware(AgentMiddleware):
    """Masks PII in the latest human turn before the model sees it."""

    name = "pii_middleware"

    def wrap_model_call(self, request: ModelRequest, handler: Callable[..., Any]) -> Any:
        new_messages: list[BaseMessage] = []
        changed = False
        for m in request.messages:
            if isinstance(m, HumanMessage):
                cleaned = mask_pii(_message_text(m))
                if cleaned != _message_text(m):
                    changed = True
                    new_messages.append(HumanMessage(content=cleaned))
                else:
                    new_messages.append(m)
            else:
                new_messages.append(m)
        if changed:
            logger.info("PII middleware masked content in HumanMessage(s).")
            request = request.override(messages=new_messages)
        return handler(request)


class DynamicPromptMiddleware(AgentMiddleware):
    """Appends task-specific system guidance based on the user query."""

    name = "dynamic_prompt_middleware"

    def wrap_model_call(self, request: ModelRequest, handler: Callable[..., Any]) -> Any:
        text = _last_human_text(request.messages)
        addon = _prompt_addon_for_query(text)
        if request.system_message:
            base = request.system_message.text
        else:
            base = ""
        merged = f"{base}\n\n[Context-specific guidance]\n{addon}"
        new_sys = SystemMessage(content=merged)
        return handler(request.override(system_message=new_sys))


class DynamicModelMiddleware(AgentMiddleware):
    """Uses a stronger chat model when the query looks like a coding task."""

    name = "dynamic_model_middleware"

    def __init__(self) -> None:
        super().__init__()
        self._default_model = create_chat_model(default_llm_model_name(), temperature=0.2)
        self._coding_model = create_chat_model(coding_llm_model_name(), temperature=0.1)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[..., Any]) -> Any:
        text = _last_human_text(request.messages)
        model = self._coding_model if _is_coding_query(text) else self._default_model
        if model is not request.model:
            logger.info("Dynamic model selection: using %s", chat_model_label(model))
        return handler(request.override(model=model))


class HumanInTheLoopMiddleware(AgentMiddleware):
    """Asks for confirmation before executing sensitive tools."""

    name = "human_in_the_loop_middleware"

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Any],
    ) -> Any:
        name = request.tool_call.get("name", "")
        if name not in SENSITIVE_TOOL_NAMES:
            return handler(request)

        args = request.tool_call.get("args") or {}
        preview = str(args)[:400]
        ok = _confirm(
            f"Sensitive tool '{name}' is about to run with arguments (truncated): {preview!s}.\nProceed?"
        )
        if not ok:
            logger.info("User declined execution of tool %s", name)
            return ToolMessage(
                content="User declined execution of this tool.",
                tool_call_id=str(request.tool_call["id"]),
            )
        return handler(request)


def build_middleware_stack() -> list[AgentMiddleware]:
    """Order: outermost first (logging → guardrails → PII → prompt → model → HITL)."""
    return [
        LoggingMiddleware(),
        GuardrailsMiddleware(),
        PIIMiddleware(),
        DynamicPromptMiddleware(),
        DynamicModelMiddleware(),
        HumanInTheLoopMiddleware(),
    ]
