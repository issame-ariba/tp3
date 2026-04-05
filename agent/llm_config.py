"""
Chat LLM: Groq si GROQ_API_KEY est définie, sinon OpenAI.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel


def _use_groq() -> bool:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return bool(key) and not key.lower().startswith("your_")


def create_chat_model(model_name: str, *, temperature: float = 0.2) -> BaseChatModel:
    if _use_groq():
        from langchain_groq import ChatGroq

        return ChatGroq(model=model_name, temperature=temperature)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=temperature)


def default_llm_model_name() -> str:
    if _use_groq():
        return os.getenv("GROQ_MODEL_DEFAULT", "llama-3.1-8b-instant")
    return os.getenv("OPENAI_MODEL_DEFAULT", "gpt-4o")


def coding_llm_model_name() -> str:
    if _use_groq():
        return os.getenv("GROQ_MODEL_CODING", "llama-3.3-70b-versatile")
    return os.getenv("OPENAI_MODEL_CODING", "gpt-4o")


def chat_model_label(model: BaseChatModel) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or model.__class__.__name__
    )


def llm_backend_name() -> str:
    return "groq" if _use_groq() else "openai"
