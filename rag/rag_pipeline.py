"""
Agentic RAG: build a FAISS-backed retriever and expose it as a LangChain tool.

The agent chooses when to call `knowledge_base_search` versus other tools.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample.txt"
_HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _openai_embeddings_ok() -> bool:
    k = (os.getenv("OPENAI_API_KEY") or "").strip()
    return bool(k) and not k.lower().startswith("your_")


def build_embeddings() -> Embeddings:
    """OpenAI si clé valide ; sinon Hugging Face local (Groq seul, sans OpenAI)."""
    if _openai_embeddings_ok():
        from langchain_openai import OpenAIEmbeddings

        logger.info("RAG: OpenAIEmbeddings")
        return OpenAIEmbeddings()
    from langchain_community.embeddings import HuggingFaceEmbeddings

    logger.info("RAG: HuggingFaceEmbeddings (%s)", _HF_EMBED_MODEL)
    return HuggingFaceEmbeddings(model_name=_HF_EMBED_MODEL)


def load_documents_from_path(path: Path) -> str:
    """Load raw text from a .txt file (extend for PDF via pypdf if needed)."""
    if not path.exists():
        logger.warning("Data file missing at %s — using empty corpus.", path)
        return ""
    text = path.read_text(encoding="utf-8")
    logger.info("Loaded %d characters from %s", len(text), path)
    return text


def load_pdf_text(path: Path) -> str:
    """Optional: extract text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed; skipping PDF %s", path)
        return ""
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)


def build_rag_retriever_tool(
    data_path: Path | None = None,
    pdf_path: Path | None = None,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    k: int = 4,
) -> BaseTool:
    """
    Split documents, embed (OpenAI ou Hugging Face), FAISS, outil retriever.
    """
    data_path = data_path or DEFAULT_DATA_PATH
    corpus_parts: list[str] = []
    corpus_parts.append(load_documents_from_path(data_path))
    if pdf_path and pdf_path.exists():
        corpus_parts.append(load_pdf_text(pdf_path))

    full_text = "\n\n".join(p for p in corpus_parts if p).strip()
    if not full_text:
        full_text = "No documents loaded. Inform the user that the knowledge base is empty."

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)
    logger.info("Split corpus into %d chunks", len(chunks))

    embeddings = build_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    tool = create_retriever_tool(
        retriever,
        name="knowledge_base_search",
        description=(
            "Search the local course knowledge base (FAISS). "
            "Use for questions about the project architecture, ReAct, RAG, middleware, "
            "or content in data/sample.txt. Do not use for live web facts."
        ),
    )
    return tool
