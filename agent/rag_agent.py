"""
RAG Agent — the heart of the system.

Wires together:
  • Google ADK LlmAgent with our custom OllamaLlm
  • retrieve_from_knowledge_base tool
  • SQLite persistent memory (MemoryStore)
  • FAISS vector store (via the tool)
  • Automatic conversation summarization
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

import ollama
from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Register OllamaLlm with the ADK registry (import triggers LLMRegistry.register)
import agent.ollama_llm  # noqa: F401

from agent.prompts import SUMMARIZE_PROMPT, SYSTEM_PROMPT
from agent.tools import calculate, get_current_datetime, retrieve_from_knowledge_base
from config import settings
from ingestion.chunker import chunk_text
from ingestion.loader import load_document
from storage.memory_store import memory_store
from storage.vector_store import vector_store

log = logging.getLogger(__name__)

_USER_ID = "local_user"  # single-user personal app


class RAGAgent:
    """Orchestrates the full RAG pipeline using Google ADK."""

    def __init__(self) -> None:
        self._session_service = InMemorySessionService()

        # Dynamic instruction: injects per-session summary when available
        def _instruction(ctx: ReadonlyContext) -> str:
            summary = memory_store.get_summary(ctx.session.id)
            prompt = SYSTEM_PROMPT
            if summary:
                prompt += (
                    "\n\n=== PREVIOUS CONVERSATION SUMMARY ===\n"
                    + summary
                    + "\n(Use this summary for context on earlier turns.)"
                )
            return prompt

        self._agent = Agent(
            name="knowledge_base_rag_agent",
            model=f"ollama/{settings.ollama_model}",
            description=(
                "A grounded personal knowledge base assistant that answers "
                "questions exclusively from ingested documents."
            ),
            instruction=_instruction,
            tools=[retrieve_from_knowledge_base, calculate, get_current_datetime],
        )

        self._runner = Runner(
            app_name=settings.app_name,
            agent=self._agent,
            session_service=self._session_service,
            auto_create_session=True,
        )

    # public API 

    async def chat(self, session_id: str, user_message: str) -> dict:
        """
        Send a message and get a grounded response.

        Returns:
            {"answer": str, "session_id": str}
        """
        memory_store.add_message(session_id, "user", user_message)

        new_message = types.Content(
            role="user",
            parts=[types.Part(text=user_message)],
        )

        final_text = ""
        async for event in self._runner.run_async(
            user_id=_USER_ID,
            session_id=session_id,
            new_message=new_message,
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts or []:
                    if part.text:
                        final_text += part.text

        memory_store.add_message(session_id, "assistant", final_text)

        # Trigger background summarization every N turns
        count = memory_store.message_count(session_id)
        if count > 0 and count % settings.summarize_after_turns == 0:
            asyncio.create_task(self._summarize_session(session_id))

        return {"answer": final_text, "session_id": session_id}

    def ingest_file(self, file_path: str | Path) -> dict:
        """
        Load, chunk, embed, and index a single document.

        Returns:
            {"file": str, "chunks": int, "status": str}
        """
        path = Path(file_path)
        log.info("[ingest] Loading %s", path.name)
        content = load_document(path)
        chunks = chunk_text(content, path.name)
        vector_store.add_chunks(chunks)
        log.info("[ingest] Done — %d chunks from %s", len(chunks), path.name)
        return {"file": path.name, "chunks": len(chunks), "status": "ingested"}

    def ingest_directory(self, dir_path: str | Path) -> list[dict]:
        """Ingest all supported documents in a directory."""
        from ingestion.loader import load_directory
        results = []
        for filename, content in load_directory(Path(dir_path)):
            chunks = chunk_text(content, filename)
            vector_store.add_chunks(chunks)
            results.append({"file": filename, "chunks": len(chunks), "status": "ingested"})
        return results

    def new_session(self) -> str:
        """Generate a fresh session id and register it in SQLite."""
        session_id = str(uuid.uuid4())
        memory_store.create_session(session_id)
        return session_id

    def get_history(self, session_id: str, last_n: int = 50) -> list[dict]:
        return memory_store.get_history(session_id, last_n=last_n)

    def list_sessions(self) -> list[dict]:
        return memory_store.list_sessions()

    def delete_session(self, session_id: str) -> None:
        memory_store.delete_session(session_id)

    def kb_stats(self) -> dict:
        return vector_store.stats()

    # private helpers 

    async def _summarize_session(self, session_id: str) -> None:
        """Summarize the last N messages and persist the summary to SQLite."""
        history = memory_store.get_history(
            session_id, last_n=settings.summarize_after_turns
        )
        if not history:
            return

        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        )
        prompt = SUMMARIZE_PROMPT.format(history=history_text)

        try:
            client = ollama.AsyncClient(host=settings.ollama_base_url)
            response = await client.generate(
                model=settings.ollama_model, prompt=prompt
            )
            new_summary = response.response

            existing = memory_store.get_summary(session_id)
            if existing:
                new_summary = existing + "\n\n[Recent update]\n" + new_summary

            memory_store.save_summary(session_id, new_summary)
            log.info("[summarize] Session %s summary updated.", session_id)
        except Exception as exc:
            log.warning("[summarize] Failed for session %s: %s", session_id, exc)


# Module-level singleton — import this everywhere
rag_agent = RAGAgent()
