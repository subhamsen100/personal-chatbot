# Personal Knowledge Base AI

A fully offline AI assistant that answers questions strictly from your own documents. No cloud APIs, no subscriptions, no data leaving your machine.

---

## Why I Built This

I wanted something that could actually read my notes, PDFs, and reports and answer questions about them without making things up. Most AI tools either hallucinate when they don't know something, or they send your data to some external server. I wanted neither of those things.

The idea was simple: give the model only what it needs from my documents and nothing else. If the answer isn't in my files, it should say so instead of guessing.

---

## How It Works

When you ask a question, the agent figures out what kind of question it is and routes it to the right tool:

- If you ask about something in your documents, it searches the vector database for relevant chunks and answers only from those.
- If you ask for a calculation, it runs it through a safe math evaluator.
- If you ask what time or date it is, it checks the system clock.
- If you are just chatting or referring to something said earlier in the conversation, it answers directly from the conversation history.

The model never makes things up for document questions. If your documents don't have the answer, it says exactly that.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Agent framework | Google ADK | Handles tool routing and the agent loop cleanly |
| LLM | Ollama (qwen3-vl:8b) | Runs locally, no API key needed, VLM capable |
| Embeddings | nomic-embed-text via Ollama | Also local, decent quality for semantic search |
| Vector store | FAISS | Simple, fast, no server to run |
| Chat memory | SQLite | Lightweight, built into Python, survives restarts |
| Backend | FastAPI | Clean async API, easy file upload handling |
| UI | Streamlit | Quick to build, good enough for a personal tool |

---

## Project Structure

```
personal-chatbot/
    config.py
    main.py
    app.py
    requirements.txt
    .env.example
    ingestion/
        loader.py
        chunker.py
        embedder.py
    storage/
        vector_store.py
        memory_store.py
    agent/
        ollama_llm.py
        tools.py
        prompts.py
        rag_agent.py
    data/
        docs/
        indexes/
    db/
        chat_memory.db
```

---

## File Breakdown

### config.py

Holds all settings using pydantic-settings so everything is configurable through a `.env` file. Model names, chunk sizes, file paths, port numbers. It also creates the `data/` and `db/` directories on first import so you never have to do that manually.

### ingestion/loader.py

Handles reading documents. Supports PDF (using PyMuPDF which handles complex layouts better than most), Word documents, Markdown files, and plain text. Returns the raw text content from any supported file.

### ingestion/chunker.py

Takes the raw text from a document and splits it into smaller overlapping pieces. Uses LangChain's RecursiveCharacterTextSplitter which tries to split on paragraph breaks first, then line breaks, then sentences. The overlap between chunks means context is not lost at the boundaries.

### ingestion/embedder.py

Converts text into vectors using Ollama's nomic-embed-text model. These vectors are what makes semantic search possible. Two similar sentences will have vectors that are close to each other in space, even if they use different words.

### storage/vector_store.py

Wraps FAISS to store and search the document vectors. When you ingest a document, the chunks and their vectors go in here. When you ask a question, your question gets embedded and the store finds the most similar chunks. Everything is saved to disk so you don't lose your index when the server restarts.

### storage/memory_store.py

Handles conversation history using raw SQLite. Stores sessions, messages, and a running summary per session. The summary is used to keep context when a conversation gets long enough that older messages would not fit in the context window anymore.

### agent/ollama_llm.py

This is the bridge between Google ADK and Ollama. ADK was built around Gemini and expects models to follow a specific interface. Since we are using Ollama locally, this file implements that interface manually by translating ADK's message format into what Ollama expects and translating the response back. It also handles converting tool definitions and tool call results between the two formats. Without this, ADK would not know how to talk to Ollama at all.

### agent/tools.py

Defines the three tools the agent can call:

**retrieve_from_knowledge_base** searches the FAISS index for relevant document chunks. It only gets called for document questions. Returns formatted text with source references so the model can cite where the answer came from.

**calculate** evaluates math expressions safely. It parses the expression using Python's AST module instead of raw eval, so only math operations and standard math functions are allowed. Nothing arbitrary can run through it.

**get_current_datetime** returns the current local date and time. Simple but necessary since the model has no internal clock.

### agent/prompts.py

Contains the system prompt that tells the model how to behave. The main thing it does is give the model a clear routing table: which tool to use for which type of question, and when to answer directly without any tool. The grounding rules for document answers are also here, specifically that the model must only use what the retrieval tool returns and must say it cannot find the answer if nothing relevant comes back.

### agent/rag_agent.py

The main orchestrator. Creates the ADK agent with all three tools and a dynamic system prompt that injects a conversation summary when one exists. The `chat()` method sends a message through the ADK runner, saves the exchange to SQLite, and triggers background summarization every 16 messages. Also handles document ingestion by calling the loader, chunker, and vector store in sequence.

### main.py

The FastAPI backend. Exposes REST endpoints for chatting, ingesting documents, managing sessions, and checking knowledge base stats. The Streamlit UI talks to this. Running them as separate processes means the UI can be restarted without losing the agent state.

### app.py

The Streamlit frontend. Has a sidebar for uploading documents and switching between sessions. The main area is a standard chat interface. Talks to the FastAPI backend over HTTP using httpx.

---

## Setup

Copy the example env file and adjust if needed:

```bash
cp .env.example .env
```

Create and activate the virtual environment:

```bash
python3 -m venv chatenv
source chatenv/bin/activate
pip install -r requirements.txt
```

Make sure Ollama is running and the required models are pulled:

```bash
ollama pull qwen3-vl:8b
ollama pull nomic-embed-text
```

---

## Running

You need two terminals.

Terminal 1, the backend:

```bash
source chatenv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2, the UI:

```bash
source chatenv/bin/activate
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Using It

1. Upload a document from the sidebar. PDF, DOCX, Markdown, and plain text are all supported.
2. Create a new session and start asking questions about your document.
3. The agent will retrieve the relevant parts and answer from them. It will cite the source file and chunk number at the end of each answer.
4. If you ask something that is not in your documents, it will tell you instead of making something up.
5. Math questions, date questions, and general conversation all work without needing any documents.

---

## Notes

The agent uses an in-memory ADK session for the active conversation and SQLite for persistence across server restarts. If you restart the backend, the conversation history is still in SQLite but the ADK session is new. Long conversations are summarized automatically and that summary gets injected into the system prompt so the model keeps context even after the session is fresh.

The LiteLLM library is not used. ADK's built-in LiteLLM support requires an extra install and adds complexity. Instead there is a custom `OllamaLlm` class that implements ADK's `BaseLlm` interface directly. It handles message format conversion, tool schema conversion, and streaming.
