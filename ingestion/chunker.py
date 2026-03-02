"""Text chunking using LangChain's RecursiveCharacterTextSplitter."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks.

    Returns a list of dicts with keys: text, source, chunk_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    raw_chunks = splitter.split_text(text)
    return [
        {"text": chunk, "source": source, "chunk_id": i}
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]
