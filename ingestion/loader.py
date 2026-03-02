"""Document loaders for PDF, Markdown, DOCX, and plain text."""

from pathlib import Path
from typing import Generator

import fitz  # pymupdf
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".docx", ".txt", ".text"}


def load_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (handles complex layouts)."""
    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(pages)


def load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_docx(path: Path) -> str:
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_document(path: Path) -> str:
    """Load a single document and return its text content."""
    suffix = path.suffix.lower()
    loaders = {
        ".pdf": load_pdf,
        ".md": load_markdown,
        ".markdown": load_markdown,
        ".docx": load_docx,
        ".txt": load_text,
        ".text": load_text,
    }
    if suffix not in loaders:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loaders[suffix](path)


def load_directory(dir_path: Path) -> Generator[tuple[str, str], None, None]:
    """Yield (filename, text_content) for every supported document in a directory."""
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                content = load_document(file_path)
                if content.strip():
                    yield file_path.name, content
            except Exception as exc:
                print(f"[loader] Skipping {file_path.name}: {exc}")
