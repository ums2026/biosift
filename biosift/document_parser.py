from __future__ import annotations

import io

from pypdf import PdfReader


def extract_document_text(file_bytes: bytes, filename: str) -> str:
    """Extract text from PDF, TXT, or Markdown protocol files."""
    lower = filename.lower()

    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()

    return file_bytes.decode("utf-8", errors="replace").strip()
