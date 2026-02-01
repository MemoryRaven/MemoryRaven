from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PdfText:
    text: str
    pages: int | None = None


def extract_text_pymupdf(pdf_path: str) -> PdfText:
    """Fast local PDF text extraction.

    This is a *fallback* when you can't run GROBID.
    For scholarly PDFs, prefer GROBID for section structure + references.
    """

    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF not installed. pip install 'aks[pdf]' ") from e

    doc = fitz.open(pdf_path)
    parts: list[str] = []
    for page in doc:
        parts.append(page.get_text("text"))
    return PdfText(text="\n\n".join(parts), pages=doc.page_count)
