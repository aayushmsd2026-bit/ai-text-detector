"""Extract plain text from uploaded files."""

from __future__ import annotations

from pathlib import Path


def read_any(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".docx":
        import docx
        d = docx.Document(str(path))
        return "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
    if suffix == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    if suffix in {".txt", ".md", ".rtf", ""}:
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {suffix}. Use .txt, .md, .docx or .pdf.")
