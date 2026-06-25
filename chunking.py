from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 8,
    overlap: int = 2,
) -> List[str]:
    """
    Split text into overlapping chunks of lines.

    Example:
        chunk_size=8
        overlap=2

        Chunk 1: lines 1-8
        Chunk 2: lines 7-14
        Chunk 3: lines 13-20
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    chunks = []

    step = max(1, chunk_size - overlap)

    for i in range(0, len(lines), step):
        chunk_lines = lines[i:i + chunk_size]

        if not chunk_lines:
            continue

        chunk = "\n".join(chunk_lines)
        chunks.append(chunk)

    return chunks


def chunk_text_by_chars(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping character chunks.
    Better suited for PDFs and long documents.
    """

    chunks = []

    step = max(1, chunk_size - overlap)

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks