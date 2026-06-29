from pathlib import Path
import pandas as pd
import io
from pypdf import PdfReader
from chunking import (
    chunk_text,
    chunk_text_by_chars,
)
from image_extraction import extract_text_and_images_from_pdf
from ocr_utils import extract_text_with_ocr_fallback


def load_knowledge_base(folder="knowledge_base"):
    documents = []

    folder = Path(folder)

    if not folder.exists():
        return documents

    for file in folder.iterdir():

        if file.suffix.lower() == ".txt":
            text = file.read_text(
                encoding="utf-8",
                errors="replace"
            )

            documents.extend(
                chunk_text(
                    text,
                    chunk_size=8,
                    overlap=2
                )
            )

        elif file.suffix.lower() == ".pdf":
            # Try to extract text and images, with OCR fallback for scanned PDFs
            text_chunks, image_descriptions = extract_text_and_images_from_pdf(str(file))
            ocr_chunks, ocr_used = extract_text_with_ocr_fallback(str(file))
            
            documents.extend(text_chunks)
            documents.extend(image_descriptions)
            documents.extend(ocr_chunks)

        elif file.suffix.lower() == ".csv":
            df = pd.read_csv(file)

            for row in df.astype(str).itertuples(
                index=False,
                name=None
            ):
                joined = " ".join(
                    value
                    for value in row
                    if value and value.strip()
                )

                if joined.strip():
                    documents.append(joined.strip())

    return documents