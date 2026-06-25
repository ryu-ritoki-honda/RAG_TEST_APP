from pathlib import Path
import pandas as pd
import io
from pypdf import PdfReader
from chunking import (
    chunk_text,
    chunk_text_by_chars,
)


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
            reader = PdfReader(str(file))

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            documents.extend(
                chunk_text_by_chars(
                    text,
                    chunk_size=1000,
                    overlap=200
                )
            )

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