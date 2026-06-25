from pathlib import Path
import numpy as np

from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Text,
    select,
)

from embeddings_utils import embed_texts

# =====================================================
# Paths
# =====================================================

Path("repository").mkdir(
    exist_ok=True
)

DB_PATH = "sqlite:///repository/documents.db"

EMBEDDING_DIM = 1536


# =====================================================
# Database
# =====================================================

engine = create_engine(DB_PATH)

metadata = MetaData()

documents_table = Table(
    "documents",
    metadata,

    Column(
        "id",
        Integer,
        primary_key=True
    ),

    Column(
        "filename",
        String
    ),

    Column(
        "chunk_text",
        Text
    ),

    Column(
        "source",
        String
    )
)

metadata.create_all(engine)

# =====================================================
# Add documents
# =====================================================

def add_documents(
    filename: str,
    chunks: list[str],
):
    """
    Save chunks into SQLite and FAISS.
    """

    if len(chunks) == 0:
        return

    conn = engine.connect()

    embeddings = embed_texts(
        chunks
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    for chunk in chunks:
        conn.execute(
            documents_table.insert().values(
                filename=filename,
                chunk_text=chunk,
                source="database"
            )
        )

    conn.commit()
    conn.close()


# =====================================================
# Load chunks
# =====================================================

def load_repository_documents():
    """
    Return all chunk text.
    """

    conn = engine.connect()

    stmt = select(
        documents_table.c.chunk_text
    )

    rows = conn.execute(
        stmt
    ).fetchall()

    conn.close()

    return [
        row[0]
        for row in rows
    ]


# =====================================================
# Load filenames
# =====================================================

def load_repository_files():
    """
    Return unique filenames.
    """

    conn = engine.connect()

    stmt = select(
        documents_table.c.filename
    )

    rows = conn.execute(
        stmt
    ).fetchall()

    conn.close()

    filenames = sorted(
        set(
            row[0]
            for row in rows
        )
    )

    return filenames


# =====================================================
# Delete everything
# =====================================================

def clear_repository():

    conn = engine.connect()

    conn.execute(
        documents_table.delete()
    )

    conn.commit()
    conn.close()


# =====================================================
# Delete one file
# =====================================================

def remove_file(
    filename: str
):
    """
    Removes chunks belonging to one file.

    NOTE:
    SQLite rows are removed,
    but FAISS is rebuilt afterwards.
    """

    conn = engine.connect()

    conn.execute(
        documents_table.delete().where(
            documents_table.c.filename
            == filename
        )
    )

    conn.commit()
    conn.close()