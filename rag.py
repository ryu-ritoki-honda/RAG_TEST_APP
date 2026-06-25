import numpy as np
from embeddings_utils import embed_text, embed_texts
from aoai_client import get_aoai_client
from document_store import documents as default_documents
from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi


def build_doc_embeddings(documents: List[str]) -> np.ndarray:
    return embed_texts(documents)


def retrieve(
    query: str,
    documents: Optional[List[str]] = None,
    doc_embeddings: Optional[np.ndarray] = None,
    k: int = 15,
) -> Tuple[List[Tuple[str, float]], np.ndarray]:
    # Use default documents if none were passed
    if documents is None:
        documents = default_documents

    # Build embeddings if none were passed
    if doc_embeddings is None:
        doc_embeddings = build_doc_embeddings(documents)

    # Embed the query
    query_embedding = embed_text(query)

    # Cosine similarity
    dots = doc_embeddings @ query_embedding
    norms = (
        np.linalg.norm(doc_embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )

    sims = dots / np.clip(norms, 1e-12, None)

    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()
    bm25_scores = np.array(
        bm25.get_scores(tokenized_query)
    )

    bm25_scores = bm25_scores / (
    np.max(bm25_scores) + 1e-12
    )

    sims = sims / (
        np.max(sims) + 1e-12
    )

    final_score = (
        0.7 * sims +
        0.3 * bm25_scores
    )
    # Hybrid score
    final_score = (
        0.7 * sims +
        0.3 * bm25_scores
    )

    # Get top-k indices
    idx = np.argsort(final_score)[::-1][:k]

    # Build results
    scores = [
        (documents[i], float(final_score[i]))
        for i in idx
    ]

    return scores, query_embedding


def answer_question(question, chunks):

    # Build context first
    context = ""

    for i, (doc, score) in enumerate(chunks):
        context += (
            f"[Chunk {i+1}] "
            f"(score={score:.3f})\n"
            f"{doc}\n\n"
        )

    # Debug
    print("\nRetrieved Chunks")
    print("=" * 50)

    for i, (doc, score) in enumerate(chunks):
        print(f"\nChunk {i+1}")
        print(f"Score: {score:.3f}")
        print(doc)

    prompt = f"""
Answer using ONLY the provided context.

If the answer requires combining information from multiple chunks,
reason through the clues step by step.

If the answer cannot be determined from the context, say:
"I cannot determine the answer from the provided context."

When answering:
1. Identify relevant chunks.
2. Combine information if necessary.
3. Cite chunk numbers.

Context:
{context}

Question:
{question}

Answer:
"""

    client = get_aoai_client()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content