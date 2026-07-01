import numpy as np
from embeddings_utils import embed_text, embed_texts
from aoai_client import get_aoai_client
from document_store import documents as default_documents
from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi


# -----------------------------
# Helpers
# -----------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def get_hybrid_weights(query: str):
    q = query.lower()

    if any(w in q for w in ["what is", "who is", "explain", "why", "how"]):
        return 0.85, 0.15

    if any(w in q for w in ["list", "all", "every", "count"]):
        return 0.5, 0.5

    if any(w in q for w in ["name", "keyword", "specific"]):
        return 0.3, 0.7

    return 0.7, 0.3


def mmr_select(doc_embeddings, scores, k=15, lambda_=0.7):
    selected = []
    candidates = list(range(len(scores)))

    # normalize embeddings for cosine reuse
    normed = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12)

    while len(selected) < k and candidates:
        best = None
        best_score = -1

        for i in candidates:
            relevance = scores[i]

            diversity = 0
            if selected:
                diversity = max(
                    np.dot(normed[i], normed[j])
                    for j in selected
                )

            mmr_score = lambda_ * relevance - (1 - lambda_) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best = i
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)

    return selected


def rerank_with_llm(query: str, docs: List[str]):
    """
    Lightweight LLM reranker (optional but high impact)
    """
    client = get_aoai_client()

    prompt = f"""
You are a ranking system.

Rank these documents by relevance to the query.
Return ONLY a JSON list of indices (best to worst).

Query:
{query}

Documents:
""" + "\n".join(f"{i}. {d}" for i, d in enumerate(docs))

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        import json

        content = response.choices[0].message.content or "[]"
        order = json.loads(content)

        return order
    except:
        return list(range(len(docs)))


# -----------------------------
# Core RAG retrieval
# -----------------------------

def build_doc_embeddings(documents: List[str]) -> np.ndarray:
    return embed_texts(documents)


def retrieve(
    query: str,
    documents: Optional[List[str]] = None,
    doc_embeddings: Optional[np.ndarray] = None,
    k: int = 15,
) -> Tuple[List[Tuple[str, float, float, float]], np.ndarray]:

    if documents is None:
        documents = default_documents

    if len(documents) == 0:
        return [], embed_text(query)

    if doc_embeddings is None:
        doc_embeddings = build_doc_embeddings(documents)

    query_lower = query.lower()

    aggregation_words = {
        "highest", "lowest", "maximum", "minimum", "average",
        "sum", "count", "top", "all", "every", "above", "below",
        "greater", "less", "at least", "or higher", "or lower"
    }

    if any(word in query_lower for word in aggregation_words):
        k = 50

    # -----------------------------
    # Stage 1: Embedding retrieval
    # -----------------------------
    query_embedding = embed_text(query)

    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding) + 1e-12

    sims = (doc_embeddings @ query_embedding) / (doc_norms * query_norm + 1e-12)

    # -----------------------------
    # Stage 2: BM25
    # -----------------------------
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query_lower.split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    # -----------------------------
    # Stage 3: normalization (stable)
    # -----------------------------
    eps = 1e-12

    sims = (sims - np.min(sims)) / (np.max(sims) - np.min(sims) + eps)
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + eps)

    # -----------------------------
    # Stage 4: adaptive hybrid scoring
    # -----------------------------
    w_sim, w_bm25 = get_hybrid_weights(query)
    final_score = w_sim * sims + w_bm25 * bm25_scores

    # -----------------------------
    # Stage 5: candidate pruning
    # -----------------------------
    candidate_k = min(50, len(documents))
    top_idx = np.argsort(final_score)[::-1][:candidate_k]

    candidate_docs = [documents[i] for i in top_idx]
    candidate_scores = final_score[top_idx]
    candidate_embeddings = doc_embeddings[top_idx]

    # -----------------------------
    # Stage 6: reranking (LLM)
    # -----------------------------
    try:
        order = rerank_with_llm(query, candidate_docs)
        top_idx = [top_idx[i] for i in order if i < len(top_idx)]
        candidate_scores = final_score[top_idx]
    except:
        pass

    # -----------------------------
    # Stage 7: diversity selection (MMR)
    # -----------------------------
    final_selected = mmr_select(
        doc_embeddings[top_idx],
        candidate_scores,
        k=k
    )

    selected_indices = [top_idx[i] for i in final_selected]

    scores = [
        (
            documents[i],
            float(final_score[i]),   # ranking score (used internally)
            float(sims[i]),          # semantic similarity
            float(bm25_scores[i])    # keyword score
        )
        for i in selected_indices
    ]

    return scores, query_embedding


# -----------------------------
# Answering (unchanged logic, cleaner formatting)
# -----------------------------

def answer_question(question, chunks):

    context = "\n".join(
        f"[Chunk {i+1}] (score={score:.3f})\n{doc}\n"
        for i, (doc, score, sim, bm25) in enumerate(chunks)
    )

    print("\nRetrieved Chunks")
    print("=" * 50)

    for i, (doc, score, sim, bm25) in enumerate(chunks):
        print(f"\nChunk {i+1}")
        print(f"Score: {score:.3f}")
        print(doc)

    prompt = f"""
Before answering:

- Use ONLY the provided context.
- Reject false assumptions explicitly.
- If missing, say "I cannot determine the answer from the provided context."
- Combine chunks when needed.
- If asked for ALL items, scan everything.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    client = get_aoai_client()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content