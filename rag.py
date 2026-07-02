import numpy as np
from embeddings_utils import embed_text, embed_texts
from aoai_client import get_aoai_client
from document_store import documents as default_documents
from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi


# -----------------------------
# Helpers
# -----------------------------

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
    selected_mmr_scores = []
    candidates = list(range(len(scores)))

    normed = doc_embeddings / (
        np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12
    )

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
        selected_mmr_scores.append(best_score)
        candidates.remove(best)

    return selected, selected_mmr_scores


def build_doc_embeddings(documents: List[str]) -> np.ndarray:
    return embed_texts(documents)


# -----------------------------
# Core RAG retrieval
# -----------------------------

def retrieve(
    query: str,
    documents: Optional[List[str]] = None,
    doc_embeddings: Optional[np.ndarray] = None,
    k: int = 15,
    sort_mode: str = "relevance"
) -> Tuple[List[Tuple[str, float, float, float, float]], np.ndarray]:

    if documents is None:
        documents = default_documents

    if len(documents) == 0:
        return [], embed_text(query)

    if doc_embeddings is None:
        doc_embeddings = build_doc_embeddings(documents)

    query_lower = query.lower()

    # -----------------------------
    # embeddings
    # -----------------------------
    query_embedding = embed_text(query)

    doc_norms = np.linalg.norm(doc_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding) + 1e-12

    sims = (doc_embeddings @ query_embedding) / (doc_norms * query_norm + 1e-12)

    # -----------------------------
    # BM25
    # -----------------------------
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query_lower.split()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))

    # -----------------------------
    # normalization
    # -----------------------------
    eps = 1e-12

    sims_norm = (sims - np.min(sims)) / (np.max(sims) - np.min(sims) + eps)
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + eps)

    # -----------------------------
    # hybrid score
    # -----------------------------
    w_sim, w_bm25 = get_hybrid_weights(query)
    final_score = w_sim * sims_norm + w_bm25 * bm25_norm


    # =====================================================
    # MODE 1: PURE SORTING MODES
    # =====================================================

    if sort_mode == "semantic":
        idx = np.argsort(sims_norm)[::-1]

    elif sort_mode == "bm25":
        idx = np.argsort(bm25_norm)[::-1]

    elif sort_mode == "relevance":
        idx = np.argsort(final_score)[::-1]

    # =====================================================
    # MODE 2: MMR (selection mode)
    # =====================================================
    elif sort_mode == "mmr":
        candidate_k = min(50, len(documents))
        top_idx = np.argsort(final_score)[::-1][:candidate_k]

        mmr_selected, mmr_scores = mmr_select(
            doc_embeddings[top_idx],
            final_score[top_idx],
            k=k
        )

        selected_indices = [top_idx[i] for i in mmr_selected]

        return [
            (
                documents[i],
                float(final_score[i]),
                float(sims_norm[i]),
                float(bm25_norm[i]),
                float(mmr_scores[j])
            )
            for j, i in enumerate(selected_indices)
        ], query_embedding

    else:
        idx = np.argsort(final_score)[::-1]

    # -----------------------------
    # standard output path
    # -----------------------------
    idx = idx[:k]

    return [
        (
            documents[i],
            float(final_score[i]),
            float(sims_norm[i]),
            float(bm25_norm[i]),
            0.0  # no MMR in non-MMR modes
        )
        for i in idx
    ], query_embedding

# -----------------------------
# Answering
# -----------------------------

def answer_question(question, chunks):

    context = "\n".join(
        f"[Chunk {i+1}] (score={score:.3f})\n{doc}\n"
        for i, (doc, score, *_) in enumerate(chunks)
    )

    client = get_aoai_client()

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": f"{context}\n\nQuestion:\n{question}"}
        ],
    )

    return response.choices[0].message.content