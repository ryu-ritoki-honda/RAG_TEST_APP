import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from pypdf import PdfReader
from pathlib import Path

from service.rag_service import get_chunks, get_answer
from embeddings_utils import embed_texts

from chunking import chunk_text, chunk_text_by_chars
from image_extraction import extract_text_and_images_from_pdf
from ocr_utils import ocr_image_file, extract_text_with_ocr_fallback

from store.document_store import DocumentStore

from repository_manager import (
    add_documents,
    load_repository_files,
    remove_file,
    clear_repository,
)

# =====================================================
# SESSION INIT
# =====================================================

if "doc_store" not in st.session_state:
    store = DocumentStore()
    store.load_kb()
    store.load_repo()
    st.session_state["doc_store"] = store


# =====================================================
# SIDEBAR - REPOSITORY
# =====================================================

with st.sidebar:
    st.header("Repository")

    repo_files = load_repository_files()

    if repo_files:
        st.write("Stored files:")
        for f in repo_files:
            c1, c2 = st.columns([4, 1])

            with c1:
                st.write(f)

            with c2:
                if st.button("❌", key=f"delete_{f}"):
                    remove_file(f)
                    st.rerun()

    if st.button("Clear Repository"):
        clear_repository()

        store = st.session_state["doc_store"]
        store.repo_documents = []
        store.repo_embeddings = None

        st.rerun()

    repo_upload = st.file_uploader(
        "Add files to repository",
        type=["txt", "csv", "pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="repo_upload",
    )

    if repo_upload:
        store = st.session_state["doc_store"]

        new_repo_files = []

        for uploaded_file in repo_upload:
            new_repo_files.append(uploaded_file)

        for uploaded_file in new_repo_files:

            chunks = []

            if uploaded_file.name.lower().endswith(".csv"):
                text = io.TextIOWrapper(uploaded_file, encoding="utf-8", errors="replace")
                df = pd.read_csv(text)

                for row in df.astype(str).itertuples(index=False, name=None):
                    joined = " ".join(v for v in row if v and v.strip())
                    if joined.strip():
                        chunks.append(joined)

            elif uploaded_file.name.lower().endswith(".pdf"):
                import tempfile, os

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    text_chunks, image_descriptions = extract_text_and_images_from_pdf(tmp_path)
                    ocr_chunks, _ = extract_text_with_ocr_fallback(tmp_path)
                    chunks = text_chunks + image_descriptions + ocr_chunks
                finally:
                    os.unlink(tmp_path)

            elif uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                import tempfile, os

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    chunks = ocr_image_file(tmp_path)
                finally:
                    os.unlink(tmp_path)

            else:
                text = uploaded_file.read().decode("utf-8", errors="replace")
                chunks = chunk_text(text, chunk_size=8, overlap=2)

            add_documents(uploaded_file.name, chunks)

        # refresh store
        store.load_repo()
        st.rerun()


# =====================================================
# MAIN UI
# =====================================================

st.title("Embedding Visualizer")

text_input = st.text_area("Sentences")

if st.button("Generate Embeddings"):

    texts = [t.strip() for t in text_input.splitlines() if t.strip()]

    if len(texts) < 2:
        st.warning("Need at least 2 sentences")
        st.stop()

    embeddings = embed_texts(texts)

    X = np.asarray(embeddings, dtype=np.float64)

    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(X)

    df = pd.DataFrame({
        "Sentence": texts,
        "X": coords[:, 0],
        "Y": coords[:, 1],
    })

    fig = px.scatter(df, x="X", y="Y", hover_data=["Sentence"])
    st.plotly_chart(fig, use_container_width=True)

    sim = cosine_similarity(X)
    st.dataframe(pd.DataFrame(sim, index=texts, columns=texts).round(3))


# =====================================================
# UPLOAD DOCS (separate from repo)
# =====================================================

uploaded = st.file_uploader(
    "Upload documents",
    type=["txt", "csv", "pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
    accept_multiple_files=True
)

if uploaded:

    new_documents = []

    for f in uploaded:

        if f.name.endswith(".csv"):
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            for row in df.astype(str).itertuples(index=False, name=None):
                new_documents.append(" ".join(v for v in row if v and v.strip()))

        elif f.name.endswith(".pdf"):
            import tempfile, os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.getbuffer())
                tmp_path = tmp.name

            try:
                text_chunks, image_desc = extract_text_and_images_from_pdf(tmp_path)
                ocr_chunks, _ = extract_text_with_ocr_fallback(tmp_path)
                new_documents += text_chunks + image_desc + ocr_chunks
            finally:
                os.unlink(tmp_path)

        else:
            new_documents.append(f.read().decode("utf-8", errors="replace"))

    store = st.session_state["doc_store"]
    store.add_uploads(new_documents)

    st.rerun()


# =====================================================
# QUESTION ANSWERING
# =====================================================

question = st.text_input("Ask a question")

sort_mode = st.selectbox(
    "Sort retrieved chunks by",
    ["relevance", "semantic", "bm25", "mmr"]
)

if st.button("Ask"):

    store = st.session_state["doc_store"]

    documents = store.get_documents()
    doc_embeddings = store.get_embeddings()

    if len(documents) == 0:
        st.warning("No documents available")
        st.stop()

    if not question.strip():
        st.warning("Enter a question")
        st.stop()

    chunks, query_embedding = get_chunks(
        question,
        documents,
        doc_embeddings,
        sort_mode=sort_mode,
    )

    timings = {}

    start = time.perf_counter()
    semantic_chunks, _ = get_chunks(
        question,
        documents,
        doc_embeddings,
        sort_mode="semantic",
    )
    timings["Semantic"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    bm25_chunks, _ = get_chunks(
        question,
        documents,
        doc_embeddings,
        sort_mode="bm25",
    )
    timings["BM25"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    relevance_chunks, _ = get_chunks(
        question,
        documents,
        doc_embeddings,
        sort_mode="relevance",
    )
    timings["Relevance"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    mmr_chunks, _ = get_chunks(
        question,
        documents,
        doc_embeddings,
        sort_mode="mmr",
    )
    timings["MMR"] = (time.perf_counter() - start) * 1000

    with st.expander("Retrieved Chunks", expanded=True):
        for doc, score, sim, bm25, mmr in chunks:
            st.markdown(f"""
            <div style="display:flex; gap:20px;">
                <div style="padding:12px; border-radius:10px; color:#00ff99;">
                    Semantic similarity<br>
                    <b>{sim:.3f}</b>
                </div>
                <div style="padding:12px; border-radius:10px; color:#66b3ff;">
                    BM25 score<br>
                    <b>{bm25:.3f}</b>
                </div>      
                <div style="padding:12px; border-radius:10px; color:#ff6666;">
                    Relevance score<br>
                    <b>{score:.3f}</b>
                </div>
                <div style="padding:12px; border-radius:10px; color:#ffcc00;">
                    MMR score<br>
                    <b>{mmr:.3f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write(doc)
            st.divider()

    answer = get_answer(question, chunks)

    st.subheader("Answer")
    st.write(answer)

    # =================================================
    # VISUALIZATION DASHBOARD
    # =================================================

    st.header("Retrieval Visualization")

    document_embeddings = np.asarray(doc_embeddings, dtype=np.float64)
    query_embedding = np.asarray(query_embedding, dtype=np.float64)

    all_embeddings = np.vstack([document_embeddings, query_embedding])

    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(all_embeddings)

    retrieved_docs = {doc for doc, *_ in chunks}

    # =================================================
    # 1. PCA Projection
    # =================================================

    st.subheader("1. PCA Projection")

    plot_df = pd.DataFrame({
        "Sentence": documents + [question],
        "X": coords[:, 0],
        "Y": coords[:, 1],
    })

    plot_df["Type"] = [
        "Retrieved" if doc in retrieved_docs else "Document"
        for doc in documents
    ] + ["Question"]

    fig = px.scatter(
        plot_df,
        x="X",
        y="Y",
        color="Type",
        hover_data=["Sentence"],
        title="Embedding Space"
    )

    fig.update_traces(marker=dict(size=10))

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # 2. Retrieval Pipeline
    # =================================================

    st.subheader("2. Retrieval Pipeline")

    pipeline_df = pd.DataFrame({
        "Stage": [
            "All Documents",
            "Candidate Pool",
            "Returned Chunks"
        ],
        "Count": [
            len(documents),
            min(50, len(documents)),
            len(chunks)
        ]
    })

    fig = px.bar(
        pipeline_df,
        x="Stage",
        y="Count",
        text="Count",
        title="Retrieval Pipeline"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # 3. Retrieved Chunk Scores
    # =================================================

    st.subheader("3. Retrieved Chunk Scores")

    score_df = pd.DataFrame({
        "Chunk": [f"Chunk {i+1}" for i in range(len(chunks))],
        "Semantic": [c[2] for c in chunks],
        "BM25": [c[3] for c in chunks],
        "Relevance": [c[1] for c in chunks],
        "MMR": [c[4] for c in chunks],
    })

    score_long = score_df.melt(
        id_vars="Chunk",
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        score_long,
        x="Chunk",
        y="Score",
        color="Metric",
        barmode="group",
        title="Comparison of Retrieval Scores"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # 4. Semantic vs BM25
    # =================================================

    st.subheader("4. Semantic vs BM25")

    scatter_df = pd.DataFrame({
        "Semantic": [c[2] for c in chunks],
        "BM25": [c[3] for c in chunks],
        "Relevance": [c[1] for c in chunks],
        "Chunk": [f"Chunk {i+1}" for i in range(len(chunks))]
    })

    fig = px.scatter(
        scatter_df,
        x="Semantic",
        y="BM25",
        color="Relevance",
        size="Relevance",
        hover_name="Chunk",
        title="Semantic Similarity vs BM25"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # 5. Similarity Heatmap
    # =================================================

    st.subheader("5. Similarity Between Retrieved Chunks")

    retrieved_embeddings = []

    for doc, *_ in chunks:
        idx = documents.index(doc)
        retrieved_embeddings.append(doc_embeddings[idx])

    retrieved_embeddings = np.asarray(retrieved_embeddings)

    heat = cosine_similarity(retrieved_embeddings)

    heat_df = pd.DataFrame(
        heat,
        index=[f"C{i+1}" for i in range(len(chunks))],
        columns=[f"C{i+1}" for i in range(len(chunks))]
    )

    fig = px.imshow(
        heat_df,
        text_auto=".2f", # pyright: ignore[reportArgumentType]
        color_continuous_scale="Viridis",
        title="Cosine Similarity Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # 6. Relevance Distribution (Entire Corpus)
    # =================================================

    st.subheader("6. Distribution of Semantic Scores")

    semantic_scores = np.dot(
        document_embeddings,
        query_embedding
    ) / (
        np.linalg.norm(document_embeddings, axis=1)
        * np.linalg.norm(query_embedding)
    )

    dist_df = pd.DataFrame({
        "Semantic Score": semantic_scores
    })

    fig = px.histogram(
        dist_df,
        x="Semantic Score",
        nbins=30,
        title="Semantic Similarity Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # 7. MMR Selection Order
    # =================================================

    st.subheader("7. MMR Selection Order")

    mmr_df = pd.DataFrame({
        "Selection": range(1, len(chunks)+1),
        "MMR Score": [c[4] for c in chunks]
    })

    fig = px.line(
        mmr_df,
        x="Selection",
        y="MMR Score",
        markers=True,
        title="MMR Selection Order"
    )

    st.plotly_chart(fig, use_container_width=True)

    # =================================================
    # 8. Retrieval Method Comparison
    # =================================================

    st.subheader("Retrieval Method Comparison")
    compare_df = pd.DataFrame({
        "Semantic": [
            c[0][:60]
            for c in semantic_chunks
        ],
        "BM25": [
            c[0][:60]
            for c in bm25_chunks
        ],
        "Relevance": [
            c[0][:60]
            for c in relevance_chunks
        ],
        "MMR": [
            c[0][:60]
            for c in mmr_chunks
        ]
    })

    st.dataframe(compare_df)

    # =================================================
    # 9. Retrieval Timing
    # =================================================

    st.subheader("9. Retrieval Time")

    timing_df = pd.DataFrame({
        "Method": list(timings.keys()),
        "Time (ms)": list(timings.values())
    })

    fig = px.bar(
        timing_df,
        x="Method",
        y="Time (ms)",
        text="Time (ms)",
        title="Retrieval Speed Comparison"
    )

    fig.update_traces(
        texttemplate="%{y:.1f} ms",
        textposition="outside"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        timing_df.style.format({
            "Time (ms)": "{:.2f}"
        })
    )