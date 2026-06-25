import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
import umap
from rag import retrieve, answer_question
from embeddings_utils import embed_texts, embed_text

def chunk_text(
    text: str,
    chunk_size: int = 8,
    overlap: int = 2,
):
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

if "uploaded_documents" not in st.session_state:
    st.session_state["uploaded_documents"] = []

if "uploaded_doc_embeddings" not in st.session_state:
    st.session_state["uploaded_doc_embeddings"] = None

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = []

# Use centralized embedding utilities (batching + cache)


# --------------------------
# Streamlit UI
# --------------------------
st.title("Embedding Visualizer")

st.write(
    """
    Enter one sentence per line.
    Similar sentences should appear closer together.
    """
)

text_input = st.text_area(
    "Sentences",
    height=250,
    placeholder="""Dog
Puppy
Cat
Kitten
I love Japanese food.
I enjoy sushi.
Quantum mechanics is fascinating."""
)

if st.button("Generate Embeddings"):

    texts = [
        line.strip()
        for line in text_input.splitlines()
        if line.strip()
    ]

    if len(texts) < 2:
        st.warning("Please enter at least two sentences.")
        st.stop()

    with st.spinner("Generating embeddings..."):
        embeddings = embed_texts(texts)

    X = np.asarray(embeddings, dtype=np.float64)
    st.session_state["texts"] = texts
    st.session_state["embeddings"] = X

    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )

    coords: np.ndarray = np.asarray(reducer.fit_transform(X), dtype=np.float64)
    st.session_state["coords"] = coords

    if coords.ndim != 2 or coords.shape[1] < 2:
        st.warning("Embedding projection did not return two dimensions.")
        st.stop()

    df = pd.DataFrame({
        "Sentence": texts,
        "X": coords[:, 0],
        "Y": coords[:, 1]
    })

    fig = px.scatter(
        df,
        x="X",
        y="Y",
        hover_data=["Sentence"],
        title="Embedding Space"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    st.dataframe(df)

    # --------------------------
    # Cosine Similarity
    # --------------------------

    similarities = cosine_similarity(X)

    sim_df = pd.DataFrame(
        similarities,
        index=texts,
        columns=texts
    )

    st.subheader("Cosine Similarity Matrix")
    st.dataframe(sim_df.round(3))

uploaded = st.file_uploader(
    "Upload documents",
    type=["txt", "csv"],
    accept_multiple_files=True
)

if uploaded:
    processed_uploads = st.session_state["processed_uploads"]
    new_documents = []
    new_files = []

    for uploaded_file in uploaded:
        file_key = (uploaded_file.name, uploaded_file.size)
        if file_key not in processed_uploads:
            new_files.append(uploaded_file)
            processed_uploads.append(file_key)

    if new_files:
        for uploaded_file in new_files:
            if uploaded_file.type == "text/csv" or uploaded_file.name.lower().endswith(".csv"):
                text = io.TextIOWrapper(uploaded_file, encoding="utf-8", errors="replace")
                df = pd.read_csv(text)
                # flatten CSV rows into lines
                for row in df.astype(str).itertuples(index=False, name=None):
                    joined = " ".join(value for value in row if value and value.strip())
                    if joined.strip():
                        new_documents.append(joined.strip())
            else:
                text = uploaded_file.read().decode(
                    "utf-8",
                    errors="replace"
                )

                chunks = chunk_text(
                    text,
                    chunk_size=8,
                    overlap=2
                )

                new_documents.extend(chunks)

        uploaded_documents = st.session_state.get("uploaded_documents", []) + new_documents
        st.session_state["uploaded_documents"] = uploaded_documents
        st.session_state["processed_uploads"] = processed_uploads

        with st.spinner("Embedding uploaded documents..."):
            uploaded_doc_embeddings = embed_texts(uploaded_documents)

        st.session_state["uploaded_doc_embeddings"] = uploaded_doc_embeddings

        st.success(
            f"Loaded {len(uploaded_documents)} documents from {len(processed_uploads)} file(s)."
        )
    elif uploaded:
        st.info("No new files to process. Existing uploads are already loaded.")

if st.button("Reset uploaded documents"):
    st.session_state["uploaded_documents"] = []
    st.session_state["uploaded_doc_embeddings"] = None
    st.session_state["processed_uploads"] = []
    st.success("Uploaded documents have been reset.")

question = st.text_input(
    "Ask a question"
)

if st.button("Ask"):
    uploaded_documents = st.session_state.get("uploaded_documents", [])
    uploaded_doc_embeddings = st.session_state.get("uploaded_doc_embeddings")

    if len(uploaded_documents) == 0:
        st.warning("Upload a txt file first.")
        st.stop()

    if uploaded_doc_embeddings is None:
        st.warning("Upload a txt file first.")
        st.stop()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # RAG retrieval using uploaded documents only
    chunks, query_embedding = retrieve(
        question,
        uploaded_documents,
        uploaded_doc_embeddings,
    )

    st.subheader("Retrieved Chunks")

    for chunk in chunks:
        st.write(chunk)

    answer = answer_question(
        question,
        chunks
    )

    st.subheader("Answer")
    st.write(answer)

    # Get current uploaded embeddings
    uploaded_documents = st.session_state.get("uploaded_documents", [])
    uploaded_doc_embeddings = st.session_state.get("uploaded_doc_embeddings")

    if uploaded_doc_embeddings is None or len(uploaded_documents) == 0:
        st.warning("Upload a txt file first.")
        st.stop()

    document_embeddings = np.asarray(uploaded_doc_embeddings, dtype=np.float64)
    texts = uploaded_documents

    # Add question embedding
    query_embedding = np.asarray(query_embedding, dtype=np.float64)
    all_embeddings = np.vstack([document_embeddings, query_embedding])

    # Re-run UMAP including question
    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )

    all_coords: np.ndarray = np.asarray(reducer.fit_transform(all_embeddings), dtype=np.float64)

    if all_coords.ndim != 2 or all_coords.shape[1] < 2:
        st.warning("Embedding projection did not return two dimensions.")
        st.stop()

    # Build dataframe
    plot_df = pd.DataFrame({
        "Sentence": texts + [question],
        "X": all_coords[:, 0],
        "Y": all_coords[:, 1]
    })

    retrieved_docs = [doc for doc, score in chunks]

    plot_df["Type"] = "Document"

    for i, text in enumerate(texts):
        if text in retrieved_docs:
            plot_df.loc[i, "Type"] = "Retrieved"

    plot_df.loc[
        len(plot_df) - 1,
        "Type"
    ] = "Question"

    # Plot
    fig = px.scatter(
        plot_df,
        x="X",
        y="Y",
        color="Type",
        hover_data=["Sentence"],
        title="Embedding Space + Question"
    )

    fig.update_traces(marker=dict(size=12))

    st.plotly_chart(
        fig,
        use_container_width=True
    )