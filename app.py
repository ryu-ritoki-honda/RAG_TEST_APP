import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import load_knowledge_base
import umap
from pypdf import PdfReader
from rag import retrieve, answer_question
from embeddings_utils import embed_texts
from chunking import (
    chunk_text,
    chunk_text_by_chars,
)

if "uploaded_documents" not in st.session_state:
    st.session_state["uploaded_documents"] = []

if "uploaded_doc_embeddings" not in st.session_state:
    st.session_state["uploaded_doc_embeddings"] = None

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = []

if "kb_documents" not in st.session_state:
    st.session_state["kb_documents"] = []

if "kb_doc_embeddings" not in st.session_state:
    st.session_state["kb_doc_embeddings"] = None

if not st.session_state["kb_documents"]:

    kb_documents = load_knowledge_base()

    st.session_state["kb_documents"] = kb_documents

    if kb_documents:
        st.session_state["kb_doc_embeddings"] = embed_texts(
            kb_documents
        )

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
    type=["txt", "csv", "pdf"],
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
            elif uploaded_file.name.lower().endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"
                chunks = chunk_text_by_chars(
                    text,
                    chunk_size=1000,
                    overlap=200
                )
                new_documents.extend(chunks)
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

    # --------------------------
    # Get all documents
    # --------------------------
    kb_documents = st.session_state.get(
        "kb_documents",
        []
    )

    uploaded_documents = st.session_state.get(
        "uploaded_documents",
        []
    )

    documents = (
        kb_documents +
        uploaded_documents
    )

    # --------------------------
    # Get all embeddings
    # --------------------------
    kb_embeddings = st.session_state.get(
        "kb_doc_embeddings"
    )

    uploaded_embeddings = st.session_state.get(
        "uploaded_doc_embeddings"
    )

    embedding_dim = 1536

    if kb_embeddings is None:
        kb_embeddings = np.empty((0, embedding_dim))

    if uploaded_embeddings is None:
        uploaded_embeddings = np.empty((0, embedding_dim))

    doc_embeddings = np.vstack([
        kb_embeddings,
        uploaded_embeddings
    ])

    # --------------------------
    # Validation
    # --------------------------
    if len(documents) == 0:
        st.warning(
            "No documents available. "
            "Add files to the knowledge_base folder "
            "or upload documents."
        )
        st.stop()

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # --------------------------
    # Retrieval
    # --------------------------
    chunks, query_embedding = retrieve(
        question,
        documents,
        doc_embeddings,
    )

    st.subheader("Retrieved Chunks")

    for doc, score in chunks:
        st.write(f"Score: {score:.3f}")
        st.write(doc)
        st.divider()

    # --------------------------
    # Generate answer
    # --------------------------
    answer = answer_question(
        question,
        chunks
    )

    st.subheader("Answer")
    st.write(answer)

    # --------------------------
    # Embedding visualization
    # --------------------------
    document_embeddings = np.asarray(
        doc_embeddings,
        dtype=np.float64
    )

    texts = documents

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float64
    )

    all_embeddings = np.vstack([
        document_embeddings,
        query_embedding
    ])

    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )

    all_coords = np.asarray(
        reducer.fit_transform(all_embeddings),
        dtype=np.float64
    )

    if all_coords.ndim != 2 or all_coords.shape[1] < 2:
        st.warning(
            "Embedding projection did not return two dimensions."
        )
        st.stop()

    # --------------------------
    # Build dataframe
    # --------------------------
    plot_df = pd.DataFrame({
        "Sentence": texts + [question],
        "X": all_coords[:, 0],
        "Y": all_coords[:, 1]
    })

    retrieved_docs = [
        doc
        for doc, score in chunks
    ]

    plot_df["Type"] = "Document"

    for i, text in enumerate(texts):
        if text in retrieved_docs:
            plot_df.loc[i, "Type"] = "Retrieved"

    plot_df.loc[
        len(plot_df) - 1,
        "Type"
    ] = "Question"

    # --------------------------
    # Plot
    # --------------------------
    fig = px.scatter(
        plot_df,
        x="X",
        y="Y",
        color="Type",
        hover_data=["Sentence"],
        title="Embedding Space + Question"
    )

    fig.update_traces(
        marker=dict(size=12)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )