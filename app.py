import io
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from knowledge_base import load_knowledge_base
from sklearn.decomposition import PCA
from pypdf import PdfReader
from rag import retrieve, answer_question
from embeddings_utils import embed_texts
from chunking import (
    chunk_text,
    chunk_text_by_chars,
)
from image_extraction import extract_text_and_images_from_pdf
from ocr_utils import ocr_image_file, extract_text_with_ocr_fallback
from repository_manager import (
    add_documents,
    load_repository_documents,
    load_repository_files,
    remove_file,
    clear_repository,
)

if "uploaded_documents" not in st.session_state:
    st.session_state["uploaded_documents"] = []

if "uploaded_doc_embeddings" not in st.session_state:
    st.session_state["uploaded_doc_embeddings"] = None

if "processed_uploads" not in st.session_state:
    st.session_state["processed_uploads"] = []

if "processed_repo_uploads" not in st.session_state:
    st.session_state["processed_repo_uploads"] = []

if "kb_documents" not in st.session_state:
    st.session_state["kb_documents"] = []

if "kb_doc_embeddings" not in st.session_state:
    st.session_state["kb_doc_embeddings"] = None

if "repository_documents" not in st.session_state:
    st.session_state["repository_documents"] = []

if "repository_doc_embeddings" not in st.session_state:
    st.session_state["repository_doc_embeddings"] = None

if not st.session_state["kb_documents"]:

    kb_documents = load_knowledge_base()

    st.session_state["kb_documents"] = kb_documents

    if kb_documents:
        st.session_state["kb_doc_embeddings"] = embed_texts(
            kb_documents
        )

if not st.session_state["repository_documents"]:

    repository_documents = load_repository_documents()

    st.session_state["repository_documents"] = repository_documents

    if repository_documents:
        st.session_state["repository_doc_embeddings"] = embed_texts(
            repository_documents
        )


# =====================================================
# Repository Manager
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
        st.session_state["processed_repo_uploads"] = []
        st.session_state["repository_documents"] = []
        st.session_state["repository_doc_embeddings"] = None
        st.rerun()

    repo_upload = st.file_uploader(
        "Add files to repository",
        type=["txt", "csv", "pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        key="repo_upload",
    )

    if repo_upload:
        processed_repo = st.session_state["processed_repo_uploads"]
        new_repo_files = []
        
        for uploaded_file in repo_upload:
            file_key = (uploaded_file.name, uploaded_file.size)
            if file_key not in processed_repo:
                new_repo_files.append(uploaded_file)
                processed_repo.append(file_key)
        
        if new_repo_files:
            for uploaded_file in new_repo_files:

                chunks = []

                if uploaded_file.name.lower().endswith(".csv"):
                    text = io.TextIOWrapper(
                        uploaded_file,
                        encoding="utf-8",
                        errors="replace",
                    )

                    df = pd.read_csv(text)

                    for row in df.astype(str).itertuples(index=False, name=None):
                        joined = " ".join(
                            value
                            for value in row
                            if value and value.strip()
                        )

                        if joined.strip():
                            chunks.append(joined)

                elif uploaded_file.name.lower().endswith(".pdf"):
                    # Save temporarily to process with image extraction and OCR fallback
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    
                    try:
                        # Extract text and images
                        text_chunks, image_descriptions = extract_text_and_images_from_pdf(tmp_path)
                        # Also try OCR for scanned PDFs
                        ocr_chunks, ocr_used = extract_text_with_ocr_fallback(tmp_path)
                        chunks = text_chunks + image_descriptions + ocr_chunks
                    except Exception as e:
                        st.warning(f"Error processing {uploaded_file.name}: {e}. Attempting text extraction only.")
                        try:
                            # Fallback to text only
                            reader = PdfReader(uploaded_file)
                            text = ""
                            for page in reader.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    text += page_text + "\n"
                            chunks = chunk_text_by_chars(
                                text,
                                chunk_size=1000,
                                overlap=200,
                            )
                        except Exception as e2:
                            st.error(f"Could not extract text from {uploaded_file.name}: {e2}")
                            chunks = []
                    finally:
                        import os
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass

                elif uploaded_file.name.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                    # Handle image files with OCR
                    import tempfile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name
                    
                    try:
                        chunks = ocr_image_file(tmp_path)
                        if not chunks:
                            st.warning(f"Could not extract text from image {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error processing image {uploaded_file.name}: {e}")
                        chunks = []
                    finally:
                        import os
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass

                else:
                    text = uploaded_file.read().decode(
                        "utf-8",
                        errors="replace",
                    )

                    chunks = chunk_text(
                        text,
                        chunk_size=8,
                        overlap=2,
                    )

                add_documents(
                    uploaded_file.name,
                    chunks,
                )

            # Reload repository documents and embeddings into session state
            repository_documents = load_repository_documents()
            st.session_state["repository_documents"] = repository_documents
            
            if repository_documents:
                repository_embeddings = embed_texts(repository_documents)
                st.session_state["repository_doc_embeddings"] = repository_embeddings

            st.rerun()


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

    reducer = PCA(n_components=2)
    coords = reducer.fit_transform(X)
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
    type=["txt", "csv", "pdf", "jpg", "jpeg", "png", "bmp", "tiff"],
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
                # Save temporarily to process with image extraction and OCR fallback
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                try:
                    # Extract text and images
                    text_chunks, image_descriptions = extract_text_and_images_from_pdf(tmp_path)
                    # Also try OCR for scanned PDFs
                    ocr_chunks, ocr_used = extract_text_with_ocr_fallback(tmp_path)
                    chunks = text_chunks + image_descriptions + ocr_chunks
                except Exception as e:
                    st.warning(f"Error processing {uploaded_file.name}: {e}. Attempting text extraction only.")
                    try:
                        # Fallback to text only
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
                    except Exception as e2:
                        st.error(f"Could not extract text from {uploaded_file.name}: {e2}")
                        chunks = []
                finally:
                    import os
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
                
                new_documents.extend(chunks)
                
            elif uploaded_file.name.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                # Handle image files with OCR
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name
                
                try:
                    chunks = ocr_image_file(tmp_path)
                    if chunks:
                        new_documents.extend(chunks)
                    else:
                        st.warning(f"Could not extract text from image {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing image {uploaded_file.name}: {e}")
                finally:
                    import os
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
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

    repository_documents = st.session_state.get(
        "repository_documents",
        []
    )

    documents = (
        kb_documents +
        repository_documents +
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

    repository_embeddings = st.session_state.get(
        "repository_doc_embeddings"
    )

    if repository_embeddings is None:
        repository_embeddings = np.empty((0, embedding_dim))

    doc_embeddings = np.vstack([
        kb_embeddings,
        repository_embeddings,
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

    with st.expander("📄 Retrieved Chunks", expanded=True):
        for doc, score, sim, bm25 in chunks:
            st.markdown(f"""
            **Score:** {score:.3f}  
            **Semantic similarity:** {sim:.3f}  
            **Keyword match (BM25):** {bm25:.3f}  

            ---
            """)
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

    reducer = PCA(n_components=2)

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
        for doc, score, sim, bm25 in chunks
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