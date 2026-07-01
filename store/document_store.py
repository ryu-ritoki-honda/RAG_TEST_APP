import numpy as np
from embeddings_utils import embed_texts
from knowledge_base import load_knowledge_base
from repository_manager import load_repository_documents


class DocumentStore:
    def __init__(self):
        self.kb_documents = []
        self.repo_documents = []
        self.upload_documents = []

        self.kb_embeddings = None
        self.repo_embeddings = None
        self.upload_embeddings = None

    # ------------------------
    # Load base sources
    # ------------------------
    def load_kb(self):
        self.kb_documents = load_knowledge_base()

        if self.kb_documents:
            self.kb_embeddings = embed_texts(self.kb_documents)

    def load_repo(self):
        self.repo_documents = load_repository_documents()

        if self.repo_documents:
            self.repo_embeddings = embed_texts(self.repo_documents)

    # ------------------------
    # Upload handling
    # ------------------------
    def add_uploads(self, documents):
        self.upload_documents.extend(documents)

        if self.upload_documents:
            self.upload_embeddings = embed_texts(self.upload_documents)

    # ------------------------
    # Unified access
    # ------------------------
    def get_documents(self):
        return (
            self.kb_documents +
            self.repo_documents +
            self.upload_documents
        )

    def get_embeddings(self):
        embedding_dim = 1536

        kb = self.kb_embeddings
        repo = self.repo_embeddings
        upload = self.upload_embeddings

        if kb is None:
            kb = np.empty((0, embedding_dim))
        if repo is None:
            repo = np.empty((0, embedding_dim))
        if upload is None:
            upload = np.empty((0, embedding_dim))

        return np.vstack([kb, repo, upload])