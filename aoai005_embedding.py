import numpy as np
from embeddings_utils import embed_text, cosine_similarity_matrix


def get_embedding(text, model="text-embedding-3-small"):
    return embed_text(text, model=model)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

