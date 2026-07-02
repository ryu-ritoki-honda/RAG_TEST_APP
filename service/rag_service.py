from rag import retrieve, answer_question

def get_chunks(question, documents, embeddings, sort_mode="relevance"):
    return retrieve(question, documents, embeddings, sort_mode=sort_mode)

def get_answer(question, chunks):
    return answer_question(question, chunks)