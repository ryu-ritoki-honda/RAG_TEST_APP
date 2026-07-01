from rag import retrieve, answer_question

def get_chunks(question, documents, embeddings):
    return retrieve(question, documents, embeddings)

def get_answer(question, chunks):
    return answer_question(question, chunks)