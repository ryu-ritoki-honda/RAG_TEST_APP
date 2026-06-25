from rag import retrieve

results, query_embedding = retrieve(
    "Tell me about Honda SUVs"
)

for doc, score in results:
    print(score)
    print(doc)