with open(
    "documents.txt",
    encoding="utf-8"
) as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]