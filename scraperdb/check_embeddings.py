from embeddings.vector_store import DEFAULT_CHROMA_PATH, DEFAULT_COLLECTION, get_chroma_collection

print("DEFAULT_CHROMA_PATH =", DEFAULT_CHROMA_PATH)
print("DEFAULT_COLLECTION  =", DEFAULT_COLLECTION)

col = get_chroma_collection()
print("CHROMA COUNT =", col.count())
