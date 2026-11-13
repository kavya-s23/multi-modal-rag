import chromadb

client = chromadb.PersistentClient(path="data/chroma")
collections = client.list_collections()
print(f"Collections found: {len(collections)}")

for c in collections:
    col = client.get_collection(c.name if hasattr(c, "name") else c)
    print(f"\n{col.name}: {col.count()} chunks")
    sample = col.get(limit=1, include=["metadatas", "embeddings"])
    if sample["ids"]:
        print("  metadata keys:", list(sample["metadatas"][0].keys()))
        print("  sample metadata:", sample["metadatas"][0])
        print("  embedding dim:", len(sample["embeddings"][0]))