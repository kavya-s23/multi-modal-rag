import chromadb

col = chromadb.PersistentClient(path="data/chroma").get_collection("doc_f79ae900bd73_text")
res = col.get(include=["documents", "metadatas"])

sizes = sorted(len(d) for d in res["documents"])
n = len(sizes)
print(f"chunks: {n}")
print(f"min {sizes[0]}  median {sizes[n // 2]}  max {sizes[-1]}")
print(f"under 100 chars: {sum(s < 100 for s in sizes)}")
print(f"over 800 chars:  {sum(s > 800 for s in sizes)}")

by_type = {}
for m in res["metadatas"]:
    by_type[m["type"]] = by_type.get(m["type"], 0) + 1
print("by type:", by_type)

print("\nshortest 5:")
shortest = sorted(zip(res["documents"], res["metadatas"]), key=lambda x: len(x[0]))[:5]
for d, m in shortest:
    print(f"  p.{m['page_number']} {len(d)} chars | {d[:80]!r}")