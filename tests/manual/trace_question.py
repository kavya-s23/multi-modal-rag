from src.generation.generator import generate_answer
from src.models.llm_decider import ask
from src.query.validation import load_settings
from src.retrieval.grader import GRADER_PROMPT, grade
from src.retrieval.hybrid import hybrid_search
from src.retrieval.reranker import rerank
from src.storage.vector_store.chroma_store import ChromaStore

cfg = load_settings()
docs = input("Document id (blank = all indexed): ").strip()
doc_ids = [docs] if docs else ChromaStore(cfg).document_ids()
question = input("Question: ").strip()

ranked = rerank(question, hybrid_search(question, doc_ids, cfg), cfg)
print("\nTop chunks after rerank:")
for i, c in enumerate(ranked, 1):
    print(f"  {i}. score={c['rerank_score']:.3f} {c['id']} | {c['text'][:90]!r}")

top = ranked[:cfg["grading"]["chunks_to_grade"]]
context = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(top, 1))
print("\nGrader sees these chunks:", [c["id"] for c in top])
print("Grader raw reply:", ask(GRADER_PROMPT.format(question=question, context=context), cfg))
print("Grader verdict  :", grade(question, ranked, cfg))

result = generate_answer(question, ranked, cfg)
print("\nGenerator answer:", result["answer"] or "(empty, so it would refuse)")
for n, c in enumerate(result["citations"], 1):
    print(f"\n[{n}] {c['chunk_id']} p.{c['page']}")
    print(c["text"][:300])