import argparse
import contextlib
import io
import json
import statistics
import time
import uuid
from pathlib import Path

import yaml

from src.graph.builder import build_graph
from src.query.validation import load_settings
from src.storage.vector_store.chroma_store import ChromaStore

HERE = Path(__file__).parent


def run_one(graph, question: str, doc_ids: list[str]) -> tuple[dict, float]:
    state = {
        "query": {"modality": "text", "text": question},
        "doc_ids": doc_ids,
        "retries": 0, "errors": [], "refused": False,
        "answer": "", "citations": [],
    }
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}   # fresh session each time
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):               # hide the debug prints
        result = graph.invoke(state, config)
    return result, time.perf_counter() - start


def judge(item: dict, result: dict) -> tuple[str, str]:
    """Returns (status, note). Status: pass, fail or error."""
    if result.get("errors"):
        return "error", result["errors"][0][:80]
    refused = bool(result.get("refused"))

    if not item["answerable"]:
        return ("pass", "") if refused else ("fail", "answered an out-of-scope question")

    if refused:
        return "fail", "wrongly refused"
    pages = {c.get("page") for c in result.get("citations", [])}
    if item.get("pages") and not pages & set(item["pages"]):
        return "fail", f"cited pages {sorted(p for p in pages if p)} not in {item['pages']}"
    answer = result.get("answer", "").lower()
    keywords = [str(k).lower() for k in item.get("keywords", [])]
    if keywords and not any(k in answer for k in keywords):
        return "fail", f"none of the keywords {keywords} in the answer"
    return "pass", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", help="document ids (default: all indexed)")
    parser.add_argument("--file", default=str(HERE / "questions.yaml"))
    args = parser.parse_args()

    cfg = load_settings()
    doc_ids = args.docs or ChromaStore(cfg).document_ids()
    if not doc_ids:
        raise SystemExit("Nothing is indexed. Run ingestion first.")
    items = yaml.safe_load(Path(args.file).read_text(encoding="utf-8"))

    graph = build_graph()
    rows = []
    print(f"Documents: {doc_ids}\n")
    for item in items:
        result, secs = run_one(graph, item["question"], doc_ids)
        status, note = judge(item, result)
        rows.append({**item, "status": status, "note": note, "seconds": round(secs, 1),
                     "answer": result.get("answer", "")})
        print(f"{status.upper():5} {secs:5.1f}s [{item.get('type', '-'):9}] "
              f"{item['question']}" + (f"  <- {note}" if note else ""))

    def rate(rows_, label):
        done = [r for r in rows_ if r["status"] != "error"]
        ok = sum(r["status"] == "pass" for r in done)
        return f"{label}: {ok}/{len(done)}" if done else f"{label}: n/a"

    answerable = [r for r in rows if r["answerable"]]
    refusable = [r for r in rows if not r["answerable"]]
    print("\n--- Summary ---")
    print(rate(answerable, "Answerable questions answered correctly"))
    print(rate(refusable, "Out-of-scope questions refused"))
    print(f"Errors (not counted above): {sum(r['status'] == 'error' for r in rows)}")
    print(f"Median time: {statistics.median(r['seconds'] for r in rows):.1f}s")

    out = HERE / "last_run.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full results saved to {out}")


if __name__ == "__main__":
    main()