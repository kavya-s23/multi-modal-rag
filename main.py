import argparse
import uuid
from pathlib import Path

from langgraph.types import Command

from src.graph.builder import build_graph

PROCESSED = Path("data/processed")


def list_documents() -> list[str]:
    """Each processed document is a folder named by its document_id."""
    if not PROCESSED.exists():
        return []
    return sorted(p.name for p in PROCESSED.iterdir() if p.is_dir())


def choose_documents(cli_docs: list[str] | None) -> list[str]:
    available = list_documents()
    if cli_docs:
        unknown = [d for d in cli_docs if d not in available]
        if unknown:
            raise SystemExit(f"Unknown document id(s): {unknown}")
        return cli_docs
    if not available:
        raise SystemExit("No indexed documents. Run ingestion first.")
    print("Indexed documents:")
    for i, d in enumerate(available, 1):
        print(f"  {i}. {d}")
    raw = input("Select by number (comma separated, max 5): ")
    picked = [available[int(x) - 1] for x in raw.split(",") if x.strip()]
    if not picked:
        raise SystemExit("Select at least one document.")
    return picked[:5]


def read_turn() -> dict | None:
    """Ask for one query. Returns None to quit."""
    mode = input("\nInput [t]ext / [a]udio / [i]mage / [q]uit: ").strip().lower()
    if mode == "q":
        return None
    if mode == "a":
        return {"modality": "audio",
                "audio_path": input("Audio file path: ").strip().strip('"')}
    if mode == "i":
        return {
            "modality": "image",
            "image_path": input("Image file path: ").strip().strip('"'),
            "text": input("Question about the image (required): ").strip(),
        }
    return {"modality": "text", "text": input("Question: ").strip()}


def handle_interrupts(graph, result: dict, config: dict) -> dict:
    """Resume the graph while it keeps asking for user confirmation."""
    while result.get("__interrupt__"):
        payload = result["__interrupt__"][0].value
        print(f"\nTranscript: {payload}")
        reply = input("Correct? [y] yes / [n] cancel / or type the corrected text: ").strip()
        answer = {"confirmed": True} if reply.lower() in ("", "y") else (
            {"confirmed": False} if reply.lower() == "n"
            else {"confirmed": True, "text": reply}
        )
        result = graph.invoke(Command(resume=answer), config)
    return result


def print_result(result: dict) -> None:
    if result.get("errors"):
        for e in result["errors"]:
            print(f"\n{e}")
        return
    if result.get("refused"):
        print("\nNot in the document.")
        return
    print(f"\n{result.get('answer', '')}\n")
    for n, c in enumerate(result.get("citations", []), 1):
        label = f" ({c['section']})" if c.get("section") else ""
        print(f"  [{n}] {c.get('doc_id')} p.{c.get('page')}{label}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", help="document ids to search")
    args = parser.parse_args()

    graph = build_graph()
    doc_ids = choose_documents(args.docs)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}  # one session
    print(f"Searching in: {doc_ids}")

    while True:
        turn = read_turn()
        if turn is None:
            break
        state_in = {
            "query": turn,
            "doc_ids": doc_ids,
            # reset per-turn fields; history persists via the checkpointer
            "retries": 0,
            "refused": False,
            "answer": "",
            "citations": [],
        }
        result = graph.invoke(state_in, config)
        result = handle_interrupts(graph, result, config)
        print_result(result)


if __name__ == "__main__":
    main()