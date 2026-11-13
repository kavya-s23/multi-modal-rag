from src.query.validation import load_settings
from src.ingestion.scanner import scan_path
from src.ingestion.selection import choose_files
from src.ingestion.skip_processing import ingest_files
from src.storage.vector_store.chroma_store import ChromaStore
cfg = load_settings()

while True:
    raw = input("\nFile or folder path (blank to quit): ").strip()
    if not raw:
        break

    result = scan_path(raw, cfg)
    if result.error:
        print("Error:", result.error)
        continue

    chosen = choose_files(result, cfg)
    report = ingest_files(chosen, cfg)

    print("\nSummary")
    print("  Ingested:", report["ingested"])
    print("  Skipped :", report["skipped"])
    print("  Failed  :", report["failed"])