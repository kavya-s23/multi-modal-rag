import json
import shutil
from .outline import make_outline
from ..models import text_embedder
from ..storage.vector_store.chroma_store import ChromaStore
from .chunking.docx_chunker import chunk_docx
from .chunking.pdf_chunker import chunk_pdf
from .chunking.txt_chunker import chunk_txt
from .extractors.pdf_images import PROCESSED, extract_images
from .registry import record, status
from .scanner import FileInfo


def _ingest_document(info: FileInfo, cfg: dict, store) -> dict:
    """Reads, chunks, embeds and stores one document. Raises on failure."""
    if info.extension not in (".pdf", ".docx", ".txt"):
        raise NotImplementedError(f"{info.extension} files are not supported.")

    out_dir = PROCESSED / info.document_id
    if out_dir.exists():
        shutil.rmtree(out_dir)            # rebuild from scratch
    out_dir.mkdir(parents=True)

    problems: list[str] = []
    ocr_pages: list[int] = []
    if info.extension == ".pdf":
        chunks, ocr_pages = chunk_pdf(info.path, info.document_id, cfg)
        if cfg["ingestion"].get("extract_images", True):
            image_chunks, problems = extract_images(info.path, info.document_id, cfg)
            chunks += image_chunks
    elif info.extension == ".docx":
        chunks = chunk_docx(info.path, info.document_id, cfg)
        problems = []                     # images inside DOCX are not read yet
    else:
        chunks = chunk_txt(info.path, info.document_id, cfg)

    if not chunks:
        raise ValueError("No content could be extracted from this document.")
    chunks += make_outline(chunks, info.document_id, info.name, cfg)
    # ... everything from here down (chunks.json, embeddings, store, summary) is unchanged

    if not chunks:
        raise ValueError("No content could be extracted from this document.")

    (out_dir / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # Embed every chunk's text and store it in this document's own collection
    vectors = text_embedder.embed_documents([c["text"] for c in chunks], cfg)
    store.add_chunks(info.document_id, chunks, vectors)

    summary = {
        "chunks": len(chunks),
        "text": sum(c["metadata"]["type"] == "text" for c in chunks),
        "tables": sum(c["metadata"]["type"] == "table" for c in chunks),
        "images": sum(c["metadata"]["type"] == "image" for c in chunks),
        "ocr_pages": ocr_pages,
        "problems": problems,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps({"document_id": info.document_id, "name": info.name,
                    "file_hash": info.file_hash, **summary}, indent=2),
        encoding="utf-8")
    return summary


def ingest_files(files: list[FileInfo], cfg: dict, force: bool = False) -> dict:
    report = {"ingested": [], "skipped": [], "failed": [],
              "incomplete": [], "details": {}}
    store = ChromaStore(cfg)

    for i, f in enumerate(files, start=1):
        s = status(f)

        if s == "indexed" and not force:
            print(f"[{i}/{len(files)}] Skipping {f.name}: already indexed, unchanged")
            report["skipped"].append(f.name)
            continue

        print(f"[{i}/{len(files)}] {'Rebuilding' if s == 'changed' else 'Ingesting'} {f.name}")
        try:
            summary = _ingest_document(f, cfg, store)
            report["details"][f.name] = summary
            print(f"    {summary['chunks']} chunks stored: {summary['text']} text, "
                  f"{summary['tables']} tables, {summary['images']} images"
                  + (f", OCR needed on pages {summary['ocr_pages']}" if summary["ocr_pages"] else ""))
            for p in summary["problems"]:
                print("    Problem:", p)

            if any("could not be captioned" in p for p in summary["problems"]):
                report["incomplete"].append(f.name)
                print("    Not marked as indexed. Run again to retry the failed images.")
            else:
                record(f)                  # only after Chroma has the data
                report["ingested"].append(f.name)
        except Exception as exc:           # one bad file must not stop the run
            print(f"    Failed: {exc}")
            report["failed"].append((f.name, str(exc)))

    return report