import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from src.graph.builder import build_graph
from src.ingestion.scanner import scan_path
from src.ingestion.skip_processing import ingest_files
from src.query.validation import load_settings

PROCESSED = Path("data/processed")
UPLOADS = Path("data/uploads")      # images and audio attached to a question
LIBRARY = Path("data/library")      # documents added for searching
ALLOWED = {".pdf", ".docx", ".txt"}

app = FastAPI()
graph = build_graph()               # built once, so chat memory lives as long as the server
jobs: dict[str, dict] = {}
ingest_lock = threading.Lock()


def doc_names() -> dict[str, str]:
    names = {}
    if PROCESSED.exists():
        for p in sorted(PROCESSED.iterdir()):
            if not p.is_dir():
                continue
            name = p.name
            manifest = p / "manifest.json"
            if manifest.exists():
                try:
                    name = json.loads(manifest.read_text(encoding="utf-8")).get("name", name)
                except (OSError, ValueError):
                    pass
            names[p.name] = name
    return names


def pack(result: dict) -> dict:
    """Turn the graph's result into one simple JSON reply for the browser."""
    if result.get("__interrupt__"):
        return {"status": "confirm", "transcript": result["__interrupt__"][0].value}
    if result.get("errors"):
        return {"status": "error", "message": " ".join(result["errors"])}
    if result.get("refused"):
        return {"status": "refused", "message": "Not in the document."}
    names = doc_names()
    cites = [{"n": i, "doc": names.get(c.get("doc_id"), c.get("doc_id")),
              "page": c.get("page"), "section": c.get("section"),
              "type": c.get("type"), "text": c.get("text", ""),
              "sentences": c.get("sentences", [])}
             for i, c in enumerate(result.get("citations", []), 1)]
    return {"status": "answer", "answer": result.get("answer", ""), "citations": cites}


class ChatIn(BaseModel):
    thread: str
    doc_ids: list[str]
    modality: str = "text"          # text | image | audio
    text: str = ""
    file: str | None = None         # value returned by /api/upload


class ConfirmIn(BaseModel):
    thread: str
    confirmed: bool
    text: str = ""


# ---------- documents ----------

@app.get("/api/documents")
def documents():
    return [{"id": i, "name": n} for i, n in doc_names().items()]


def run_ingest(job_id: str, paths: list[Path]) -> None:
    job = jobs[job_id]
    try:
        with ingest_lock:                      # one ingestion at a time
            cfg = load_settings()
            for p in paths:
                scan = scan_path(str(p), cfg)
                if scan.error:
                    job["failed"].append([p.name, scan.error])
                    continue
                report = ingest_files(scan.files, cfg)
                job["ingested"] += report["ingested"]
                job["skipped"] += report["skipped"]
                job["incomplete"] += report["incomplete"]
                job["failed"] += [list(x) for x in report["failed"]]
    except Exception as e:
        job["failed"].append(["ingestion", str(e)])
    finally:
        job["done"] = True


@app.post("/api/ingest")
def ingest(files: list[UploadFile] = File(...)):
    LIBRARY.mkdir(parents=True, exist_ok=True)
    paths, rejected = [], []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in ALLOWED:
            rejected.append(name)
            continue
        target = LIBRARY / name
        target.write_bytes(f.file.read())
        paths.append(target)
    if not paths:
        raise HTTPException(400, "Only PDF, DOCX and TXT files can be added.")
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"done": False, "ingested": [], "skipped": [], "incomplete": [],
                    "failed": [[n, "unsupported file type"] for n in rejected]}
    threading.Thread(target=run_ingest, args=(job_id, paths), daemon=True).start()
    return {"job": job_id}


@app.get("/api/ingest/{job_id}")
def ingest_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job.")
    return jobs[job_id]


# ---------- question files (image / audio) ----------

@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    folder = uuid.uuid4().hex
    name = Path(file.filename or "upload").name
    target = UPLOADS / folder / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(file.file.read())
    return {"file": f"{folder}/{name}"}


# ---------- chat ----------

@app.post("/api/chat")
def chat(body: ChatIn):
    if not body.doc_ids:
        raise HTTPException(400, "Select at least one document.")
    query = {"modality": body.modality, "text": body.text}
    if body.file:
        path = (UPLOADS / body.file).resolve()
        if UPLOADS.resolve() not in path.parents or not path.exists():
            raise HTTPException(400, "Unknown uploaded file.")
        query["image_path" if body.modality == "image" else "audio_path"] = str(path)

    state_in = {"query": query, "doc_ids": body.doc_ids[:5], "retries": 0, "errors": [],
                "refused": False, "answer": "", "citations": []}
    config = {"configurable": {"thread_id": body.thread}}
    return pack(graph.invoke(state_in, config))


@app.post("/api/confirm")
def confirm(body: ConfirmIn):
    answer = {"confirmed": body.confirmed}
    if body.confirmed and body.text:
        answer["text"] = body.text
    config = {"configurable": {"thread_id": body.thread}}
    return pack(graph.invoke(Command(resume=answer), config))


app.mount("/", StaticFiles(directory="web", html=True), name="web")   # must stay last