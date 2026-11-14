# Multimodal RAG

A question-answering system that answers **only** from documents the user selects (PDF, DOCX, TXT). If the documents do not contain the answer, it replies **"Not in the document"** instead of guessing. Every sentence of an answer carries a citation.

## Features

- **Inputs:** text, audio (Whisper transcription with user confirmation) or an image plus a typed question.
- **Documents:** one or several per query, each with its own Chroma collection, with a mandatory document-ID filter on every search.
- **Content types:** text, tables (stored as markdown) and images (captioned by a vision LLM).
- **Hybrid retrieval:** vector search + BM25, merged with Reciprocal Rank Fusion, then reranked with a cross-encoder.
- **Relevance check:** reranker score threshold, then an LLM grader that must quote exact evidence (the code verifies the quote), with one query rewrite and retry before refusing.
- **Cited answers:** the LLM returns each sentence with a source number; the code builds the citation (document, page, bounding box, chunk ID), so sources cannot be invented.
- **Follow-up questions:** a rewriter uses chat history to make follow-ups self-contained.
- **LLM fallback:** Gemini first; on rate limit (429) or overload (503) it switches to an NVIDIA model, waits and retries (up to 3 rounds).

## How it works

```
validate
  ├─ text  ──────────────────────────────┐
  ├─ audio → transcribe → confirm ───────┤
  └─ image → caption_image ──────────────┤
                                         ▼
                                       clean
                                         ▼
                       rewrite_followup (only if history exists)
                                         ▼
                      retrieve (vector + BM25 + RRF, top 15)
                                         ▼
                           rerank (cross-encoder, top 5)
                                         ▼
                                       grade
                  ┌──────────────────────┼─────────────────────┐
              relevant        irrelevant, retries < 1       otherwise
                  ▼                      ▼                      ▼
              generate        rewrite_query → retrieve        refuse
```

Any node that sets `errors` stops the graph, and `main.py` prints them.

## Project layout

```
multi_modal_rag/
├── main.py                     launcher: pick documents, ask, confirm audio, print answer
├── configs/settings.yaml       all settings
├── .env                        API keys (not committed)
├── data/
│   ├── processed/<document_id>/   chunks.json, manifest.json, images/
│   └── chroma/                    vector store
├── src/
│   ├── query/                  schema, validation, follow-up rewriter, handlers, spell correction
│   ├── models/                 stt, vision_model_query, llm_for_follow_up, llm_decider, text_embedder
│   ├── ingestion/              scanner, registry, selection, skip_processing, chunking/, extractors/
│   ├── storage/vector_store/   base, chroma_store
│   ├── retrieval/              hybrid.py, reranker.py, grader.py
│   ├── generation/             generator.py
│   └── graph/                  state.py, builder.py, nodes/
└── tests/
    ├── manual/                 helper scripts
    └── eval/                   questions.yaml, run_eval.py
```

## Setup (Windows)

Run everything from the `multi_modal_rag/` folder.

```
python -m venv ayantrik
ayantrik\Scripts\activate
pip install -r requirements.txt
```

<!-- CHECK: I have not seen requirements.txt. If it does not exist, create it with: pip freeze > requirements.txt -->

Notes on packages:

- Install `openai-whisper`, not `whisper`.
- `rank-bm25` is required for hybrid search.
- `python-docx` is required for DOCX files (the package is `python-docx`, not `docx`).
- Whisper needs `ffmpeg` available on your PATH.

### API keys

Create a `.env` file in `multi_modal_rag/`:

```
WHISPER_API_KEY=
NVIDIA_API_KEY=
GEMINI_API_KEY=
```

Never commit this file.

### Configuration

Settings live in `configs/settings.yaml`. Main blocks:

```yaml
retrieval:  {vector_k: 20, bm25_k: 20, rrf_k: 60, candidates: 15}
reranker:   {model: BAAI/bge-reranker-base, top_n: 5}
grading:    {min_rerank_score: 0.3, chunks_to_grade: 3}
llm:        {model: gemini-3.6-flash, fallback_model: <NVIDIA model name>, timeout_seconds: 30}
generation: {chunks_to_answer: 5}
```

Optional chunking settings (defaults apply if missing): `heading_max_chars` (80), `paragraph_gap` (0.7), `margin_fraction` (0.1), `running_text_share` (0.4).

A missing or mis-indented key in `settings.yaml` raises a `KeyError`.

## Usage

```
python main.py
```

<!-- CHECK: confirm that main.py is the only command needed (it selects documents, ingests them if needed, then asks questions). -->

You choose the documents, then the input mode (text, audio or image). For audio you confirm the transcript before the question runs.

### Evaluation

```
python tests\eval\run_eval.py --pause 8
```

`--pause` waits the given number of seconds between questions to stay under rate limits. `--docs` takes document IDs (default: all indexed). Results are saved to `tests/eval/last_run.json`. The question set (`tests/eval/questions.yaml`) includes out-of-scope questions that must be refused.

## Re-ingesting documents

`data/processed/registry.json` (data) records file hashes. Unchanged files are skipped. Delete that JSON to re-ingest everything, and do so after any chunker change. Do not confuse it with `src/ingestion/registry.py` (code).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `getaddrinfo failed` | Network dropped. Once models are cached, set `HF_HUB_OFFLINE=1` to stop the embedder contacting the Hub. |
| 429 / `RESOURCE_EXHAUSTED` | Provider rate limit. The fallback model handles it; use `--pause` in evaluation. |
| `KeyError` on startup | Missing or mis-indented key in `settings.yaml`. |
| Key not found | Check the names in `.env` (without printing the values). |

## Status

**Working (tested on a PDF):** text, audio and image input; hybrid retrieval, reranking and grading; cited answers; correct refusals; follow-up rewriting.

## Limitations

- Images from documents and questions are sent to third-party LLM providers (Gemini, NVIDIA) for captioning, grading and generation.
- Free-tier providers cause rate limits and slower responses.
- Up to 5 documents per query.
- Citations for image chunks use the caption text.
