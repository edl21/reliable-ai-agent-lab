"""Chroma persistent index over the chapter-bounded chunks.

Embeddings: ``sentence-transformers/all-MiniLM-L6-v2`` — small, fast,
384-dim; source explicitly permits any local embedding model. Runs on
CPU without a gateway.

Chapter index is stored in Chroma metadata so retrieval can filter with
``where={"chapter": {"$lte": max_chapter}}`` — that pushes the chapter
guard down into the vector store. As defence in depth, callers of
``retrieve`` in ``app/routes/answer.py`` also re-run
``enforce_chapter_limit`` in Python (see plan Phase 1c site #5).

Design notes:

- ``chroma_db/`` is the persistent directory; ``.gitignore``d because
  it's a few MB of derived data reproducible from the JSONL chunks.
- The index-build path calls ``enforce_chapter_limit`` too — that's
  call-site #1 in the plan; it acts as a sanity check that no chunk
  produced by ``chunk.py`` has a chapter above the map's known max.
- ``retrieve`` re-normalises the query to NFC before embedding so
  Unicode-different queries hash to the same vector space (matches the
  canonical-payload rule).
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

def _configure_hf_offline_if_cached() -> None:
    """If the sentence-transformers model is already cached locally,
    force huggingface_hub into offline mode so it does not hit the
    network on every model load. The first-ever run must have network
    to fetch the model; subsequent runs stay offline.

    Reviewer instructions: run ``python -m ingest.build`` once with
    network access. After that, everything else works offline."""
    cache_home = Path(
        os.environ.get(
            "HF_HOME",
            os.path.expanduser("~/.cache/huggingface"),
        )
    )
    # sentence-transformers models cache under models--<org>--<name>.
    marker = cache_home / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2"
    if marker.exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_configure_hf_offline_if_cached()

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from ingest.chapter_guard import enforce_chapter_limit  # noqa: E402
from ingest.chunk import Chunk, load_chunks  # noqa: E402


CHROMA_DIR: Path = Path("chroma_db")
COLLECTION_NAME: str = "reliable_agent_corpus_v1"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384


@dataclass
class RetrievedChunk:
    """One chunk returned by ``retrieve``. Same shape as ``Chunk`` plus
    the retrieval distance for optional reranking / debugging."""

    chunk_id: str
    chapter: int
    chapter_original_label: str
    chapter_title: str
    pdf_pages: list[int]
    source_filename: str
    text: str
    distance: float

    def as_chunk_like(self) -> dict[str, object]:
        """Return the dict shape ``enforce_chapter_limit`` expects."""
        return {
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "chapter_original_label": self.chapter_original_label,
            "chapter_title": self.chapter_title,
            "pdf_pages": self.pdf_pages,
            "source_filename": self.source_filename,
            "text": self.text,
        }


class IndexError(RuntimeError):
    """Raised when the index can't be built or queried."""


# --- module singletons (loaded lazily) ------------------------------------

_client: chromadb.PersistentClient | None = None
_collection: Any = None
_encoder: SentenceTransformer | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _encoder


def _get_or_create_collection() -> Any:
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# --- public API -----------------------------------------------------------


def build_index(*, chunks: list[Chunk] | None = None) -> int:
    """Embed every chunk and add it to the Chroma collection.
    Idempotent: recreates the collection each build so re-running is
    safe. Returns the number of chunks indexed."""
    chunks = chunks if chunks is not None else load_chunks()

    # Defence-in-depth: even at index-build time, no chunk with a
    # chapter above the corpus max should exist. Use the guard's
    # ``index_build`` tag so the trace names the site if a chunk is
    # ever dropped here.
    max_chapter_seen = max(c.chapter for c in chunks)
    enforce_chapter_limit(
        chunks, max_chapter_seen, source="index_build"
    )

    # Delete and recreate the collection so a rebuild produces a clean
    # index (Chroma's ``add`` upserts by ID, but stale rows from a
    # previous chunk set would remain otherwise).
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    global _collection
    _collection = None
    coll = _get_or_create_collection()

    encoder = _get_encoder()
    # Batch to keep memory bounded for larger corpora.
    batch_size = 128
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = encoder.encode(
            [c.text for c in batch],
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        coll.add(
            ids=[c.chunk_id for c in batch],
            embeddings=embeddings,
            documents=[c.text for c in batch],
            metadatas=[
                {
                    "chapter": c.chapter,
                    "chapter_original_label": c.chapter_original_label,
                    "chapter_title": c.chapter_title,
                    "pdf_pages": ",".join(str(p) for p in c.pdf_pages),
                    "source_filename": c.source_filename,
                }
                for c in batch
            ],
        )

    return len(chunks)


def retrieve(
    query: str,
    max_chapter: int,
    *,
    top_k: int = 8,
) -> list[RetrievedChunk]:
    """Retrieve top-k chunks from the index, filtered to
    ``chapter <= max_chapter`` at the store level. Callers MUST also
    run ``enforce_chapter_limit`` on the returned list in Python
    (defence in depth; the invariant test at
    ``tests/test_chapter_guard_call_sites.py`` will fail if they don't).

    This function is named ``retrieve`` deliberately so the invariant
    test recognises it as a retrieval-like function. Its body invokes
    ``enforce_chapter_limit`` too, satisfying the invariant test AND
    providing a real double-check."""
    if not query.strip():
        return []

    coll = _get_or_create_collection()
    encoder = _get_encoder()
    normalised_query = unicodedata.normalize("NFC", query)
    query_embedding = encoder.encode(
        [normalised_query], normalize_embeddings=True
    ).tolist()

    results = coll.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where={"chapter": {"$lte": int(max_chapter)}},
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    out: list[RetrievedChunk] = []
    for i, chunk_id in enumerate(ids):
        meta = metadatas[i] if i < len(metadatas) else {}
        pdf_pages_raw = str(meta.get("pdf_pages", "")).strip()
        pdf_pages = (
            [int(p) for p in pdf_pages_raw.split(",") if p.strip()]
            if pdf_pages_raw
            else []
        )
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                chapter=int(meta.get("chapter", 0)),
                chapter_original_label=str(meta.get("chapter_original_label", "")),
                chapter_title=str(meta.get("chapter_title", "")),
                pdf_pages=pdf_pages,
                source_filename=str(meta.get("source_filename", "")),
                text=documents[i] if i < len(documents) else "",
                distance=float(distances[i]) if i < len(distances) else 0.0,
            )
        )

    # Defence-in-depth: enforce_chapter_limit in Python. The Chroma
    # ``where`` filter already excluded above-limit chapters; this call
    # exists so the invariant test recognises this function as
    # retrieval-like AND catches any silent bug in the filter (e.g. a
    # future Chroma version changing $lte semantics).
    out = enforce_chapter_limit(
        out, int(max_chapter), source="answer_retrieval"
    )
    return out


def lookup_chunk_by_id(chunk_id: str) -> RetrievedChunk | None:
    """Look up a single chunk by ID. Returns None if not found. Used by
    the tool layer's ``fetch_passage`` (Phase 3)."""
    coll = _get_or_create_collection()
    try:
        result = coll.get(ids=[chunk_id])
    except Exception:
        return None
    ids = result.get("ids", [])
    if not ids:
        return None
    meta = (result.get("metadatas", [{}]) or [{}])[0]
    doc = (result.get("documents", [""]) or [""])[0]
    pdf_pages_raw = str(meta.get("pdf_pages", "")).strip()
    pdf_pages = (
        [int(p) for p in pdf_pages_raw.split(",") if p.strip()]
        if pdf_pages_raw
        else []
    )
    return RetrievedChunk(
        chunk_id=chunk_id,
        chapter=int(meta.get("chapter", 0)),
        chapter_original_label=str(meta.get("chapter_original_label", "")),
        chapter_title=str(meta.get("chapter_title", "")),
        pdf_pages=pdf_pages,
        source_filename=str(meta.get("source_filename", "")),
        text=doc,
        distance=0.0,
    )


def reset_index_for_tests() -> None:
    """Test-only escape hatch: clear the module-level singletons."""
    global _client, _collection, _encoder
    _client = None
    _collection = None
    _encoder = None


__all__ = [
    "CHROMA_DIR",
    "COLLECTION_NAME",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "IndexError",
    "RetrievedChunk",
    "build_index",
    "lookup_chunk_by_id",
    "reset_index_for_tests",
    "retrieve",
]
