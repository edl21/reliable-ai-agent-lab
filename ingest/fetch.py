"""Public-domain corpus fetch + fingerprint.

The local demo uses Herman Melville's ``Moby-Dick; or, The Whale``:
a long, chapter-structured work that is public domain in the United
States. The PDF is a GITenberg/Project Gutenberg-derived edition
archived by the Internet Archive. Its text is public domain; the cover
art is separately licensed CC BY-NC 4.0 and is not used by the app.

This module is idempotent: if the PDF is already on disk with the
expected SHA-256, we do not re-download. That keeps ``python -m
ingest.build`` cheap to re-run and the reviewer's environment stable.

The manifest is written to ``corpus/manifest.json`` and referenced from
the README so users can verify the corpus without re-downloading.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx


CORPUS_URL: str = (
    "https://ia802804.us.archive.org/5/items/MobyDickGit/Moby-Dick.pdf"
)
CORPUS_TITLE: str = "Moby-Dick; or, The Whale"
CORPUS_AUTHOR: str = "Herman Melville"
CORPUS_FILENAME: str = "moby-dick.pdf"
CORPUS_RIGHTS: str = "Public domain in the United States"
CORPUS_RIGHTS_URL: str = "https://www.gutenberg.org/ebooks/2701"
CORPUS_DIR: Path = Path("corpus")

# Expected SHA-256 pinned so a silently-changed upstream fails loudly
# rather than corrupting downstream chunk IDs. Computed at first
# successful fetch and stored in the manifest; also asserted here for
# fast fail-loud detection. If the reviewer's fetched file differs,
# update the manifest AND this constant in the same commit.
EXPECTED_SHA256: str = (
    "dae176d68d290b954040b819ac84f3901b8fdb308523a7325fddd91b99d79dad"
)


@dataclass
class CorpusManifest:
    """Metadata written to ``corpus/manifest.json``."""

    title: str
    author: str
    source_url: str
    download_date_utc: str
    sha256: str
    file_size_bytes: int
    rights: str = CORPUS_RIGHTS
    rights_url: str = CORPUS_RIGHTS_URL
    pdf_pages: int | None = None            # populated by extract.py
    cleaned_word_count: int | None = None   # populated by extract.py

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "author": self.author,
            "source_url": self.source_url,
            "download_date_utc": self.download_date_utc,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "rights": self.rights,
            "rights_url": self.rights_url,
            "pdf_pages": self.pdf_pages,
            "cleaned_word_count": self.cleaned_word_count,
        }


class CorpusFetchError(RuntimeError):
    """Raised when the download fails or the SHA-256 does not match."""


def fetch_corpus(*, force: bool = False) -> CorpusManifest:
    """Ensure the corpus PDF is on disk with the expected SHA-256, and
    return the manifest.

    Args:
        force: If True, re-download even when the file already exists.
               Useful if a partial file was left on
               disk. Normally leave False for idempotency.

    Raises:
        CorpusFetchError: if the download fails, the response body is
        empty, or the SHA-256 does not match ``EXPECTED_SHA256``.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = CORPUS_DIR / CORPUS_FILENAME
    manifest_path = CORPUS_DIR / "manifest.json"

    if pdf_path.exists() and not force:
        got = _sha256_of_file(pdf_path)
        if got == EXPECTED_SHA256:
            # File is already correct — reuse the on-disk manifest if
            # present, else recompute one.
            if manifest_path.exists():
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                return CorpusManifest(**data)
            manifest = _build_manifest(pdf_path, got)
            _write_manifest(manifest_path, manifest)
            return manifest
        # Hash mismatch — fall through and re-download.

    # Download. Use httpx (already pinned) so the CA bundle from
    # certifi is used automatically, avoiding platform-specific
    # certificate-store differences.
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=60.0,
            headers={"User-Agent": "reliable-ai-agent-lab/1.0"},
        ) as client:
            resp = client.get(CORPUS_URL)
            resp.raise_for_status()
            body = resp.content
    except httpx.HTTPError as exc:
        raise CorpusFetchError(f"could not fetch corpus: {exc}") from exc

    if not body:
        raise CorpusFetchError("corpus fetch returned an empty response")

    got = hashlib.sha256(body).hexdigest()
    if got != EXPECTED_SHA256:
        raise CorpusFetchError(
            f"corpus SHA-256 mismatch: expected {EXPECTED_SHA256}, got {got}. "
            "Upstream changed; update EXPECTED_SHA256 and manifest in the same commit."
        )

    pdf_path.write_bytes(body)
    manifest = _build_manifest(pdf_path, got)
    _write_manifest(manifest_path, manifest)
    return manifest


def load_manifest() -> CorpusManifest:
    """Read the on-disk manifest. Callers should have already run
    ``fetch_corpus``. Raises if the manifest is missing."""
    manifest_path = CORPUS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise CorpusFetchError(
            f"{manifest_path} missing; run `python -m ingest.build` first"
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return CorpusManifest(**data)


def update_manifest(**fields: object) -> CorpusManifest:
    """Rewrite the manifest with the given fields overridden (used by
    extraction to fill in ``pdf_pages`` and ``cleaned_word_count``)."""
    manifest_path = CORPUS_DIR / "manifest.json"
    current = load_manifest().to_dict()
    current.update(fields)
    updated = CorpusManifest(**current)
    _write_manifest(manifest_path, updated)
    return updated


# --- internals -----------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_manifest(pdf_path: Path, sha256: str) -> CorpusManifest:
    return CorpusManifest(
        title=CORPUS_TITLE,
        author=CORPUS_AUTHOR,
        source_url=CORPUS_URL,
        download_date_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        sha256=sha256,
        file_size_bytes=pdf_path.stat().st_size,
    )


def _write_manifest(path: Path, manifest: CorpusManifest) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CORPUS_AUTHOR",
    "CORPUS_DIR",
    "CORPUS_FILENAME",
    "CORPUS_RIGHTS",
    "CORPUS_RIGHTS_URL",
    "CORPUS_TITLE",
    "CORPUS_URL",
    "CorpusFetchError",
    "CorpusManifest",
    "EXPECTED_SHA256",
    "fetch_corpus",
    "load_manifest",
    "update_manifest",
]
