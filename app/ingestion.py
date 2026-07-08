import uuid
import os
from typing import BinaryIO

import fitz
import docx
import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from app.config import settings
from app.embeddings import embed_texts


def _get_qdrant() -> QdrantClient:
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )
    return client


def _ensure_collection(client: QdrantClient):
    collections = client.get_collections().collections
    if not any(c.name == settings.qdrant_collection for c in collections):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )
        logfire.info("Created collection", name=settings.qdrant_collection)

    try:
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def _chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + settings.chunk_size
        chunk = words[start:end]
        if chunk:
            chunks.append(" ".join(chunk))
        start += settings.chunk_size - settings.chunk_overlap
    return chunks or [text]


def _read_file_content(file: BinaryIO, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        return file.read().decode("utf-8", errors="replace")
    elif ext == ".pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def ingest_document(file: BinaryIO, filename: str, doc_name: str | None = None) -> dict:
    with logfire.span("ingest_document", filename=filename):
        text = _read_file_content(file, filename)

        if not text.strip():
            raise ValueError("File is empty")

        chunks = _chunk_text(text)
        doc_id = doc_name or str(uuid.uuid4())

        logfire.info(
            "Document chunked",
            filename=filename,
            doc_id=doc_id,
            chunks=len(chunks),
        )

        embeddings = embed_texts(chunks)

        client = _get_qdrant()
        _ensure_collection(client)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "text": chunks[i],
                    "doc_id": doc_id,
                    "doc_name": filename,
                    "chunk_index": i,
                },
            )
            for i in range(len(chunks))
        ]

        client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
        )

        logfire.info(
            "Document ingested",
            doc_id=doc_id,
            chunks=len(chunks),
        )

        return {
            "doc_id": doc_id,
            "doc_name": filename,
            "chunks": len(chunks),
        }
