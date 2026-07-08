import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)

from app.config import settings
from app.embeddings import embed_text


def _get_qdrant() -> QdrantClient:
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
    )


def search(query: str, top_k: int = 5) -> list[dict]:
    with logfire.span("search_qdrant", query=query[:50], top_k=top_k):
        query_vector = embed_text(query)
        client = _get_qdrant()

        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
        ).points

        hits = []
        for r in results:
            if r.payload:
                hits.append(
                    {
                        "text": r.payload.get("text", ""),
                        "score": r.score,
                        "doc_name": r.payload.get("doc_name", ""),
                        "doc_id": r.payload.get("doc_id", ""),
                        "chunk_index": r.payload.get("chunk_index", 0),
                    }
                )

        logfire.info("Search results", count=len(hits))
        return hits


def list_documents() -> list[dict]:
    with logfire.span("list_documents"):
        client = _get_qdrant()
        try:
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=10000,
                with_payload=["doc_id", "doc_name"],
                with_vectors=False,
            )
        except Exception:
            return []

        seen = {}
        for point in scroll_result[0]:
            if point.payload:
                did = point.payload.get("doc_id", "")
                dname = point.payload.get("doc_name", "")
                if did not in seen:
                    seen[did] = {"doc_id": did, "doc_name": dname, "chunk_count": 0}
                seen[did]["chunk_count"] += 1

        return list(seen.values())


def delete_document(doc_id: str) -> bool:
    with logfire.span("delete_document", doc_id=doc_id):
        client = _get_qdrant()
        try:
            scroll_result = client.scroll(
                collection_name=settings.qdrant_collection,
                limit=10000,
                with_payload=False,
                with_vectors=False,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
            )
            point_ids = [p.id for p in scroll_result[0]]
            if not point_ids:
                logfire.warning("No points found to delete", doc_id=doc_id)
                return False

            client.delete(
                collection_name=settings.qdrant_collection,
                points_selector=point_ids,
            )
            logfire.info("Document deleted", doc_id=doc_id, points=len(point_ids))
            return True
        except Exception as e:
            logfire.error("Delete failed", doc_id=doc_id, error=str(e))
            return False
