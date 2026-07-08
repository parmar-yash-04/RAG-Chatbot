import os
import time
import uuid
import base64
import requests as _requests
from typing import Callable

import logfire
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.embeddings import embed_text, embed_texts
from app.llm import ask_groq
from app.guardrails import sanitize_input, validate_output
from app.models import ChatResponse, Source
from app.cache import get_cached_chat, set_cached_chat, clear_cache


class WorkflowError(Exception):
    pass


class State(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


class Node:
    def __init__(self, name: str, fn: Callable, metadata: dict | None = None):
        self.name = name
        self.fn = fn
        self.metadata = metadata if metadata is not None else {}

    def run(self, state: State) -> State:
        with logfire.span("node", name=self.name, **self.metadata):
            start = time.perf_counter()
            try:
                result = self.fn(state)
            except Exception as e:
                raise WorkflowError(f"Node '{self.name}' failed: {e}") from e
            elapsed = time.perf_counter() - start
            logfire.info(f"[{self.name}] done", duration_ms=round(elapsed * 1000))
            return result


class Edge:
    def __init__(self, source: str, target: str, condition: Callable | None = None, label: str = ""):
        self.source = source
        self.target = target
        self.condition = condition
        self.label = label

    def evaluate(self, state: State) -> str | None:
        if self.condition:
            try:
                return self.target if self.condition(state) else None
            except Exception as e:
                raise WorkflowError(f"Condition on edge {self.source}->{self.target} failed: {e}") from e
        return self.target


class WorkflowGraph:
    def __init__(self, name: str = "Workflow"):
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._entry_point: str | None = None

    def add_node(self, name: str, fn: Callable, **metadata) -> "WorkflowGraph":
        self._nodes[name] = Node(name, fn, metadata)
        return self

    def add_edge(self, source: str, target: str) -> "WorkflowGraph":
        self._edges.append(Edge(source, target))
        return self

    def add_conditional_edge(
        self, source: str, target_true: str, target_false: str, condition: Callable,
        label_true: str = "Yes", label_false: str = "No",
    ) -> "WorkflowGraph":
        self._edges.append(Edge(source, target_true, condition, label_true))
        self._edges.append(Edge(source, target_false, lambda s: not condition(s), label_false))
        return self

    def set_entry_point(self, name: str) -> "WorkflowGraph":
        if name not in self._nodes:
            raise WorkflowError(f"Entry node '{name}' not found")
        self._entry_point = name
        return self

    def compile(self) -> "CompiledGraph":
        if not self._entry_point:
            raise WorkflowError("No entry point set")
        adj: dict[str, list[Edge]] = {n: [] for n in self._nodes}
        for edge in self._edges:
            if edge.target is not None and edge.target not in self._nodes and edge.target != edge.source:
                pass
            adj.setdefault(edge.source, []).append(edge)
        return CompiledGraph(
            name=self.name,
            nodes=self._nodes,
            edges=self._edges,
            adj=adj,
            entry_point=self._entry_point,
        )


class CompiledGraph:
    def __init__(self, name: str, nodes: dict[str, Node], edges: list[Edge], adj: dict[str, list[Edge]], entry_point: str):
        self.name = name
        self._nodes = nodes
        self._edges = edges
        self._adj = adj
        self._entry_point = entry_point

    def get_graph(self):
        return self

    def draw_mermaid_png(self) -> bytes:
        try:
            mermaid_code = self.to_mermaid()
            encoded = base64.b64encode(mermaid_code.encode()).decode()
            resp = _requests.get(f"https://mermaid.ink/img/{encoded}", timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            raise WorkflowError(f"Failed to render graph image: {e}") from e

    def to_mermaid(self) -> str:
        lines = ["flowchart TB"]
        has_conditional = {e.source for e in self._edges if e.label}
        node_ids, shape_map = {}, {}
        for i, n in enumerate(self._nodes):
            nid = f"N{i}"
            node_ids[n] = nid
            meta = self._nodes[n].metadata
            ntype = meta.get("type", "")
            label = n.replace("_", " ").title()
            if n in has_conditional or ntype == "decision":
                shape_map[nid] = "decision"
                lines.append(f'    {nid}{{{{"<b>{label}</b>"}}}}')
            elif ntype == "output":
                shape_map[nid] = "output"
                lines.append(f'    {nid}["<b>{label}</b>"]')
            else:
                shape_map[nid] = "process"
                tag = ntype.upper() if ntype else "NODE"
                lines.append(f'    {nid}["<b>{label}</b><br/><i>{tag}</i>"]')

        for edge in self._edges:
            src, tgt = node_ids.get(edge.source), node_ids.get(edge.target)
            if not src or not tgt:
                continue
            if edge.label:
                lines.append(f"    {src} -->|{edge.label}| {tgt}")
            else:
                lines.append(f"    {src} --> {tgt}")

        lines.append("")
        lines.append("    classDef process fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px")
        lines.append("    classDef decision fill:#fff9c4,stroke:#fdd835,stroke-width:2px")
        lines.append("    classDef output fill:#e8f5e9,stroke:#43a047,stroke-width:2px")
        for nid, shape in shape_map.items():
            lines.append(f"    class {nid} {shape}")
        return "\n".join(lines)

    def invoke(self, initial: dict | None = None) -> State:
        state = State(initial or {})
        state["_path"] = []
        current = self._entry_point
        step, max_steps = 0, 100

        while current and step < max_steps:
            step += 1
            node = self._nodes.get(current)
            if not node:
                raise WorkflowError(f"Node '{current}' not found")
            state["_path"].append(current)
            state = node.run(state)

            for edge in self._adj.get(current, []):
                target = edge.evaluate(state)
                if target is not None:
                    current = target
                    break
            else:
                current = None

        if step >= max_steps:
            logfire.warning(
                "Workflow hit max steps", path=state["_path"], max_steps=max_steps
            )
        return state


# ── Qdrant helpers ──

_client: QdrantClient | None = None


def _get_qdrant() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return _client


def _ensure_collection(client: QdrantClient):
    cols = client.get_collections().collections
    if not any(c.name == settings.qdrant_collection for c in cols):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
        )
    try:
        client.create_payload_index(
            collection_name=settings.qdrant_collection,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        logfire.warning("Payload index may already exist", detail=str(e))


# ── Ingestion node functions ──

def _read_file(state: State) -> State:
    raw = state["_raw"]
    filename = state["filename"]
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".txt":
        state["text"] = raw.read().decode("utf-8", errors="replace")
    elif ext == ".pdf":
        try:
            import fitz
        except ModuleNotFoundError:
            raise WorkflowError("PyMuPDF (fitz) is required to read PDF files")
        doc = fitz.open(stream=raw.read(), filetype="pdf")
        state["text"] = "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        try:
            import docx
        except ModuleNotFoundError:
            raise WorkflowError("python-docx is required to read .docx files")
        doc = docx.Document(raw)
        state["text"] = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise WorkflowError(f"Unsupported file type: {ext}")
    if not state["text"].strip():
        raise WorkflowError("File is empty")
    return state


def _chunk_text(state: State) -> State:
    words = state["text"].split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + settings.chunk_size
        chunk = words[start:end]
        if chunk:
            chunks.append(" ".join(chunk))
        start += settings.chunk_size - settings.chunk_overlap
    state["chunks"] = chunks or [state["text"]]
    state["chunk_count"] = len(state["chunks"])
    return state


def _embed_chunks(state: State) -> State:
    state["embeddings"] = embed_texts(state["chunks"])
    return state


def _store_in_qdrant(state: State) -> State:
    client = _get_qdrant()
    _ensure_collection(client)
    doc_id = state.get("doc_id") or str(uuid.uuid4())
    filename = state["filename"]
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=state["embeddings"][i],
            payload={
                "text": state["chunks"][i],
                "doc_id": doc_id,
                "doc_name": filename,
                "chunk_index": i,
            },
        )
        for i in range(state["chunk_count"])
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    state["doc_id"] = doc_id
    return state


def _clear_cache_node(state: State) -> State:
    clear_cache()
    return state


# ── Chat node functions ──

def _parse_question(state: State) -> State:
    question = state["question"]
    if not question.strip():
        raise WorkflowError("Question cannot be empty")
    state["question"] = sanitize_input(question)
    return state


def _cache_lookup(state: State) -> State:
    state["cached_response"] = get_cached_chat(state["question"])
    return state


def _cache_hit_condition(state: State) -> bool:
    return state.get("cached_response") is not None


def _return_cached(state: State) -> State:
    state["response"] = state["cached_response"]
    return state


def _embed_question(state: State) -> State:
    state["query_vector"] = embed_text(state["question"])
    return state


def _search_qdrant(state: State) -> State:
    client = _get_qdrant()
    try:
        results = client.query_points(
            collection_name=settings.qdrant_collection,
            query=state["query_vector"],
            limit=state.get("top_k", 5),
        ).points
    except UnexpectedResponse as e:
        if e.status_code == 404:
            results = []
        else:
            raise
    hits = []
    for r in results:
        if r.payload:
            hits.append({
                "text": r.payload.get("text", ""),
                "score": r.score,
                "doc_name": r.payload.get("doc_name", ""),
                "doc_id": r.payload.get("doc_id", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
            })
    state["search_results"] = hits
    return state


def _check_results_condition(state: State) -> bool:
    return len(state.get("search_results", [])) > 0


def _no_results(state: State) -> State:
    state["response"] = ChatResponse(
        answer="No relevant documents found. Please upload a document first.",
        sources=[],
    )
    return state


def _build_context(state: State) -> State:
    hits = state["search_results"]
    state["sources"] = [
        Source(
            text=h["text"][:300],
            score=h["score"],
            doc_name=h["doc_name"],
            chunk_index=h["chunk_index"],
        )
        for h in hits
    ]
    return state


def _call_llm(state: State) -> State:
    state["llm_answer"] = ask_groq(state["question"], state["search_results"])
    return state


def _validate_output_node(state: State) -> State:
    state["llm_answer"] = validate_output(state["llm_answer"])
    return state


def _cache_result(state: State) -> State:
    response = ChatResponse(
        answer=state["llm_answer"],
        sources=state["sources"],
    )
    set_cached_chat(state["question"], response)
    state["response"] = response
    return state


# ── Compiled workflow singletons ──

_chat_workflow: CompiledGraph | None = None
_ingestion_workflow: CompiledGraph | None = None


def get_chat_workflow() -> CompiledGraph:
    global _chat_workflow
    if _chat_workflow is None:
        g = WorkflowGraph("RAG Chat")
        g.add_node("parse_question", _parse_question, type="input")
        g.add_node("cache_lookup", _cache_lookup, type="cache")
        g.add_node("cache_hit_return", _return_cached, type="output")
        g.add_node("embed_question", _embed_question, type="embedding")
        g.add_node("search_qdrant", _search_qdrant, type="retrieval")
        g.add_node("no_results_found", _no_results, type="output")
        g.add_node("build_context", _build_context, type="context")
        g.add_node("call_llm", _call_llm, type="llm")
        g.add_node("validate_output", _validate_output_node, type="guardrail")
        g.add_node("cache_and_return", _cache_result, type="cache")

        g.set_entry_point("parse_question")
        g.add_edge("parse_question", "cache_lookup")
        g.add_conditional_edge("cache_lookup", "cache_hit_return", "embed_question", _cache_hit_condition)
        g.add_edge("cache_hit_return", None)
        g.add_edge("embed_question", "search_qdrant")
        g.add_conditional_edge("search_qdrant", "build_context", "no_results_found", _check_results_condition)
        g.add_edge("no_results_found", None)
        g.add_edge("build_context", "call_llm")
        g.add_edge("call_llm", "validate_output")
        g.add_edge("validate_output", "cache_and_return")
        g.add_edge("cache_and_return", None)
        _chat_workflow = g.compile()
    return _chat_workflow


def get_ingestion_workflow() -> CompiledGraph:
    global _ingestion_workflow
    if _ingestion_workflow is None:
        g = WorkflowGraph("Document Ingestion")
        g.add_node("read_file", _read_file, type="extract")
        g.add_node("chunk_text", _chunk_text, type="chunking")
        g.add_node("embed_chunks", _embed_chunks, type="embedding")
        g.add_node("store_in_qdrant", _store_in_qdrant, type="storage")
        g.add_node("clear_cache", _clear_cache_node, type="cache")
        g.set_entry_point("read_file")
        g.add_edge("read_file", "chunk_text")
        g.add_edge("chunk_text", "embed_chunks")
        g.add_edge("embed_chunks", "store_in_qdrant")
        g.add_edge("store_in_qdrant", "clear_cache")
        _ingestion_workflow = g.compile()
    return _ingestion_workflow
