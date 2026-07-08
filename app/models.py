from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    text: str
    score: float
    doc_name: str
    chunk_index: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    doc_id: str
    doc_name: str
    chunks: int
    message: str


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    chunk_count: int
