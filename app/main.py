import logfire
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

from app.logging_config import setup_logging
from app.models import (
    ChatRequest,
    ChatResponse,
    UploadResponse,
    DocumentInfo,
)
from app.retrieval import list_documents, delete_document
from app.workflow import get_chat_workflow, get_ingestion_workflow

app = FastAPI(title="RAG Chatbot", version="1.0.0")
setup_logging(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/graph")
async def get_graph_image():
    try:
        wf = get_chat_workflow()
        png_bytes = wf.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...), doc_name: str | None = Form(None)
):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("txt", "pdf", "docx"):
        raise HTTPException(400, f"Unsupported file type: .{ext}")

    try:
        wf = get_ingestion_workflow()
        state = wf.invoke({
            "_raw": file.file,
            "filename": file.filename,
            "doc_id": doc_name,
        })
        return UploadResponse(
            doc_id=state["doc_id"],
            doc_name=file.filename,
            chunks=state["chunk_count"],
            message=f"Document '{file.filename}' ingested ({state['chunk_count']} chunks)",
        )
    except Exception as e:
        logfire.error("Upload failed", error=str(e))
        raise HTTPException(500, str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    wf = get_chat_workflow()
    state = wf.invoke({
        "question": req.question,
        "top_k": req.top_k,
    })

    response: ChatResponse = state["response"]
    return response


@app.get("/documents", response_model=list[DocumentInfo])
async def get_documents():
    docs = list_documents()
    return [DocumentInfo(**d) for d in docs]


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: str):
    success = delete_document(doc_id)
    if not success:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    from app.cache import clear_cache
    clear_cache()
    return {"message": f"Document '{doc_id}' deleted"}
