import logfire
from fastapi import FastAPI
from app.config import settings


def setup_logging(app: FastAPI):
    if settings.logfire_token:
        logfire.configure(token=settings.logfire_token, service_name="rag-chatbot")
        logfire.instrument_fastapi(app)
        logfire.info("Logfire initialized", service="rag-chatbot")
