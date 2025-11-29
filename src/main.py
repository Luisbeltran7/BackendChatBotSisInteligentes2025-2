from contextlib import asynccontextmanager
import csv
import datetime
import os
import time
from datetime import datetime
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path
import shutil

# Cargar variables de entorno desde .env
load_dotenv()

from src.models.schemas import (
    QuestionRequest, 
    QuestionResponse, 
    UploadResponse,
    HealthResponse
)
from .services.rag_service import RAGService
from src.services.pdf_service import prepare_docs_for_chroma
from src.services.pdf_service import process_pdf_with_langchain
from starlette.middleware.base import BaseHTTPMiddleware

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

log_path = os.getenv("CSV_LOG_PATH","/consumo_logs.csv")
LOG_FILE = Path("logs/consumo_logs.csv")


rag_service = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Código que se ejecuta al INICIAR la app
    print("Inicializando RAG service...")
    data_folder = Path("data")
    #data_folder = Path("/app/data")
    if list(data_folder.glob("*.pdf")):
        rag_service.initialize_from_pdfs(data_folder)
        print(f"RAG inicializado con {len(list(data_folder.glob('*.pdf')))} PDFs")
    else:
        print("No se encontraron PDFs en /app/data")
    
    yield  # Aquí la app está corriendo
    
    # Código que se ejecuta al APAGAR la app (cleanup)
    print("Cerrando RAG service...")

app = FastAPI(title="Proyecto1V2", lifespan=lifespan)

def log_consumption(session_id: str, query: str, tokens_used: int, cost_estimated: float, latency_sec: float):
    if not LOG_FILE.exists():
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["session_id", "timestamp", "query", "tokens_used", "cost_estimated", "latency_sec"])
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            session_id,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            query,
            tokens_used,
            cost_estimated,
            latency_sec
        ])

@app.get("/")
async def read_root():
    return {"message": "RAG API funcionando en Render"}


@app.post("/question", response_model=QuestionResponse)
async def process_question(request: QuestionRequest):
    # Tu lógica aquí
    if not rag_service.initialized:
        raise HTTPException(status_code=503, detail="RAG no está inicializado")
    response = rag_service.answer_question(request.question, request.model_provider, request.top_k, request.mode)
    #consumption = response["consumption"]
    # Extraer datos de consumo
    consumption = response.get("consumption", {})
    tokens_used = consumption.get("tokens_used", 0)
    cost_estimated = consumption.get("cost_estimated", 0.0)
    latency_sec = consumption.get("latency_sec", 0.0)
    #log_consumption(session_id=str(uuid.uuid4()), query=request.question, tokens_used=tokens_used, cost_estimated=cost_estimated, latency_sec= round(latency_sec,2))
    
    return QuestionResponse(
        answer=response["answer"],
        model_provider=request.model_provider,
        sources=[src for src in response["sources"]],
        mode=request.mode if hasattr(request, 'mode') else "breve",
        confidence=0.85
    )

from fastapi import UploadFile, File, HTTPException
from pathlib import Path

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Solo archivos PDF permitidos")
    
    # Guardar PDF en disco
    upload_dir = Path("/app/data")
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename
    
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Procesar PDF (extraer chunks)
    chunks = process_pdf_with_langchain(file_path)
    
    # Preparar chunks para ChromaDB
    docs = prepare_docs_for_chroma(chunks)
    
    # Guardar en ChromaDB
    rag_service.embedding_service.add_documents(docs)
    
    rag_service.initialized = True  # actualizar flag si es necesario
    
    return {
        "message": f"Archivo {file.filename} guardado y chunks indexados en ChromaDB",
        "chunks_indexed": len(docs)
    }


@app.post("/rebuild_index")
async def rebuild_index():
    """Fuerza reconstrucción completa del índice"""
    try:
        data_folder = Path("/app/data")
        rag_service.initialize_from_pdfs(data_folder, force=True)
        return {
            "status": "success",
            "message": "Índice reconstruido completamente"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Optional health endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}
