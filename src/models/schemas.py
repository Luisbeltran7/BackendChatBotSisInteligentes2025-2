from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Modelos para el endpoint /question
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Pregunta del usuario")
    model_provider: str = Field(..., description="Proveedor del modelo LLM")
    mode: Optional[str] = Field(..., description="Modo de respuesta: breve o detallada")
    top_k: Optional[int] = Field(3, description="Número de chunks a recuperar")


class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Respuesta generada")
    model_provider: Optional[str] = Field(..., description="Proveedor del modelo LLM usado")
    sources: List[str] = Field(default=[], description="Fuentes utilizadas")
    mode: str = Field(..., description="Modo de respuesta usado")
    confidence: Optional[float] = Field(None, description="Nivel de confianza (opcional)")

# Modelos para el endpoint /upload_pdf
class UploadResponse(BaseModel):
    message: str = Field(..., description="Mensaje de confirmación")
    filename: str = Field(..., description="Nombre del archivo subido")
    location: str = Field(..., description="Ruta donde se guardó")
    chunks_generated: Optional[int] = Field(None, description="Número de chunks generados")

# Modelos para información de chunks (opcional, para debugging)
class ChunkInfo(BaseModel):
    content: str = Field(..., description="Contenido del chunk")
    metadata: Dict[str, Any] = Field(default={}, description="Metadata del chunk")
    distance: Optional[float] = Field(None, description="Distancia semántica")

class SearchResult(BaseModel):
    query: str = Field(..., description="Query buscada")
    results: List[ChunkInfo] = Field(..., description="Chunks más relevantes")
    total_found: int = Field(..., description="Total de resultados")

# Modelos para health check
class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio")
    rag_initialized: bool = Field(..., description="Si RAG está inicializado")
    total_documents: Optional[int] = Field(None, description="Total de documentos procesados")
