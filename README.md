# Proyecto1V2 - FastAPI + RAG (Retrieval-Augmented Generation)

Proyecto de API REST basado en FastAPI con procesamiento de documentos PDF, almacenamiento vectorial (FAISS/Chroma) e integración con modelos de IA (OpenAI, Groq) para un sistema de pregunta-respuesta mejorado.

## Descripción del Proyecto

Este proyecto implementa un sistema RAG completo que permite:
- Procesar y estructurar documentos PDF
- Crear embeddings y almacenarlos en bases de datos vectoriales
- Responder preguntas basadas en contexto de documentos
- Integración con múltiples proveedores de LLM (OpenAI, Groq)
- Evaluación de calidad de respuestas mediante Gold Standard

## Estructura del Proyecto

```
proyecto1v2/
│
├── src/                           # Código fuente de la aplicación
│   ├── __init__.py               # Marca el directorio como paquete Python
│   ├── main.py                   # Punto de entrada de la aplicación FastAPI
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py            # Esquemas de datos Pydantic
│   └── services/
│       ├── embedding_service.py  # Servicio de embeddings
│       ├── embedding_service_chroma.py  # Embeddings con Chroma
│       ├── pdf_service.py        # Procesamiento de PDFs
│       ├── rag_service.py        # Lógica RAG principal
│       ├── modelClientFactory.py # Factory para clientes de LLM
│       └── __pycache__/
│
├── scripts/                       # Scripts de utilidad
│   ├── preparar_corpus.py        # Procesa y estructura PDFs
│   ├── contadorNo.py             # Analiza respuestas
│   ├── pdf_processing.py
│   └── datasets/                 # Datasets de PDFs
│       ├── corpuspdfs/           # PDFs originales
│       └── corpuspdfestructuradov2/  # PDFs procesados
│
├── metricas y evaluacion/        # Evaluación del sistema
│   ├── preguntasGold.py          # Script de evaluación con Gold Standard
│   └── PreguntasGold.csv         # Preguntas de referencia
│
├── tests/                         # Tests unitarios y de integración
├── docs/                          # Documentación del proyecto
├── data/                          # Datos y recursos
├── logs/                          # Logs del sistema
├── vector_store/                  # Almacenamiento de vectores
│   ├── index.faiss               # Índice FAISS
│   └── file_registry.json        # Registro de archivos
│
├── env/                           # Entorno virtual Python
├── Dockerfile                     # Configuración para construir la imagen Docker
├── docker-compose.yml             # Configuración de servicios Docker
├── requirements.txt               # Dependencias Python del proyecto
├── .env.example                   # Plantilla de variables de entorno
└── README.md                      # Este archivo
```

## Requisitos

- Docker y Docker Compose (recomendado)
- Python 3.11+ (para desarrollo local)
- Claves API:
  - OpenAI API Key (opcional, si usas OpenAI)
  - Groq API Key (opcional, si usas Groq)

## Dependencias Principales

Las dependencias están especificadas en `requirements.txt`:
- **fastapi** - Framework web moderno y rápido
- **uvicorn** - Servidor ASGI para Python
- **pandas** - Procesamiento de datos
- **PyMuPDF (fitz)** - Procesamiento de PDFs
- **chromadb** - Base de datos vectorial
- **faiss-cpu** - Biblioteca de búsqueda vectorial
- **sentence-transformers** - Modelos de embeddings
- **openai** - Cliente de OpenAI API
- **groq** - Cliente de Groq API
- **reportlab** - Generación de PDFs
- **requests** - Cliente HTTP

## Configuración y Ejecución

### Configurar Variables de Entorno

1. Crear archivo `.env` basado en `.env.example`:
```powershell
Copy-Item .env.example .env
```

2. Editar `.env` con tus valores:
```env
# API Keys
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here

# Configuración del servidor
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Configuración de embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=./vector_store

# Configuración de logs
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
```

### Usando Docker (recomendado)

1. **Construir y ejecutar con Docker Compose:**
```powershell
docker compose up --build
```

2. **O construir y ejecutar manualmente:**
```powershell
# Construir imagen
docker build -t proyecto-fastapi .

# Ejecutar contenedor
docker run -p 8000:8000 --env-file .env proyecto-fastapi
```

La API estará disponible en: http://localhost:8000

### Desarrollo Local

1. **Crear entorno virtual:**
```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

2. **Instalar dependencias:**
```powershell
pip install -r requirements.txt
```

3. **Ejecutar la aplicación:**
```powershell
uvicorn src.main:app --reload
```

La API estará disponible en: http://localhost:8000

## Endpoints Disponibles

### Rutas Básicas
- `GET /` - Ruta principal
  ```json
  {"message": "Hello, FastAPI from src.main!"}
  ```
- `GET /health` - Verificación de salud
  ```json
  {"status": "ok"}
  ```

### Endpoints RAG
- `POST /question` - Hacer una pregunta
  ```json
  {
    "question": "¿Cuál es la importancia de la IA?",
    "model_provider": "openai",
    "mode": "detallada",
    "top_k": 3
  }
  ```
  Respuesta:
  ```json
  {
    "question": "¿Cuál es la importancia de la IA?",
    "answer": "...",
    "sources": [...],
    "confidence": 0.85
  }
  ```

### Documentación Interactiva
- `GET /docs` - Swagger UI
- `GET /redoc` - ReDoc

## Workflows y Scripts

### Procesamiento de Documentos

1. **Preparar Corpus de PDFs:**
```powershell
python scripts/preparar_corpus.py
```
Esto:
- Extrae texto de PDFs
- Detecta títulos y negrillas
- Estructura el contenido en Markdown
- Genera PDFs ordenados

2. **Contar Respuestas Negativas:**
```powershell
python scripts/contadorNo.py
```

### Evaluación del Sistema

1. **Generar Respuestas con Gold Standard:**
```powershell
python 'metricas y evaluacion/preguntasGold.py'
```
Esto:
- Lee preguntas de referencia
- Llama a la API para obtener respuestas
- Guarda resultados en CSV

2. **Archivos de Resultados:**
- `preguntas_gold_con_respuestas_openai.csv` - Respuestas de OpenAI
- `preguntas_gold_con_respuestas_groq.csv` - Respuestas de Groq
- `preguntas_gold_con_respuestas_pelle.csv` - Respuestas personalizadas

## Variables de Entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `OPENAI_API_KEY` | Clave API de OpenAI | (requerida si usas OpenAI) |
| `GROQ_API_KEY` | Clave API de Groq | (requerida si usas Groq) |
| `HOST` | Host del servidor | 0.0.0.0 |
| `PORT` | Puerto del servidor | 8000 |
| `DEBUG` | Modo debug | false |
| `EMBEDDING_MODEL` | Modelo de embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| `VECTOR_STORE_PATH` | Ruta del almacén vectorial | ./vector_store |
| `LOG_LEVEL` | Nivel de logs | INFO |

## Instalación de Dependencias

### Windows

```powershell
# Crear entorno virtual
python -m venv env
.\env\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Para desarrollo (incluye herramientas adicionales)
pip install -r requirements.txt pytest black flake8
```

### Linux/Mac

```bash
# Crear entorno virtual
python3 -m venv env
source env/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso de la API

### Ejemplo con curl

```bash
curl -X POST http://localhost:8000/question \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué es la inteligencia artificial?",
    "model_provider": "openai",
    "mode": "detallada",
    "top_k": 3
  }'
```

### Ejemplo con Python

```python
import requests

url = "http://localhost:8000/question"
payload = {
    "question": "¿Cuál es el impacto de la IA en la educación?",
    "model_provider": "openai",
    "mode": "detallada",
    "top_k": 3
}

response = requests.post(url, json=payload)
print(response.json())
```

## Testing

### Ejecutar Tests

```powershell
pytest tests/ -v
```

### Cobertura de Tests

```powershell
pytest tests/ --cov=src --cov-report=html
```

## Logging y Monitoreo

Los logs se guardan en `logs/` con la siguiente estructura:
- `consumo_logs.csv` - Registro de consumo de API
- `app.log` - Logs generales de la aplicación

Para ver logs en tiempo real con Docker:
```powershell
docker compose logs -f
```

## Troubleshooting

### Docker

Si Docker no arranca:
```powershell
# Verificar estado del servicio
Get-Service -Name com.docker.service

# Iniciar el servicio si está detenido
Start-Service -Name com.docker.service
```

### Puertos

Si el puerto 8000 está en uso:
```powershell
# Verificar qué proceso usa el puerto
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object OwningProcess

# Cambiar puerto en docker-compose.yml
# ports:
#   - "8080:8000"
```

### API Keys

Si obtienes errores de autenticación:
1. Verifica que `.env` existe y tiene las claves correctas
2. Reinicia el contenedor: `docker compose restart`
3. Comprueba que las claves no tienen espacios en blanco

### Encoding de Archivos

Si hay problemas con caracteres especiales:
- Asegúrate de que los CSV se abren con encoding `utf-8`
- Los PDFs se procesan automáticamente con limpieza de caracteres

## Performance y Optimización

- **Embeddings**: Se cachean automáticamente en `vector_store/`
- **Búsqueda**: Usa FAISS para búsqueda vectorial rápida
- **Timeouts**: Configurados a 60 segundos para llamadas a API

## Contribución

1. Fork el proyecto
2. Crear rama feature: `git checkout -b feature/AmazingFeature`
3. Commit cambios: `git commit -m 'Add AmazingFeature'`
4. Push a rama: `git push origin feature/AmazingFeature`
5. Abrir Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## Contacto y Soporte

Para preguntas o reportar bugs, por favor crear un issue en el repositorio.