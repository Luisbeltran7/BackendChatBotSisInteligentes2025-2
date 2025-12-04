# Proyecto1V2 - FastAPI + RAG (Retrieval-Augmented Generation)

**Chatbot inteligente basado en IA que responde preguntas usando Retrieval-Augmented Generation (RAG) con procesamiento de documentos, embeddings vectoriales y múltiples proveedores de LLM.**

![Python 3.11](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green)
![OpenAI](https://img.shields.io/badge/OpenAI-API-red)
![Groq](https://img.shields.io/badge/Groq-API-orange)

## 🎯 Características Principales

✅ **Sistema RAG Completo**
- Extracción automática de documentos PDF
- Procesamiento y segmentación de texto (chunking)
- Generación de embeddings con OpenAI API
- Almacenamiento vectorial con ChromaDB

✅ **Inteligencia Artificial**
- Integración con múltiples proveedores LLM (OpenAI, Groq)
- Generación de respuestas contextuales
- Búsqueda semántica de documentos

✅ **API REST Moderna**
- Documentación interactiva con Swagger UI
- Validación automática de datos
- CORS habilitado para integraciones

✅ **Optimización de Producción**
- Embeddings precomputados (cacheados)
- Batch processing para eficiencia
- Bajo consumo de recursos en Render

✅ **Evaluación y Monitoreo**
- Gold Standard para validación
- Logging de consumo de API
- Métricas de rendimiento

## 📋 Descripción General

Este proyecto implementa un sistema completo de pregunta-respuesta que:

1. **Procesa documentos** → Extrae y estructura PDFs
2. **Genera embeddings** → Convierte texto a vectores semánticos
3. **Almacena en vector DB** → ChromaDB para búsqueda rápida
4. **Responde preguntas** → Busca documentos relevantes + LLM

## 🗂️ Estructura del Proyecto

```
Proyecto1V2/
│
├── 📁 src/                              # Código fuente principal
│   ├── main.py                          # FastAPI app
│   ├── models/
│   │   └── schemas.py                   # Pydantic schemas
│   └── services/
│       ├── embedding_service_chroma.py  # Embeddings + ChromaDB
│       ├── rag_service.py               # Lógica RAG
│       ├── pdf_service.py               # Procesamiento PDFs
│       └── modelClientFactory.py        # Factory de LLMs
│
├── 📁 scripts/                          # Herramientas auxiliares
│   ├── preparar_corpus.py               # Procesa PDFs
│   └── datasets/                        # PDFs
│
├── 📁 metricas y evaluacion/           # Evaluación
│   ├── preguntasGold.py                 # Script de test
│   └── PreguntasGold.csv                # Preguntas de oro
│
├── 📁 chroma_persist/                   # ChromaDB persistente
│   └── embeddings_precomputed.pkl       # Embeddings cacheados
│
├── 📁 logs/                             # Logs del sistema
├── 📁 tests/                            # Tests unitarios
├── 📁 env/                              # Entorno virtual
│
├── Dockerfile                           # Imagen Docker
├── docker-compose.yml                   # Servicios
├── requirements.txt                     # Dependencias
├── .env                                 # Variables de entorno
├── .gitignore                           # Git ignores
└── README.md                            # Este archivo
```

## 🚀 Inicio Rápido

### Opción 1: Docker (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/Luisbeltran7/BackendChatBotSisInteligentes2025-2.git
cd Proyecto1V2

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus claves API

# Ejecutar con Docker
docker compose up --build
```

La API estará en: `http://localhost:8000`

### Opción 2: Instalación Local

```bash
# Crear entorno virtual
python -m venv env
.\env\Scripts\Activate.ps1  # Windows
source env/bin/activate     # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables
cp .env.example .env
# Editar .env

# Ejecutar
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## ⚙️ Configuración

### Variables de Entorno (.env)

```env
# API Keys
OPENAI_API_KEY=sk-proj-xxx...
GROQ_API_KEY=gsk_xxx...

# Servidor
HOST=0.0.0.0
PORT=8000
ENV=development
DEBUG=true

# Embeddings
EMBEDDER_ENABLED=true
USE_OPENAI_EMBEDDINGS=true

# Logs
LOG_LEVEL=INFO
```

| Variable | Descripción | Requerido |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Clave API OpenAI | Sí |
| `GROQ_API_KEY` | Clave API Groq | No |
| `HOST` | Host del servidor | No (default: 0.0.0.0) |
| `PORT` | Puerto del servidor | No (default: 8000) |
| `ENV` | Entorno (development/production) | No |
| `EMBEDDER_ENABLED` | Cargar sentence-transformers | No |
| `USE_OPENAI_EMBEDDINGS` | Usar OpenAI embeddings | No (default: true) |

## 📡 API Endpoints

### Health Check
```http
GET /health
```
Respuesta:
```json
{"status": "ok"}
```

### Hacer una Pregunta
```http
POST /question
Content-Type: application/json

{
  "question": "¿Qué es la inteligencia artificial?",
  "model_provider": "openai",
  "mode": "detallada",
  "top_k": 3
}
```

Respuesta:
```json
{
  "question": "¿Qué es la inteligencia artificial?",
  "answer": "La inteligencia artificial es...",
  "sources": [
    {
      "document": "archivo.pdf",
      "page": 5,
      "relevance": 0.95
    }
  ],
  "confidence": 0.88
}
```

### Documentación Interactiva
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔧 Scripts y Herramientas

### 1. Procesamiento de PDFs
```bash
python scripts/preparar_corpus.py
```
- Extrae texto de PDFs
- Detecta títulos y negrillas
- Estructura contenido en Markdown
- Genera PDFs ordenados

### 2. Evaluación con Gold Standard
```bash
python "metricas y evaluacion/preguntasGold.py"
```
- Lee preguntas de referencia
- Genera respuestas usando la API
- Compara con Gold Standard
- Genera reportes en CSV

### 3. Análisis de Respuestas
```bash
python scripts/contadorNo.py
```

## 📊 Ejemplos de Uso

### Con cURL
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

### Con Python
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
result = response.json()

print(f"Pregunta: {result['question']}")
print(f"Respuesta: {result['answer']}")
print(f"Confianza: {result['confidence']}")
```

### Con JavaScript
```javascript
const response = await fetch('http://localhost:8000/question', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "¿Qué es machine learning?",
    model_provider: "openai",
    mode: "detallada",
    top_k: 3
  })
});

const data = await response.json();
console.log(data.answer);
```

## 🧪 Testing

### Ejecutar Tests
```bash
pytest tests/ -v
```

### Cobertura
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📈 Monitoreo y Logs

- **Logs de aplicación**: `logs/app.log`
- **Logs de consumo**: `logs/consumo_logs.csv`
- **Ver logs en tiempo real**:
```bash
docker compose logs -f
```

## 🚀 Despliegue en Render

### Pasos

1. **Preparar repositorio**:
   ```bash
   git add .
   git commit -m "Deploy to Render"
   git push
   ```

2. **En Render.com**:
   - Crear nuevo Web Service
   - Conectar repositorio GitHub
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`

3. **Configurar variables de entorno**:
   - `OPENAI_API_KEY`: Tu clave OpenAI
   - `ENV`: production
   - `EMBEDDER_ENABLED`: false
   - `USE_OPENAI_EMBEDDINGS`: true

### Optimización para Render

Para ahorrar recursos en la versión gratuita:
- ✅ Usa OpenAI embeddings (0 MB local)
- ✅ Embeddings precomputados cacheados
- ✅ `ENV=production` (desactiva extras)
- ✅ Batch processing eficiente

## 🔒 Seguridad

- Las claves API se cargan desde `.env` (nunca en código)
- `.env` está en `.gitignore`
- CORS habilitado solo para dominios configurados
- Rate limiting recomendado en producción

## 🐛 Solución de Problemas

### Error: "No module named 'openai'"
```bash
pip install openai
```

### Error: "OPENAI_API_KEY not configured"
```bash
# Verifica que .env existe y contiene:
OPENAI_API_KEY=sk-proj-xxx...
```

### Puerto 8000 en uso
```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

### ChromaDB con errores
```bash
# Limpiar base de datos
Remove-Item -Path "chroma_persist" -Recurse -Force
# Reiniciar servidor
```

## 📚 Recursos Adicionales

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [OpenAI API](https://platform.openai.com/)
- [Groq API](https://console.groq.com/)
- [ChromaDB](https://www.trychroma.com/)
- [RAG Overview](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)

## 🤝 Contribución

1. Fork el proyecto
2. Crear rama feature: `git checkout -b feature/amazing-feature`
3. Commit cambios: `git commit -m 'Add amazing feature'`
4. Push a rama: `git push origin feature/amazing-feature`
5. Abrir Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👤 Autor

**Luis Beltrán**
- GitHub: [@Luisbeltran7](https://github.com/Luisbeltran7)
- Repositorio: [BackendChatBotSisInteligentes2025-2](https://github.com/Luisbeltran7/BackendChatBotSisInteligentes2025-2)

## 📧 Soporte

Para preguntas o reportar issues, por favor:
- Abrir un issue en GitHub
- Enviar email: [tu-email]

---

**Última actualización**: Diciembre 2025  
**Versión**: 2.0.0  
**Estado**: ✅ Producción Ready

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