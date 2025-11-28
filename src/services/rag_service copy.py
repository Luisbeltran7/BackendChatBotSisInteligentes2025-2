import hashlib
from pathlib import Path
from src.services.pdf_service import process_all_pdfs
from src.services.embedding_service import EmbeddingService
from groq import Groq
import os
from src.services.embedding_service_chroma import EmbeddingServiceChroma



class RAGService:
    def __init__(self, index_path: Path = Path("/app/vector_store")):
        self.embedding_service = EmbeddingServiceChroma()
        self.initialized = False
        self.index_path = index_path
        self.indexed_files = {}
        
        # Inicializar cliente de Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("WARNING: GROQ_API_KEY no encontrada")
            self.groq_client = None
        else:
            self.groq_client = Groq(api_key=api_key)
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Calcula hash MD5 del archivo"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def _save_file_registry(self):
        """Guarda registro de archivos indexados"""
        import json
        registry_path = self.index_path / "file_registry.json"
        with open(registry_path, "w") as f:
            json.dump(self.indexed_files, f)
    
    def _load_file_registry(self):
        """Carga registro de archivos indexados"""
        import json
        registry_path = self.index_path / "file_registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                self.indexed_files = json.load(f)
    
    def needs_reindex(self, data_folder: Path) -> bool:
        """Verifica si hay archivos nuevos o modificados"""
        current_files = {}
        for pdf_file in data_folder.glob("*.pdf"):
            current_files[pdf_file.name] = self._get_file_hash(pdf_file)
        
        if current_files != self.indexed_files:
            print("Detectados cambios en los archivos")
            return True
        return False
    
    def try_load_existing_index(self) -> bool:
        """Intenta cargar índice existente. Retorna True si exitoso."""
        if self.embedding_service.load_index(self.index_path):
            self._load_file_registry()
            self.initialized = True
            return True
        return False
    
    def initialize_from_pdfs(self, data_folder: Path, force: bool = False):
        """Procesa PDFs solo si hay cambios, force=True, o no existe índice"""
        
        # Si no se fuerza, intentar cargar índice existente
        if not force and self.try_load_existing_index():
            # Verificar si hay cambios
            if not self.needs_reindex(data_folder):
                print("✓ Índice cargado desde disco. No hay cambios.")
                return
            else:
                print("Cambios detectados, reprocesando...")
        
        print(f"Procesando PDFs en {data_folder}...")
        chunks = process_all_pdfs(data_folder)
        
        if not chunks:
            print("No se encontraron chunks para procesar")
            return
        
        print(f"Generando embeddings para {len(chunks)} chunks...")
        embeddings = self.embedding_service.generate_embeddings(chunks)
        
        print("Construyendo índice FAISS...")
        self.embedding_service.build_index(embeddings)
        
        # Guardar índice y registro
        print("Guardando índice en disco...")
        self.embedding_service.save_index(self.index_path)
        
        # Actualizar registro de archivos
        self.indexed_files = {}
        for pdf_file in data_folder.glob("*.pdf"):
            self.indexed_files[pdf_file.name] = self._get_file_hash(pdf_file)
        self._save_file_registry()
        
        self.initialized = True
        print(f"✓ RAG inicializado con {len(chunks)} chunks de {len(self.indexed_files)} PDFs")
        
    def answer_question(self, question: str, top_k: int = 3, mode: str = "breve"):
        """Responde pregunta usando RAG + LLM con ChromaDB"""
        if not self.initialized:
            return {
                "answer": "El sistema RAG no está inicializado. Por favor, sube documentos PDF primero.",
                "sources": [],
                "context": []
            }

        # 1. Buscar chunks relevantes con ChromaDB
        print(f"Buscando contexto para: {question}")
        results = self.embedding_service.query(question, n_results=top_k)
        
        # Estructura de Chroma: resultados vienen dentro de listas anidadas por consultas/ids
        matched_texts = results.get('documents', [[]])[0]  # Lista de textos
        matched_metadatas = results.get('metadatas', [[]])[0]  # Lista de diccionarios

        if not matched_texts or not matched_metadatas:
            return {
                "answer": "No se encontró información relevante en los documentos.",
                "sources": [],
                "context": []
            }

        # 2. Construir el contexto concatenado para el prompt del LLM
        context = ""
        for i, (text, meta) in enumerate(zip(matched_texts, matched_metadatas)):
            src = meta.get('source', 'desconocido')
            page = meta.get('page', 'desconocida')
            context += f"[Fragmento {i+1} - Fuente: {src}, Página: {page}]:\n{text}\n\n"

        # 3. Crear el prompt para el LLM
        if mode == "detallada":
            system_prompt = (
                "Eres un asistente experto que responde preguntas basándose ÚNICAMENTE en el contexto proporcionado. "
                "Proporciona respuestas detalladas, bien estructuradas y completas. "
                "Si la información no está en el contexto, di que no tienes suficiente información."
            )
        else:  # modo breve
            system_prompt = (
                "Eres un asistente que responde preguntas de forma BREVE y CONCISA basándote ÚNICAMENTE en el contexto proporcionado. "
                "Máximo 2-3 oraciones. "
                "Si la información no está en el contexto, di que no tienes suficiente información."
            )

        user_prompt = (
            f"Contexto:\n{context}\n"
            f"Pregunta: {question}\n"
            "Responde basándote únicamente en el contexto anterior."
        )

        # 4. Llamar al LLM (Groq)
        if not self.groq_client:
            return {
                "answer": "Error: No se pudo conectar al servicio LLM. Verifica GROQ_API_KEY.",
                "sources": [meta.get("source", "desconocido") for meta in matched_metadatas],
                "context": matched_texts
            }

        try:
            print("Generando respuesta con Groq...")
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",  # Modelo actualizado
                temperature=0.3,
                max_tokens=500 if mode == "breve" else 1500,
            )
            answer = chat_completion.choices[0].message.content

            return {
                "answer": answer,
                "sources": [meta.get("source", "desconocido") for meta in matched_metadatas],
                "context": matched_texts
            }

        except Exception as e:
            print(f"Error al llamar a Groq: {e}")
            return {
                "answer": f"Error al generar respuesta: {str(e)}",
                "sources": [meta.get("source", "desconocido") for meta in matched_metadatas],
                "context": matched_texts
            }

