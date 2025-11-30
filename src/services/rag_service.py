import hashlib
from pathlib import Path
import time
from src.services.pdf_service import prepare_docs_for_chroma, process_all_pdfs
from groq import Groq
import os
from src.services.embedding_service_chroma import EmbeddingServiceChroma
from src.services.modelClientFactory import ModelClientFactory


class RAGService:
    def __init__(self, index_path: Path = Path("/app/vector_store")):
        self.embedding_service = EmbeddingServiceChroma()
        self.initialized = False
        self.index_path = index_path
        self.indexed_files = {}
        self.client_factory = ModelClientFactory()
        # Inicializar cliente de Groq

    
    def _get_file_hash(self, filepath: Path) -> str:
        """Calcula hash MD5 del archivo"""
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    
    def _load_file_registry(self):
        """Carga registro de archivos indexados"""
        import json
        registry_path = self.index_path / "file_registry.json"
        if registry_path.exists():
            with open(registry_path, "r") as f:
                self.indexed_files = json.load(f)
    
    def try_load_existing_index(self) -> bool:
        try:
            collection = self.embedding_service.client.get_collection("document_chunks")
            if collection.count() > 0:
                print(f"Índice encontrado con {collection.count()} documentos.")
                self.initialized = True
                print("RAG Service inicializado con ChromaDB. v2")
                return True
            return False
        except Exception as e:
            print(f"Error intentando cargar índice: {e}")
            return False
    
    def needs_reindex(self, data_folder: Path) -> bool:
        """Verifica si hay nuevos archivos o cambios que requieran reindexación"""
        self._load_file_registry()
        for pdf_file in data_folder.glob("*.pdf"):
            current_hash = self._get_file_hash(pdf_file)
            recorded_hash = self.indexed_files.get(str(pdf_file))
            if recorded_hash != current_hash:
                print(f"Cambio detectado en {pdf_file.name}")
                return True
        # Implementa si quieres detectar cambios, o simplemente fuerza reindexación según lógica propia
        return False  # Por simplicidad aquí siempre reindexa para evitar problemas
    
    def initialize_from_pdfs(self, data_folder: Path, force: bool = False):
        if not force and self.try_load_existing_index():
            return

        print(f"Procesando PDFs en {data_folder}...")
        chunks = process_all_pdfs(data_folder)
        docs = prepare_docs_for_chroma(chunks)
        
        print(f"Insertando {len(docs)} chunks en ChromaDB...")
        self.embedding_service.add_documents(docs)
        
        self.initialized = True
        print("RAG Service inicializado con ChromaDB.")


    def answer_question(self, question: str, provider: str, top_k: int = 3, mode: str = "breve" ):
        """Responde pregunta usando RAG + LLM con ChromaDB"""
        if not self.initialized:
            return {
                "answer": "El sistema RAG no está inicializado. Por favor, sube documentos PDF primero.",
                "sources": [],
                "context": []
            }
        def build_prompts(context: str, question: str, mode: str) -> tuple[str, str]:
            """
            Construye system_prompt y user_prompt según el modo solicitado.
            """

            if mode == "detallada":
                system_prompt = (
                    "Eres un asistente experto que responde siempre **usando exclusivamente la información proporcionada en el contexto**."
                    "No utilices conocimientos propios ni fuentes externas. Si la respuesta no está en el contexto, indica explícitamente: 'No hay suficiente información en los documentos proporcionados para responder a esta pregunta.' "
                    "Responde de manera didáctica, detallada y con referencias (Documento origen) sólo si aparecen en el contexto."
                )
                user_prompt = (
                    f"Contexto:\n{context}\n"
                    f"Pregunta: {question}\n"
                    "Responde utilizando únicamente información encontrada en el contexto anterior. No inventes ni completes con datos externos. "
                    "Si la información no está en los fragmentos dados, responde: 'No hay suficiente información en los documentos proporcionados para responder a esta pregunta.' "
                    "Incluye referencias (nombre del documento origen) solamente si aparecen explícitamente en el contexto. Divide la respuesta en párrafos si es necesario."
                )

            elif mode == "breve":
                system_prompt = (
                    "Eres un asistente que responde siempre de forma breve, clara y concisa. "
                    "Máximo 2-3 oraciones. Usa solo el contexto proporcionado y sé directo."
                )
                user_prompt = (
                    f"Contexto:\n{context}\n"
                    f"Pregunta: {question}\n"
                    "Proporciona una respuesta rápida y precisa."
                    "Responde utilizando únicamente información encontrada en el contexto anterior. No inventes ni completes con datos externos. "
                )

            print(mode, "prompts construidos.")
            return system_prompt, user_prompt

        start_time = time.time()

        # 1. Buscar chunks relevantes con ChromaDB
        print(f"Buscando contexto para: {question}")
        results = self.embedding_service.query(question, n_results=top_k)
        
        # Estructura de Chroma: resultados vienen dentro de listas anidadas por consultas/ids
        matched_texts = results.get('documents', [[]])[0]  # Lista de textos
        matched_metadatas = results.get('metadatas', [[]])[0]  # Lista de diccionarios

        print(f"Encontrados {len(matched_texts)} fragmentos relevantes.")
        print("Metadatas:", matched_metadatas)

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
        system_prompt, user_prompt = build_prompts(context, question, mode)

        # 4. Llamar al LLM (Groq)
        if not self.client_factory.get_client("groq"):
            return {
                "answer": "Error: No se pudo conectar al servicio LLM. Verifica GROQ_API_KEY.",
                "sources": [meta.get("source", "desconocido") for meta in matched_metadatas],
                "context": matched_texts
            }
        elif not self.client_factory.get_client("openai"):
            return {
                "answer": "Error: No se pudo conectar al servicio LLM. Verifica OPENAI_API_KEY.",
                "sources": [meta.get("source", "desconocido") for meta in matched_metadatas],
                "context": matched_texts
            }

        try:
            print(f"Generando respuesta con {provider} ...")
            if provider == "openai":
                chat_completion = self.client_factory.get_client("openai").chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=500 if mode == "breve" else 1500,
                )
                answer = chat_completion.choices[0].message.content
                tokens_used = chat_completion.usage.total_tokens

            else:  # por defecto usa Groq
                chat_completion = self.client_factory.get_client("groq").chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    model="llama-3.1-8b-instant",  # Modelo actualizado
                    temperature=0.3,
                    max_tokens=500 if mode == "breve" else 1500,
                )
                answer = chat_completion.choices[0].message.content
                tokens_used = getattr(chat_completion.usage, "total_tokens",None)

            end_time = time.time()
            latency = end_time - start_time
            # Calcular costo estimado
            cost_per_token = 0.00003  # ejemplo en dólares
            cost_estimated = tokens_used * cost_per_token if tokens_used else None
            
            return {
                "answer": answer,
                "sources": [meta.get("source", "desconocido") for meta in matched_metadatas],
                "context": matched_texts,
                "consumption": {
                    "tokens_used": tokens_used,
                    "cost_estimated": cost_estimated,
                    "latency_sec": latency,
                },
            }

        except Exception as e:
            print(f"Error al llamar a Groq: {e}")
            return {
                "answer": f"Error al generar respuesta: {str(e)}",
                "sources": [meta.get("source", "desconocido") for meta in matched_metadatas],
                "context": matched_texts
            }

