import chromadb
import pickle
import numpy as np
from pathlib import Path
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Configuración
EMBEDDER_ENABLED = os.getenv("EMBEDDER_ENABLED", "false").lower() == "true"
IS_PRODUCTION = os.getenv("ENV", "development").lower() == "production"
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() == "true"

SentenceTransformer = None
if EMBEDDER_ENABLED and not IS_PRODUCTION:
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("✅ sentence-transformers cargado (DESARROLLO)")
    except ImportError as e:
        logger.warning(f"⚠️ sentence-transformers no disponible: {e}")
        SentenceTransformer = None
else:
    if IS_PRODUCTION:
        logger.info("⏭️ sentence-transformers deshabilitado (PRODUCCIÓN)")
    else:
        logger.info("⏭️ EMBEDDER_ENABLED=false, sentence-transformers no se cargará")



class EmbeddingServiceChroma:
    def __init__(self, persist_dir: str = "./chroma_persist"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        
        # ChromaDB en memoria (más ligero para Render)
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="document_chunks")
        
        # Flag para embeddings precomputados
        self.embeddings_loaded = False
        self.precomputed_data = None
        
        # Embedder solo si está habilitado
        self.embedder = None
        if SentenceTransformer and EMBEDDER_ENABLED:
            logger.info("Cargando modelo de embeddings...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Intentar cargar embeddings precomputados al inicializar
        self._try_load_precomputed()
    
        # Inicializar cliente de OpenAI si está configurado
        self.openai_client = None
        if USE_OPENAI_EMBEDDINGS and os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("✅ OpenAI embeddings disponible")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando OpenAI: {e}")

    def _try_load_precomputed(self):
        """Intenta cargar embeddings precomputados al inicio"""
        embeddings_path = self.persist_dir / "embeddings_precomputed.pkl"
        if embeddings_path.exists():
            try:
                with open(embeddings_path, 'rb') as f:
                    self.precomputed_data = pickle.load(f)
                
                # Agregar a ChromaDB (SÚPER RÁPIDO)
                self.collection.add(
                    documents=self.precomputed_data['texts'],
                    embeddings=self.precomputed_data['embeddings'].tolist(),
                    metadatas=self.precomputed_data['metadatas'],
                    ids=self.precomputed_data['ids']
                )
                self.embeddings_loaded = True
                print(f"✅ Cargados {len(self.precomputed_data['ids'])} embeddings precomputados")
            except Exception as e:
                print(f"⚠️ Error cargando precomputados: {e}")
    
    def add_documents(self, docs_with_metadata: list):
        """
        docs_with_metadata: lista de dicts con {
            'id': str,
            'text': str,
            'metadata': dict,
        }
        Primero busca embeddings precomputados para evitar replicar el trabajo.
        """
        embeddings = []
        
        # 🚀 Prioridad 1: Usar embeddings precomputados si existen
        if self.embeddings_loaded and self.precomputed_data:
            logger.info("⚡ Usando embeddings precomputados existentes")
            embeddings = self.precomputed_data['embeddings'].tolist()
        else:
            # Prioridad 2: Generar con OpenAI en batches
            batch_size = 100  # Procesar en lotes de 100
            
            if self.openai_client:
                logger.info(f"🔄 Generando embeddings con OpenAI para {len(docs_with_metadata)} docs (en lotes de {batch_size})...")
                
                for i in range(0, len(docs_with_metadata), batch_size):
                    batch = docs_with_metadata[i:i+batch_size]
                    batch_texts = [doc['text'] for doc in batch]
                    
                    try:
                        # OpenAI puede procesar múltiples textos en una llamada
                        response = self.openai_client.embeddings.create(
                            input=batch_texts,
                            model="text-embedding-3-small"
                        )
                        # Ordenar embeddings según el orden de respuesta
                        for data in response.data:
                            embeddings.append(data.embedding)
                        
                        logger.info(f"✅ Lote {i//batch_size + 1}/{(len(docs_with_metadata)-1)//batch_size + 1} completado")
                    except Exception as e:
                        logger.error(f"Error con OpenAI lote {i//batch_size + 1}: {e}")
                        raise
            elif self.embedder:
                # Fallback: Generar con sentence-transformers si OpenAI no está disponible
                logger.info("🔄 Generando embeddings con sentence-transformers")
                embeddings = [self.embedder.encode(doc['text']) for doc in docs_with_metadata]
            else:
                raise Exception("❌ OpenAI no disponible y sentence-transformers está deshabilitado. "
                              "Configure OPENAI_API_KEY o EMBEDDER_ENABLED=true.")
            
            # Guardar embeddings para próxima vez (solo si no eran precomputados)
            self._save_precomputed_embeddings(docs_with_metadata, embeddings)
        
        # Agregar a ChromaDB
        self.collection.add(
            documents=[doc['text'] for doc in docs_with_metadata],
            embeddings=embeddings,
            metadatas=[doc['metadata'] for doc in docs_with_metadata],
            ids=[doc['id'] for doc in docs_with_metadata]
        )
    
    def _save_precomputed_embeddings(self, docs_with_metadata: list, embeddings: list):
        """Guarda embeddings para usar en próximos restarts"""
        embeddings_path = self.persist_dir / "embeddings_precomputed.pkl"
        try:
            data = {
                'embeddings': np.array(embeddings),
                'texts': [doc['text'] for doc in docs_with_metadata],
                'metadatas': [doc['metadata'] for doc in docs_with_metadata],
                'ids': [doc['id'] for doc in docs_with_metadata]
            }
            with open(embeddings_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Embeddings guardados: {embeddings_path}")
        except Exception as e:
            print(f"⚠️ No se pudieron guardar embeddings: {e}")
    
    def query(self, text: str, n_results: int = 5):
        """
        Query usando OpenAI embeddings (muy ligero, sin sentence-transformers).
        Fallback a sentence-transformers si OpenAI no está disponible.
        """
        embedding = None
        
        # Prioridad 1: OpenAI embeddings (0MB local)
        if self.openai_client:
            try:
                logger.info("Generando embedding con OpenAI...")
                print("Generando embedding con OpenAI...")
                response = self.openai_client.embeddings.create(
                    input=text,
                    model="text-embedding-3-small"  # Super ligero y rápido
                )
                embedding = response.data[0].embedding
            except Exception as e:
                logger.warning(f"Error con OpenAI embeddings: {e}. Usando fallback...")
                embedding = None
        
        # Fallback: sentence-transformers si está disponible
        if embedding is None and self.embedder:
            logger.info("Generando embedding con sentence-transformers...")
            print("Generando embedding con sentence-transformers...")
            embedding = self.embedder.encode(text)
        
        # Error si no hay forma de generar embedding
        if embedding is None:
            raise Exception(
                "❌ No se puede generar embedding. "
                "Opción 1: Configure OPENAI_API_KEY. "
                "Opción 2: Configure EMBEDDER_ENABLED=true para sentence-transformers."
            )
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results

    def generate_precomputed_file(self, docs_with_metadata: list):
        """Método para generar embeddings precomputados (ejecutar UNA VEZ local)"""
        if not self.embedder:
            raise Exception("❌ sentence-transformers no disponible. Configure EMBEDDER_ENABLED=true")
        
        logger.info("🧠 Generando embeddings precomputados...")
        embeddings = [self.embedder.encode(doc['text']) for doc in docs_with_metadata]
        
        data = {
            'embeddings': np.array(embeddings),
            'texts': [doc['text'] for doc in docs_with_metadata],
            'metadatas': [doc['metadata'] for doc in docs_with_metadata],
            'ids': [doc['id'] for doc in docs_with_metadata]
        }
        
        embeddings_path = self.persist_dir / "embeddings_precomputed.pkl"
        with open(embeddings_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"✅ Archivo generado: {embeddings_path} ({data['embeddings'].nbytes/1024/1024:.1f}MB)")
        return embeddings_path
