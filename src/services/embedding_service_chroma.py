import chromadb
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

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
        
        # Embedder solo para fallback (query individual)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Intentar cargar embeddings precomputados al inicializar
        self._try_load_precomputed()
    
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
        """
        # 🚀 Prioridad 1: Usar embeddings precomputados
        if self.embeddings_loaded and self.precomputed_data:
            print("⚡ Usando embeddings precomputados (sentence-transformers DESACTIVADO)")
            embeddings = self.precomputed_data['embeddings']
        else:
            # Fallback: Generar con sentence-transformers
            print("🔄 Generando embeddings con sentence-transformers")
            embeddings = [self.embedder.encode(doc['text']) for doc in docs_with_metadata]
        
        # Guardar embeddings para próxima vez (opcional)
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
        """Query mantiene sentence-transformers (solo para la pregunta del usuario)"""
        embedding = self.embedder.encode(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results

    def generate_precomputed_file(self, docs_with_metadata: list):
        """Método para generar embeddings precomputados (ejecutar UNA VEZ local)"""
        print("🧠 Generando embeddings precomputados...")
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
        
        print(f"✅ Archivo generado: {embeddings_path} ({data['embeddings'].nbytes/1024/1024:.1f}MB)")
        return embeddings_path
