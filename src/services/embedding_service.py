from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path

class EmbeddingService:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
    
    def generate_embeddings(self, chunks):
        """Genera embeddings para los chunks"""
        self.chunks = chunks
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def build_index(self, embeddings):
        """Construye índice FAISS"""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def search(self, query: str, top_k: int = 3):
        """Busca chunks más relevantes para una query"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            results.append({
                "content": self.chunks[idx].page_content,
                "metadata": self.chunks[idx].metadata,
                "distance": float(distance)
            })
        return results
    
    def save_index(self, path: Path):
        """Guarda índice FAISS y chunks"""
        path.mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
            print(f"✓ Índice FAISS guardado en {path / 'index.faiss'}")
        
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"✓ Chunks guardados en {path / 'chunks.pkl'}")
    
    def load_index(self, path: Path) -> bool:
        """Carga índice FAISS y chunks. Retorna True si exitoso."""
        try:
            index_path = path / "index.faiss"
            chunks_path = path / "chunks.pkl"
            
            if not index_path.exists() or not chunks_path.exists():
                print(f"No se encontró índice guardado en {path}")
                return False
            
            self.index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                self.chunks = pickle.load(f)
            
            print(f"✓ Índice cargado: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"Error al cargar índice: {e}")
            return False