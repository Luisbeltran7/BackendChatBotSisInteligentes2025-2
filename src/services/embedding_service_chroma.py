import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class EmbeddingServiceChroma:
    def __init__(self, persist_dir: str = "./chroma_persist"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="document_chunks")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
    def add_documents(self, docs_with_metadata: list):
        """
        docs_with_metadata: lista de dicts con {
          'id': str,
          'text': str,
          'metadata': dict,
        }
        """
        embeddings = [self.embedder.encode(doc['text']) for doc in docs_with_metadata]
        
        self.collection.add(
            documents=[doc['text'] for doc in docs_with_metadata],
            embeddings=embeddings,
            metadatas=[doc['metadata'] for doc in docs_with_metadata],
            ids=[doc['id'] for doc in docs_with_metadata]
        )

        
    def query(self, text: str, n_results: int = 5):
        embedding = self.embedder.encode(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results


