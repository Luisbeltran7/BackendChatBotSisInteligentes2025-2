import fitz  # PyMuPDF
from pathlib import Path

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter

def process_pdf_with_langchain(pdf_path: Path, chunk_size: int = 500, chunk_overlap: int = 100):
    """
    Carga un PDF, extrae texto y lo divide en chunks usando LangChain.
    
    Parámetros:
    - pdf_path: ruta al archivo PDF
    - chunk_size: tamaño máximo de cada chunk en caracteres
    - chunk_overlap: cantidad de caracteres que se solapan entre chunks
    
    Retorna lista de documentos (cada uno es un chunk con metadata)
    """
    # Cargar el PDF
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    
    # Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Prioriza divisiones naturales
    )
    
    chunks = text_splitter.split_documents(documents)
    
    return chunks


# Ejemplo de uso
if __name__ == "__main__":
    pdf_path = Path("../data/Idea proyecto.pdf")
    chunks = process_pdf_with_langchain(pdf_path)
    
    print(f"Total de chunks: {len(chunks)}")
    
    for i, chunk in enumerate(chunks[:3]):  # Mostrar los primeros 3
        print(f"\n--- Chunk {i+1} ---")
        print(f"Contenido: {chunk.page_content[:200]}...")
        print(f"Metadata: {chunk.metadata}")
