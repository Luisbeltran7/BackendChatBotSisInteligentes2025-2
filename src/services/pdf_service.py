from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf_with_langchain(pdf_path: Path, chunk_size: int = 500, chunk_overlap: int = 100):
    """Carga un PDF y lo divide en chunks"""
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def process_all_pdfs(folder_path: Path):
    """Procesa todos los PDFs de una carpeta"""
    all_chunks = []
    for pdf_file in folder_path.glob("*.pdf"):
        chunks = process_pdf_with_langchain(pdf_file)
        all_chunks.extend(chunks)
    return all_chunks

def prepare_docs_for_chroma(chunks):
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            'id': f"chunk_{i}",
            'text': chunk.page_content,
            'metadata': {
                'source': chunk.metadata.get('source','unknown'),
                'page': chunk.metadata.get('page', -1),
                'chunk_id': f"chunk_{i}"
            }
        })
    return docs

    
