from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import *



def load_documents_and_create_vectorstore():
    """Función para cargar documentos desde el directorio y crear el vector store."""
    loader = PyPDFDirectoryLoader(DOCUMENTS_PATH)
    documentos = loader.load()
    print(f"Se cargaron {len(documentos)} documentos desde el directorio de PDFs.")
    texto_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs_split = texto_splitter.split_documents(documentos)
    print(f"Se crearon {len(docs_split)} fragmentos de texto a partir de los documentos cargados.")

    Chroma.from_documents(
        docs_split,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=(CHROMA_DB_PATH)
    )
    print(f"Vector store creado y persistido en ChromaDB.")


def get_total_fragments(data, doc_source):
    """Función para contar el total de fragmentos de un documento dado su fuente."""
    metas = data["metadatas"]
    total_fragments = sum(1 for meta in metas if meta.get('source') == doc_source)
    return total_fragments

def get_fragments_by_source(data, doc_source):
    """Función para obtener los fragmentos de un documento dado su fuente."""
    docs = data["documents"]
    metas = data["metadatas"]
    fragments = [doc for doc, meta in zip(docs, metas) if meta.get('source') == doc_source]
    return fragments