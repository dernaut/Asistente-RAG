from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st

from config import *
from prompts import *


@st.cache_resource
def initialize_rag_system():

    # Vector Store
    vectorstore = Chroma(
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DB_PATH
    )

    # Modelos
    llm_query = OllamaLLM(model=QUERY_MODEL, temperature=0)
    llm_generation = OllamaLLM(model=GENERATION_MODEL, temperature=0)

    # Retriever MMR (Maximal Marginal Relevance)
    base_retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": SEARCH_TYPE_K,
            "lambda_mult": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    #Retriever adicional con similarity para comparar
    similarity_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_TYPE_K}
    )


    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_query,
        prompt=multi_query_prompt
    )

    # Ensemble Retriever hibrido (similarity + mmr)
    if ENABLE_HYBRID_RETRIEVER:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[mmr_multi_retriever, similarity_retriever],
            weights=[0.7, 0.3], # mayor peso al MMR
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        final_retriever = ensemble_retriever
    else:
        final_retriever = mmr_multi_retriever

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Funcion para formatear y preprocesar los documentos recuperados
    def format_documents(docs):
        formatted_docs = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split('\\')[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f"- Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f"- Página: {doc.metadata['page']}"
                
            content = doc.page_content.strip()
            formatted_docs.append(f"{header}\n{content}")

        return "\n\n".join(formatted_docs)


    # Cadena RAG
    rag_chain = (
        {
            "context": final_retriever | format_documents,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, final_retriever


def query_rag(question: str):
    try:
        ragchain, retriever = initialize_rag_system()

        # Obtener la respuesta del sistema RAG
        response = ragchain.invoke(question)

        # Obtener documentos para mostrarlos
        docs = retriever.invoke(question)

        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_TYPE_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get("source", "No especificada").split('\\')[-1],
                "pagina": doc.metadata.get("page", "No especificada")
            }
            docs_info.append(doc_info)

        return response, docs_info
    except Exception as e:
        return f"Error al procesar la consulta: {str(e)}", []


def get_retriever_info():
    """Obtiene información sobre la configuración del retriever."""
    return {
        "modelo_consultas": QUERY_MODEL,
        "modelo_respuestas": GENERATION_MODEL,
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery Retriever" + (" Hybrid" if ENABLE_HYBRID_RETRIEVER else ""),
        "documentos": SEARCH_TYPE_K,
        "diversidad_lambda": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_RETRIEVER else "N/A"
    }