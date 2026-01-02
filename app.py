import streamlit as st
import os

from rag_system import query_rag, get_retriever_info, get_data_vectorstore
from fragment_documents_manager import get_fragments_by_source, load_documents_and_create_vectorstore
# Configuración de la página
st.set_page_config(
    page_title="Sistema RAG - Asistente Legal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Funciones de diálogo
@st.dialog("📄 Fragmentos de documento almacenados en ChromaDB", width="large", dismissible=True)
def get_documents_fragments():
    data = get_data_vectorstore()
    docs = data["documents"]
    ids = data["ids"]
    metas = data["metadatas"]
    previous_doc = None
    if docs:
        for meta in metas:
            if meta.get('source') != previous_doc:
                fragments = get_fragments_by_source(data, meta.get('source'))
                with st.expander(f"***Documento***: {os.path.basename(meta.get('source'))}"):
                    st.json({
                        "source": meta.get('source'),
                        "total_pages": meta.get('total_pages'),
                        "creationdate": meta.get('creationdate'),
                        "moddate": meta.get('moddate'),
                        "author": meta.get('author'),
                        "total_fragments": len(fragments)
                    })
                    for i, fragment in enumerate(fragments, 1):
                        with st.expander(f"📄 Fragmento {i}"):
                            st.text(fragment)
            previous_doc = meta.get('source')
    else:
        st.warning("No hay documentos almacenados en la base de datos.")

    # Layout principal con columnas
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Cerrar", type="primary", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🔄️Actualizar Fragmentos ChromaDB", use_container_width=True):
            with st.spinner("⚙️Actualizando base de datos ChromaDB, cargando documentos y creando vector store..."):
                load_documents_and_create_vectorstore()
                st.rerun()


# Título
st.title("⚖️ Sistema RAG - Asistente Legal")
st.divider()

# Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar simplificado
with st.sidebar:
    st.header("📋 Información del Sistema")
    
    # Información del retriever
    retriever_info = get_retriever_info()
    
    st.markdown("**🔍 Retriever:**")
    st.info(f"Tipo: {retriever_info['tipo']}")
    
    st.markdown("**🤖 Modelos:**")
    st.warning(f"Modelo Consultas: {retriever_info['modelo_consultas']}")
    st.warning(f"Modelo Respuestas: {retriever_info['modelo_respuestas']}")
    
    st.divider()

    if st.button("🗂️ Consultar Fragmentos de documento almacenados", type="secondary", use_container_width=True):
        get_documents_fragments()

    if st.button("🗑️ Limpiar Chat", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Layout principal con columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 💬 Chat")
    
    # Mostrar historial de mensajes
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

with col2:
    st.markdown("### 📄 Documentos Relevantes")
    
    # Mostrar documentos de la última consulta
    if st.session_state.messages:
        last_message = st.session_state.messages[-1]
        if last_message["role"] == "assistant" and "docs" in last_message:
            docs = last_message["docs"]
            
            if docs:
                for doc in docs:
                    with st.expander(f"📄 Fragmento {doc['fragmento']}", expanded=False):
                        st.markdown(f"**Fuente:** {doc['fuente']}")
                        st.markdown(f"**Página:** {doc['pagina']}")
                        st.markdown("**Contenido:**")
                        st.text(doc['contenido'])


# Input del usuario
if prompt := st.chat_input("Escribe tu consulta sobre contratos de arrendamiento..."):
    # Añadir mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generar respuesta
    with st.spinner("🔍 Analizando..."):
        response, docs = query_rag(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response, "docs": docs})
    
    # Recargar para mostrar los nuevos mensajes
    st.rerun()

# Footer
st.divider()
st.markdown(
    f"<div style='text-align: center; color: #666;'>🏛️ Asistente Legal con {retriever_info['tipo']} Retriever</div>", 
    unsafe_allow_html=True
)