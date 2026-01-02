# Configuración de modelos
EMBEDDING_MODEL = "bge-m3:latest"
QUERY_MODEL = "llama3.1:latest"
GENERATION_MODEL = "deepseek-r1:latest"

# Configuración del vector store
CHROMA_DB_PATH = "C:\\Users\\jgiraldoc\\proyecto_langchain_python3_13\\Tema 3\\chroma_db"

# Configuración documentos
DOCUMENTS_PATH = "C:\\Users\\jgiraldoc\\proyecto_langchain_python3_13\\Tema 3\\contratos"

# Configuración del retriever
SEARCH_TYPE = "mmr"
SEARCH_TYPE_K = 2
MMR_DIVERSITY_LAMBDA = 0.7
MMR_FETCH_K = 20

# Configuración alternativa para retriever hibrido (similarity + mmr)
ENABLE_HYBRID_RETRIEVER = True
SIMILARITY_THRESHOLD = 0.75