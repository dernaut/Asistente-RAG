# Asistente-RAG
Creación Asistente RAG donde se implementa lo siguiente:
- OllamaLLM para utilizar los modelos opensource.
- Ollama Embeddings (Para búsqueda de relación semantica).
- ChromaDB para almacenar los valores vectoriales de los fragmentos de información.
- PrompTemplate para hacer consultas más sofisticadas y cerrar más el sistema de consultas.
- RunnablePassthrough para ejecutar la pregunta por medio de una cadena chain.
- Retriever MultiQueryRetriever para obtener información con un modelo, un prompt personalizado y un base retriever donde se contiene la información al principio.
- Retriever EnsembleRetriever para combinar dos tipos de retriever y así refinar el módelo de búsqueda, se agrega un similarity retriever.
- Streamlit como herramienta para crear la interfaz de Chat.
