from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.vectorstores import Qdrant
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import os


ATLAS_CONNECTION_STRING = "mongodb+srv://hassan:wuW8ThO3mPMsDlNR@equipt-dev-cluster1.qf5b6.mongodb.net/equipt_master?retryWrites=true&w=majority"
QDRANT_URL = "https://bdf7c42a-704c-4b7d-9d0f-15ab69b2d75c.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "v310ZMoFcKZjmEPG2vynIhttlYTCCh37ww01sfJyVJbZIAmm1uppDQ"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Initialize Qdrant Client and Embedding Model
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def create_retrieval_chain(qdrant_collection_name):
    """Creates a retrieval chain using Qdrant and Ollama with a ChatPromptTemplate."""
    
    # 1. Define the Prompt Template:
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}

    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # 2. Initialize the Ollama Language Model:
    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434") 
    model = OllamaLLM(model="llama3.2:1b", base_url=ollama_host) 

    # 3. Create the RetrievalQAWithSourcesChain:
    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=qdrant_collection_name,
        embeddings=embeddings
    )
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=model, 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}  # Pass the prompt to the chain
    )

    return qa_chain
