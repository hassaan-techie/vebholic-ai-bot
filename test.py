import os

# Check for missing dependencies
try:
    import streamlit as st
    from langchain.chains import RetrievalQA
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import MongoAtlas
    from langchain.llms import LlamaCpp
    from pymongo import MongoClient
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(f"Missing module: {e.name}. Please install it using 'pip install {e.name}'")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# App Configuration
st.set_page_config(page_title="RAG App with MongoDB and Llama", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) with MongoDB and Llama")

# Sidebar for configuration
st.sidebar.header("Configuration")
mongo_uri = st.sidebar.text_input("MongoDB Atlas URI", "mongodb+srv://<username>:<password>@cluster0.mongodb.net")
db_name = st.sidebar.text_input("Database Name", "erp_system")
embedding_model_name = st.sidebar.selectbox(
    "Embedding Model", ["all-MiniLM-L6-v2", "sentence-transformers/paraphrase-mpnet-base-v2"]
)
retrieval_top_k = st.sidebar.slider("Top-K Results", 1, 10, 5)

# Load Embedding Model
st.sidebar.write("Loading embedding model...")
try:
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    st.sidebar.success("Embedding model loaded!")
except Exception as e:
    st.sidebar.error(f"Error loading embedding model: {e}")

# Connect to MongoDB Atlas
st.sidebar.write("Connecting to MongoDB Atlas...")
try:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    st.sidebar.success("Connected to MongoDB Atlas!")
except Exception as e:
    st.sidebar.error(f"Error connecting to MongoDB: {e}")

# Use MongoAtlas as VectorStore for all collections
st.sidebar.write("Setting up MongoAtlas VectorStore...")
try:
    vectorstores = []
    for collection_name in db.list_collection_names():
        collection = db[collection_name]
        vectorstore = MongoAtlas(
            collection=collection,
            embedding_function=embeddings
        )
        vectorstores.append(vectorstore)
    st.sidebar.success("MongoAtlas VectorStores ready for all collections!")
except Exception as e:
    st.sidebar.error(f"Error setting up VectorStores: {e}")

# Combine retrievers for all collections
try:
    retrievers = [vs.as_retriever(search_type="similarity", search_kwargs
                                  ={"k": retrieval_top_k}) for vs in vectorstores]
    qa_chains = [RetrievalQA(llm=LlamaCpp(model_path="./models/llama.bin", n_ctx=512), retriever=retriever) for retriever in retrievers]
except Exception as e:
    st.error(f"Error initializing retrieval chains: {e}")

# Query Interface
query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        try:
            st.write("Fetching answer...")
            answers = [qa_chain.run(query) for qa_chain in qa_chains]
            combined_answers = "\n---\n".join([f"**Collection:** {db.list_collection_names()[i]}\n**Answer:** {answers[i]}" for i in range(len(answers))])
            st.write(combined_answers)
        except Exception as e:
            st.error(f"Error processing query: {e}")
    else:
        st.error("Please enter a query.")

# Chat History
def log_chat(query, responses):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.session_state["chat_history"].append({"query": query, "responses": responses})

if "chat_history" in st.session_state:
    st.write("### Chat History")
    for idx, chat in enumerate(st.session_state.chat_history):
        st.write(f"**Q{idx+1}:** {chat['query']}")
        for i, response in enumerate(chat['responses']):
            st.write(f"**Collection {db.list_collection_names()[i]} A{idx+1}.{i+1}:** {response}")
