
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from retrievers import create_retrieval_chain
from bson import ObjectId
import logging
import os
import warnings
warnings.filterwarnings("ignore")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ATLAS_CONNECTION_STRING = "mongodb+srv://hassan:wuW8ThO3mPMsDlNR@equipt-dev-cluster1.qf5b6.mongodb.net/equipt_master?retryWrites=true&w=majority"
QDRANT_URL = "https://bdf7c42a-704c-4b7d-9d0f-15ab69b2d75c.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "v310ZMoFcKZjmEPG2vynIhttlYTCCh37ww01sfJyVJbZIAmm1uppDQ"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Initialize Qdrant Client and Embedding Model
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


# Function to convert MongoDB ObjectId to string
def convert_objectid_to_string(doc):
    if isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, dict):
        return {key: convert_objectid_to_string(value) for key, value in doc.items()}
    elif isinstance(doc, list):
        return [convert_objectid_to_string(item) for item in doc]
    else:
        return doc


def connect_to_mongodb():
    try:
        client = MongoClient(ATLAS_CONNECTION_STRING)
        db = client["equipt_master"]
        logging.info("Connected to MongoDB successfully.")
        return db
    except PyMongoError as e:
        logging.error("Failed to connect to MongoDB: %s", e)
        raise

def embed_collection_with_relations(collection_name, documents, db):
    """
    Embeds a collection with relations into Qdrant.

    Args:
    - collection_name (str): Name of the collection to embed.
    - documents (list): List of documents in the collection.
    - db (MongoDB database object): MongoDB database object.

    Returns:
    - None
    """
    Batch_size = 100
    points = []
    collection_vector_name = f"vector_{collection_name}"

    # Check if Qdrant collection exists
    if collection_vector_name not in [collection[0] for collection in qdrant_client.get_collections()]: 
    # Create Qdrant collection if it does not exist
      qdrant_client.create_collection(
          collection_name=collection_vector_name,
          vectors_config=VectorParams(size=384, distance=Distance.COSINE),
          on_disk_payload=True,
          timeout=60
      )

    for doc in documents:
        doc = convert_objectid_to_string(doc)

        if collection_name == "serializedAsset":
            related_product_category = db["productCategory"].find_one({"_id": doc.get("productCategory", "")})
            related_warehouse = db["warehouses"].find_one({"_id": doc.get("warehouse", "")})
            all_text = f"{doc.get('assetNumber', '')} {doc.get('currentOwnerType', '')} {related_product_category or ''} {related_warehouse or ''}"
        elif collection_name in ["rentalManagement", "invoice"]:
            related_customer = db["customerAccount"].find_one({"_id": doc.get("customerAccount", "")})
            related_products = db["products"].find_one({"_id": doc.get("product", "")})
            related_service = db["service"].find_one({"_id": doc.get("service", "")})
            related_packages = db["packages"].find_one({"_id": doc.get("packages", "")})
            all_text = f"{doc.get('invoiceNumber', '')} {related_customer or ''} {related_products or ''} {related_service or ''} {related_packages or ''}"
        else:
            all_text = ' '.join(str(value) for value in doc.values())

        embedding = embedding_model.encode(all_text).tolist()
        import uuid
        point_id = str(uuid.uuid4())
        points.append(PointStruct(id=point_id, vector=embedding, payload=doc))

        if len(points) >= Batch_size:
            qdrant_client.upsert(collection_name=collection_vector_name, points=points)
            points = []

    # Upsert remaining points
    if points:
        qdrant_client.upsert(collection_name=collection_vector_name, points=points)
        logging.info(f"Inserted {len(points)} embeddings into Qdrant collection '{collection_vector_name}'.")



def query_related_collections(db, main_collection, query):
    # Define related collections for each main collection
    related_collections = {
        "serializedAsset": ["products", "warehouse", "productCategory"],
        "rentalManagement": ["customerAccount", "warehouse", "products", "service", "packages"],
        "invoice": ["customerAccount", "warehouse", "products", "service", "packages"]
    }

    # Include the main collection in the query list
    collections_to_query = [main_collection] + related_collections.get(main_collection, [])

    results = []
    for collection_name in collections_to_query:
        qdrant_collection_name = f"vector_{collection_name}"
        try:
            qa_chain = create_retrieval_chain(qdrant_collection_name)
            response = qa_chain.invoke(query)
            results.append({
                "collection": collection_name,
                "response": response
            })
        except Exception as e:
            logging.warning(f"Failed to query collection '{collection_name}': {e}")

    return results
