import chromadb
from chromadb.utils import embedding_functions

CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="thesis_notes", embedding_function=embedding_model)