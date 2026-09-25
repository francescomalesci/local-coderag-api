import chromadb
from chromadb.utils import embedding_functions

# 1. Stessa configurazione del database
CHROMA_DATA_PATH = "./chroma_db"

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Recuperiamo la collezione esistente creata da ingest.py
collection = client.get_collection(
    name="my_documents", 
    embedding_function=embedding_model
)

# 2. Funzione di ricerca semantica
def query_database(query_text, n_results=2):
    print(f"\nRicerca in corso per: '{query_text}'...")
    
    # ChromaDB vettorializza la stringa di testo e calcola la distanza 
    # geometrica con tutti i chunk salvati nel database.
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    return results

if __name__ == "__main__":
    # Test: facciamo una domanda specifica su YOLO
    domanda_utente = "Come funziona il sistema di bounding box?"
    
    risultati = query_database(domanda_utente)
    
    print("\n--- RISULTATI TROVATI ---")
    
    # Estraiamo i dati dal dizionario restituito da ChromaDB
    chunks_trovati = risultati['documents'][0]
    metadati_trovati = risultati['metadatas'][0]
    distanze = risultati['distances'][0]
    
    for i in range(len(chunks_trovati)):
        print(f"\nRisultato {i+1} (Distanza vettoriale: {distanze[i]:.4f}):")
        print(f"Fonte: {metadati_trovati[i]['source']} | Indice Blocco: {metadati_trovati[i]['chunk_index']}")
        print(f"Testo: {chunks_trovati[i]}")