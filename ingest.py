import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Configurazione parametri
CHROMA_DATA_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# 2. Inizializzazione Vector Database
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
# Usa un modello di embedding leggero e ottimizzato (scaricato in automatico al primo avvio)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = client.get_or_create_collection(
    name="my_documents",
    embedding_function=embedding_model
)

# 3. Funzione di Chunking lineare
def get_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        
    return chunks

if __name__ == "__main__":
    # Test con testo di prova fittizio
    dummy_text = """La Computer Vision è un campo dell'intelligenza artificiale che addestra i computer a interpretare e comprendere il mondo visivo. 
    Utilizzando immagini digitali da fotocamere e video e modelli di deep learning, le macchine possono identificare e classificare accuratamente gli oggetti.
    YOLO (You Only Look Once) è un sistema di rilevamento oggetti in tempo reale. A differenza dei sistemi basati su classificatori, YOLO applica una singola rete neurale all'intera immagine.
    Questa rete divide l'immagine in regioni e prevede le bounding box e le probabilità per ciascuna regione."""
    
    print("Elaborazione testo in corso...")
    
    chunks = get_chunks(dummy_text, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Preparazione ID univoci e Metadati (fondamentali per il retrieval in Fase 3)
    ids = [f"doc1_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": "dummy_text.txt", "chunk_index": i} for i in range(len(chunks))]
    
    # Inserimento nel database (calcola gli embedding in background)
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Completato: {len(chunks)} chunk inseriti nel vector database.")
    print(f"\nEsempio Metadati Chunk 0: {metadatas[0]}")
    print(f"Testo Chunk 0: {chunks[0][:100]}...")