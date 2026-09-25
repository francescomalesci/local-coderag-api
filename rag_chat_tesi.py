import chromadb
import ollama
from chromadb.utils import embedding_functions

# 1. Setup del Database
CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="dati_tesi", embedding_function=embedding_model)

print("Inizializzazione Chatbot RAG completata. Digita 'esci' per chiudere.\n")

# 2. Il Loop della Chat
while True:
    # Aspetta la domanda dell'utente
    query_text = input("\nTu: ")
    
    # Condizione di uscita
    if query_text.lower() in ['esci', 'exit', 'quit']:
        print("Chiusura in corso...")
        break
        
    if not query_text.strip():
        continue

    # Ricerca nel database
    results = collection.query(query_texts=[query_text], n_results=1)
    
    # Controllo se ci sono risultati
    if not results['documents'][0]:
        print("IA: Non ho trovato informazioni pertinenti nei documenti.")
        continue
        
    contesto_estratto = results['documents'][0][0]
    
    prompt = f"""Sei un assistente tecnico. Usa ESCLUSIVAMENTE il contesto fornito per rispondere. 
    Se il contesto non è rilevante per la domanda, di' "Non ho informazioni su questo".
    
    CONTESTO: {contesto_estratto}
    
    DOMANDA: {query_text}
    
    RISPOSTA:"""
    
    # Chiamata a Ollama
    response = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}])
    
    # Stampa la risposta
    print(f"\nIA: {response['message']['content']}")