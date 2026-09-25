import chromadb
import ollama
from chromadb.utils import embedding_functions

# 1. Inizializzazione Database
CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection(name="my_documents", embedding_function=embedding_model)

def ask_rag(query_text):
    print(f"Ricerca nel database per: '{query_text}'...")
    
    # 2. Retrieval (Estrazione dei 2 blocchi più pertinenti)
    results = collection.query(query_texts=[query_text], n_results=2)
    
    # Uniamo i chunk trovati in un unico blocco di testo continuo
    contesto_estratto = "\n---\n".join(results['documents'][0])
    
    # 3. Prompt Augmentation (Iniezione del contesto)
    # Questo è il trucco del RAG: vietiamo al modello di usare le sue conoscenze pregresse.
    prompt = f"""Usa ESCLUSIVAMENTE il seguente contesto per rispondere alla domanda. 
    Se il contesto non contiene le informazioni necessarie, rispondi esplicitamente "Non ho abbastanza informazioni nei miei documenti".
    Non inventare nulla.
    
    CONTESTO:
    {contesto_estratto}
    
    DOMANDA: {query_text}
    
    RISPOSTA:"""
    
    print("Generazione della risposta tramite Ollama in corso...")
    
    # 4. Generation (Chiamata al modello locale)
    # Assicurati di usare il nome del modello che hai scaricato in precedenza (es. qwen2.5:7b o llama3.1:8b)
    response = ollama.chat(model='qwen2.5:7b', messages=[
        {
            'role': 'user',
            'content': prompt
        }
    ])
    
    return response['message']['content']

if __name__ == "__main__":
    domanda_utente = "Come funziona il sistema di bounding box?"
    risposta_finale = ask_rag(domanda_utente)
    
    print("\n--- RISPOSTA DELL'AGENTE ---")
    print(risposta_finale)