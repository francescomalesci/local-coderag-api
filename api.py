from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import chromadb
import ollama
from chromadb.utils import embedding_functions
import logging
import time
import os
from dotenv import load_dotenv

# Carica le variabili dal file .env
load_dotenv()

# --- SETUP LOGGING ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Local CodeRAG API")

# --- SETUP AUTENTICAZIONE ---
# Prende la chiave in modo sicuro dall'ambiente, non è più scritta nel codice!
API_KEY = os.getenv("API_KEY") 
api_key_header = APIKeyHeader(name="X-API-Key")

def verifica_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY or api_key != API_KEY:
        logging.warning("Tentativo di accesso negato: API Key errata o mancante.")
        raise HTTPException(status_code=403, detail="Accesso non autorizzato. API Key non valida.")
    return api_key

# --- SETUP DATABASE ---
CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="documenti_aziendali", embedding_function=embedding_model)

# --- MODELLI PYDANTIC ---
class ChatRequest(BaseModel):
    domanda: str

class ChatResponse(BaseModel):
    risposta: str
    fonte_utilizzata: str
    tempo_esecuzione_secondi: float

# --- ENDPOINT PROTETTO ---
# Nota l'aggiunta di "Depends(verifica_api_key)" che fa da scudo all'endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, api_key: str = Depends(verifica_api_key)):
    start_time = time.time()
    logging.info(f"Ricevuta query: '{req.domanda}'")
    
    # 1. Retrieval: Alziamo n_results a 3 per catturare più contesto
    results = collection.query(query_texts=[req.domanda], n_results=3)
    
    if not results['documents'][0]:
        logging.info("Nessun documento trovato nel database.")
        return ChatResponse(risposta="Nessun dato trovato.", fonte_utilizzata="Nessuna", tempo_esecuzione_secondi=0.0)
        
    # 2. Concatenazione dei chunk
    # Uniamo i 3 blocchi trovati separandoli con una linea
    contesto_estratto = "\n---\n".join(results['documents'][0])
    
    # Estraiamo i nomi dei file originali e rimuoviamo i duplicati (es. se 2 chunk vengono dallo stesso PDF)
    fonti_estratte = [meta['source'] for meta in results['metadatas'][0]]
    fonti_uniche = ", ".join(list(set(fonti_estratte)))
    
    prompt = f"""Sei un assistente tecnico. Usa ESCLUSIVAMENTE il contesto fornito.
    CONTESTO: 
    {contesto_estratto}
    
    DOMANDA: {req.domanda}
    RISPOSTA:"""
    
    # 3. Generazione
    response = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}])
    
    execution_time = round(time.time() - start_time, 2)
    logging.info(f"Risposta generata in {execution_time}s basandosi su: {fonti_uniche}")
    
    return ChatResponse(
        risposta=response['message']['content'],
        fonte_utilizzata=fonti_uniche,
        tempo_esecuzione_secondi=execution_time
    )