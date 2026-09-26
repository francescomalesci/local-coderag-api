from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import chromadb
import ollama
from chromadb.utils import embedding_functions
import logging
import time
import os
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Local CodeRAG API")

API_KEY = os.getenv("API_KEY") 
api_key_header = APIKeyHeader(name="X-API-Key")

def verifica_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="API Key non valida.")
    return api_key

CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="documenti_aziendali", embedding_function=embedding_model)

class ChatRequest(BaseModel):
    domanda: str

class ChatResponse(BaseModel):
    risposta: str
    fonte_utilizzata: str
    tempo_esecuzione_secondi: float

# ==========================================
# I NOSTRI TOOL REALI
# ==========================================

def cerca_appunti_tesi(query: str) -> str:
    """Cerca informazioni nel database vettoriale degli appunti di tesi (YOLO, SAM 2, binari, massicciata). Usa questo strumento se la domanda riguarda la tesi o i documenti aziendali."""
    logging.info(f"Esecuzione TOOL RAG: ricerca di '{query}'")
    results = collection.query(query_texts=[query], n_results=3)
    if not results['documents'][0]:
        return "Nessun dato trovato nel database della tesi."
    return "\n---\n".join(results['documents'][0])

def cerca_arxiv(query: str) -> str:
    """Cerca articoli scientifici su arXiv. Usa questo strumento SOLO se l'utente chiede esplicitamente ricerche su paper, articoli scientifici o letteratura accademica generale."""
    logging.info(f"Esecuzione TOOL arXiv: ricerca di '{query}'")
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=2'
    try:
        response = requests.get(url)
        root = ET.fromstring(response.text)
        risultati = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            titolo = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            risultati.append(f"- {titolo}")
        return "\n".join(risultati) if risultati else "Nessun articolo trovato su arXiv."
    except Exception as e:
        return f"Errore durante la ricerca su arXiv: {e}"

# Lista dei tool da passare a Ollama
tools_disponibili = {
    'cerca_appunti_tesi': cerca_appunti_tesi,
    'cerca_arxiv': cerca_arxiv
}

# ==========================================
# ENDPOINT 1: RAG PURO (Quello vecchio)
# ==========================================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, api_key: str = Depends(verifica_api_key)):
    start_time = time.time()
    
    results = collection.query(query_texts=[req.domanda], n_results=3)
    if not results['documents'][0]:
        return ChatResponse(risposta="Nessun dato trovato.", fonte_utilizzata="Nessuna", tempo_esecuzione_secondi=0.0)
        
    contesto_estratto = "\n---\n".join(results['documents'][0])
    fonti_uniche = ", ".join(list(set([meta['source'] for meta in results['metadatas'][0]])))
    
    prompt = f"Sei un assistente tecnico. Usa ESCLUSIVAMENTE il contesto fornito.\nCONTESTO:\n{contesto_estratto}\nDOMANDA: {req.domanda}\nRISPOSTA:"
    
    # RAG Usa il 7b per risparmiare tempo, è un task semplice
    response = ollama.chat(model='qwen2.5:7b', messages=[{'role': 'user', 'content': prompt}])
    
    execution_time = round(time.time() - start_time, 2)
    return ChatResponse(risposta=response['message']['content'], fonte_utilizzata=fonti_uniche, tempo_esecuzione_secondi=execution_time)


# ==========================================
# ENDPOINT 2: AGENTE INTELLIGENTE (Nuovo!)
# ==========================================
@app.post("/agent", response_model=ChatResponse)
async def agent(req: ChatRequest, api_key: str = Depends(verifica_api_key)):
    start_time = time.time()
    logging.info(f"Agente attivato per la query: '{req.domanda}'")
    
    messages = [{'role': 'user', 'content': req.domanda}]
    
    # 1. Chiediamo al 14b (che è più intelligente) di decidere cosa fare
    response = ollama.chat(
        model='qwen2.5:14b',
        messages=messages,
        tools=[cerca_appunti_tesi, cerca_arxiv]
    )
    
    tool_utilizzato = "Nessuno (Risposta Diretta)"
    
    # 2. Se l'Agente decide di usare un tool
    if response.get('message', {}).get('tool_calls'):
        messages.append(response['message'])
        
        for tool in response['message']['tool_calls']:
            nome_funzione = tool['function']['name']
            argomenti = tool['function']['arguments']
            tool_utilizzato = nome_funzione
            
            # Eseguiamo la funzione Python corrispondente
            funzione_da_chiamare = tools_disponibili[nome_funzione]
            risultato_tool = funzione_da_chiamare(**argomenti)
            
            messages.append({'role': 'tool', 'content': risultato_tool, 'name': nome_funzione})
            
            # 3. L'agente genera la risposta finale leggendo i dati del tool
            risposta_finale = ollama.chat(model='qwen2.5:14b', messages=messages)
            risposta_testo = risposta_finale['message']['content']
    else:
        # L'agente ha risposto direttamente (es. "Ciao come stai?")
        risposta_testo = response['message']['content']
        
    execution_time = round(time.time() - start_time, 2)
    
    return ChatResponse(
        risposta=risposta_testo,
        fonte_utilizzata=f"Agente tramite: {tool_utilizzato}",
        tempo_esecuzione_secondi=execution_time
    )