import os
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# 1. Setup Database
CHROMA_DATA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Usiamo una collezione pulita e definitiva
collection = client.get_or_create_collection(name="documenti_aziendali", embedding_function=embedding_model)

DOCS_DIR = "./documenti"

# 2. Funzione di Chunking (divisione intelligente)
def chunk_text(text, chunk_size=500, overlap=50):
    """Divide il testo in blocchi con una leggera sovrapposizione per non perdere il contesto."""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks

# 3. Estrazione testo da PDF
def process_pdf(file_path):
    testo = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        testo_pagina = page.extract_text()
        if testo_pagina:
            testo += testo_pagina + "\n"
    return testo

# 4. Pipeline Principale
if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)
    print(f"Cartella {DOCS_DIR} creata. Inserisci i PDF o TXT e riavvia lo script.")
    exit()

print("Inizio scansione documenti...")

file_processati = 0
for filename in os.listdir(DOCS_DIR):
    file_path = os.path.join(DOCS_DIR, filename)
    testo_completo = ""
    
    if filename.endswith(".pdf"):
        testo_completo = process_pdf(file_path)
    elif filename.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            testo_completo = f.read()
    else:
        print(f"Formato non supportato, salto: {filename}")
        continue
        
    if not testo_completo.strip():
        continue
        
    # Applica il chunking
    chunks = chunk_text(testo_completo)
    
    docs = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        docs.append(chunk)
        metadatas.append({"source": filename, "chunk_index": i})
        ids.append(f"{filename}_chunk_{i}")
        
    # Salvataggio in ChromaDB
    if docs:
        collection.add(documents=docs, metadatas=metadatas, ids=ids)
        print(f"Innestato: {filename} ({len(chunks)} blocchi vettorializzati).")
        file_processati += 1

print(f"\nPipeline completata: {file_processati} file aggiunti al Vector DB.")