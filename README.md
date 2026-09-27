# Local CodeRAG

Sistema RAG (Retrieval-Augmented Generation) e AI Agent completamente locale, costruito su appunti di tesi personali (segmentazione binari ferroviari con YOLO/SAM 2). Espone due modalità di interazione — RAG puro e Agente con tool esterni — tramite API REST, con streaming delle risposte e memoria conversazionale multi-turno.

Progetto sviluppato come portfolio tecnico per candidature Junior AI/LLM Engineer, con focus su architetture RAG, Vector Database, function calling e sistemi conversazionali.

## Caratteristiche

- **RAG locale** su documenti personali (PDF, TXT), con chunking, embedding ed indicizzazione vettoriale
- **Zero dipendenze cloud**: nessuna API key esterna a pagamento, LLM eseguiti interamente in locale via Ollama
- **AI Agent con function calling**: instrada autonomamente le richieste tra ricerca nei documenti, ricerca su arXiv e salvataggio note
- **API REST modulare** (FastAPI) con autenticazione via API key, streaming delle risposte (NDJSON) e memoria di sessione
- **Frontend Streamlit** con selezione tra modalità RAG semplice e Agente

## Architettura

```mermaid
flowchart TD
    A[File utente: .pdf .txt] --> B[ingest.py: Chunking]
    B --> C[database.py: Embedding - sentence-transformers]
    C --> D[(ChromaDB - Vector Store locale)]

    U[Utente] --> F[app_frontend.py - Streamlit]
    F -->|POST /chat o /agent| API[main.py - FastAPI]

    API -->|modalita RAG| R[Retrieval semantico su ChromaDB]
    R --> D
    R --> G7[Ollama - qwen2.5:7b]
    G7 -->|stream NDJSON| F

    API -->|modalita Agent| AG[Ollama - qwen2.5:14b + Tool Calling]
    AG -->|decide quale tool usare| T1[tools.py: search_thesis_notes]
    AG --> T2[tools.py: search_arxiv]
    AG --> T3[tools.py: save_note]
    T1 --> D
    AG -->|sintesi finale, stream NDJSON| F
```

## Stack tecnologico

| Componente         | Tecnologia                                  |
|---------------------|----------------------------------------------|
| Linguaggio          | Python 3.10+                                 |
| Vector Database     | ChromaDB (persistente, locale)               |
| Embedding           | sentence-transformers (`all-MiniLM-L6-v2`)   |
| LLM locale          | Ollama — `qwen2.5:7b` (RAG), `qwen2.5:14b` (Agent) |
| Backend API         | FastAPI + Pydantic                           |
| Frontend            | Streamlit                                    |
| Estrazione PDF      | pypdf                                        |
| Config              | python-dotenv                                |

## Requisiti hardware

Testato su GPU NVIDIA RTX 3080 (10/12 GB VRAM) + 64 GB RAM. `qwen2.5:7b` gira comodamente in VRAM; `qwen2.5:14b` quantizzato (Q4) è consigliato per GPU con VRAM limitata.

## Setup

### 1. Ambiente Python

```bash
python -m venv venv
source venv/bin/activate  # su Windows: venv\Scripts\activate
pip install fastapi uvicorn chromadb sentence-transformers ollama pypdf python-dotenv streamlit requests
```

### 2. Ollama e modelli

Installare [Ollama](https://ollama.com), poi scaricare i modelli:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:14b
```

### 3. Variabili d'ambiente

Creare un file `.env` nella root del progetto:

```
API_KEY=scegli-una-chiave-segreta
```

### 4. Documenti da ingerire

Creare la cartella `./documents` e inserire i file `.pdf` o `.txt` da indicizzare.

## Utilizzo

**1. Ingestion iniziale** (popola il Vector Database da file):

```bash
python ingest.py
```

**2. Avviare il backend API:**

```bash
uvicorn main:app --reload
```

Documentazione interattiva disponibile su `http://127.0.0.1:8000/docs`.

**3. Avviare il frontend:**

```bash
streamlit run app_frontend.py
```

## Endpoint API

Tutti gli endpoint richiedono l'header `X-API-Key`.

| Metodo | Endpoint  | Descrizione                                                        |
|--------|-----------|---------------------------------------------------------------------|
| POST   | `/chat`   | RAG puro: retrieval semantico + generazione con `qwen2.5:7b`, risposta in streaming NDJSON |
| POST   | `/agent`  | Agente con tool calling (`qwen2.5:14b`), memoria di sessione, risposta in streaming NDJSON |
| POST   | `/ingest` | Aggiunge un documento testuale al Vector Database via API (con chunking automatico) |

**Esempio richiesta `/agent`:**

```json
{
  "question": "Come ho gestito i falsi positivi sulla ghiaia nella segmentazione?",
  "session_id": "demo_session_1"
}
```

**Formato risposta (streaming NDJSON)**, una riga JSON per evento:

```
{"type": "token", "content": "Per gestire i falsi..."}
{"type": "token", "content": " positivi sulla ghiaia..."}
{"type": "done", "source_used": "Agent via: search_thesis_notes", "execution_time_seconds": 3.42}
```

## Esempi di query

- *RAG semplice*: "Riassumi il meccanismo dei pesi spaziali usato in fase di training"
- *Agente → tool RAG*: "Cosa ho scritto sulla segmentazione con SAM 2?"
- *Agente → tool arXiv*: "Cerca paper recenti sulla segmentazione semantica di scene ferroviarie"
- *Agente → tool azione*: "Salva un riassunto di questa conversazione in un file"

## Struttura del progetto

```
.
├── main.py               # FastAPI app: endpoint /chat, /agent, /ingest, streaming helpers
├── database.py            # Setup ChromaDB, collection persistente, embedding model
├── models.py                # Pydantic models (ChatRequest, IngestRequest)
├── tools.py                   # Tool dell'agente: search_thesis_notes, search_arxiv, save_note
├── ingest.py                 # Script di ingestion da file (PDF/TXT) con chunking
├── app_frontend.py         # Interfaccia Streamlit
├── documents/                # Cartella sorgente per i file da ingerire
├── chroma_db/                 # Vector store persistente (generato automaticamente)
└── .env                        # Variabili d'ambiente (non versionato)
```

## Limitazioni note e sviluppi futuri

- La memoria conversazionale è mantenuta in RAM (`dict` Python) senza scadenza: adatto per demo, andrebbe sostituito con Redis o un database con TTL in un contesto di produzione
- Nessun test automatico presente al momento
- Il routing dell'agente su modelli locali di piccole dimensioni può occasionalmente essere impreciso nella scelta del tool; modelli più grandi (14B+) migliorano l'affidabilità a costo di maggiore latenza
- Possibili estensioni: retrieval ibrido (BM25 + semantico), valutazione quantitativa del retrieval (precision/recall), supporto opzionale a provider LLM cloud dietro la stessa interfaccia
