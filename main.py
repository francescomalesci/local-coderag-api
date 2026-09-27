import time
import json
import logging
import os
import ollama
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from database import collection
from models import ChatRequest, IngestRequest
from tools import search_thesis_notes, search_arxiv, save_note, available_tools

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(title="Local CodeRAG API")

API_KEY = os.getenv("API_KEY") 
api_key_header = APIKeyHeader(name="X-API-Key")

session_memory = {}

def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY or api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key.")
    return api_key

# ==========================================
# HELPERS
# ==========================================
def stream_ndjson(ollama_stream, source: str, start_time: float, messages: list = None):
    full_response = ""
    try:
        for chunk in ollama_stream:
            piece = chunk['message']['content']
            if piece:
                full_response += piece
                yield json.dumps({"type": "token", "content": piece}) + "\n"
        
        if messages is not None:
            messages.append({'role': 'assistant', 'content': full_response})
        
        elapsed_time = round(time.time() - start_time, 2)
        yield json.dumps({"type": "done", "source_used": source, "execution_time_seconds": elapsed_time}) + "\n"
    except Exception as e:
        logging.error(f"Streaming error: {e}")
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"

def safe_stream_generator(model: str, messages: list, source: str, start_time: float):
    try:
        ollama_stream = ollama.chat(model=model, messages=messages, stream=True)
        return stream_ndjson(ollama_stream, source, start_time, messages=messages)
    except Exception as e:
        logging.error(f"Ollama connection error in stream: {e}")
        def error_gen():
            yield json.dumps({"type": "error", "content": f"LLM service unavailable (Stream). Details: {e}"}) + "\n"
        return error_gen()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ==========================================
# ENDPOINTS
# ==========================================
@app.post("/chat")
async def chat_stream(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    results = collection.query(query_texts=[req.question], n_results=3)
    
    if not results['documents'][0]:
        def empty():
            yield json.dumps({"type": "done", "source_used": "None", "execution_time_seconds": 0.0}) + "\n"
        return StreamingResponse(empty(), media_type="application/x-ndjson")
    
    extracted_context = "\n---\n".join(results['documents'][0])
    unique_sources = ", ".join(set(meta['source'] for meta in results['metadatas'][0]))
    prompt = f"You are a technical assistant. Use EXCLUSIVELY the provided context.\nCONTEXT:\n{extracted_context}\nQUESTION: {req.question}\nANSWER:"
    
    messages = [{'role': 'user', 'content': prompt}]
    
    return StreamingResponse(
        safe_stream_generator('qwen2.5:7b', messages, unique_sources, start_time), 
        media_type="application/x-ndjson"
    )

@app.post("/agent")
async def agent_stream(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    start_time = time.time()
    session = req.session_id
    messages = session_memory.setdefault(session, [])
    messages.append({'role': 'user', 'content': req.question})
    
    try:
        response = ollama.chat(model='qwen2.5:14b', messages=messages, tools=[search_thesis_notes, search_arxiv, save_note])
    except Exception as e:
        logging.error(f"Ollama connection error: {e}")
        def error_llm():
            yield json.dumps({"type": "error", "content": f"LLM service unavailable (Routing). Details: {e}"}) + "\n"
        return StreamingResponse(error_llm(), media_type="application/x-ndjson")
        
    used_tool = "None (Direct Answer)"
    
    if response.get('message', {}).get('tool_calls'):
        messages.append(response['message'])
        for tool in response['message']['tool_calls']:
            function_name = tool['function']['name']
            arguments = tool['function']['arguments']
            used_tool = function_name
            function = available_tools.get(function_name)
            tool_result = function(**arguments) if function else f"Error: tool '{function_name}' not found."
            messages.append({'role': 'tool', 'content': tool_result, 'name': function_name})
        
        return StreamingResponse(
            safe_stream_generator('qwen2.5:14b', messages, f"Agent via: {used_tool}", start_time),
            media_type="application/x-ndjson"
        )
    else:
        direct_response = response['message']['content']
        messages.append({'role': 'assistant', 'content': direct_response})
        def direct():
            yield json.dumps({"type": "token", "content": direct_response}) + "\n"
            yield json.dumps({"type": "done", "source_used": used_tool, "execution_time_seconds": round(time.time()-start_time,2)}) + "\n"
        return StreamingResponse(direct(), media_type="application/x-ndjson")

@app.post("/ingest")
async def api_ingest(req: IngestRequest, api_key: str = Depends(verify_api_key)):
    try:
        chunks = chunk_text(req.text)
        
        base_id = f"doc_{int(time.time())}"
        ids = [f"{base_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": req.source} for _ in chunks]
        
        collection.upsert(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        return {
            "status": "success", 
            "message": f"Text successfully ingested. Created {len(chunks)} chunks.", 
            "source": req.source
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")