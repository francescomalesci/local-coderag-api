from pydantic import BaseModel

class ChatRequest(BaseModel):
    domanda: str
    session_id: str = "utente_default"

class IngestRequest(BaseModel):
    testo: str
    fonte: str = "api_upload"