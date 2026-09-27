from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default_user"

class IngestRequest(BaseModel):
    text: str
    source: str = "api_upload"