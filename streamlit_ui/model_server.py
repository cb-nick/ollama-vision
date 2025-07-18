from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")  # Loads once at server start

class EncodeRequest(BaseModel):
    type: str

@app.post("/encode")
def encode(req: EncodeRequest):
    embedding = model.encode([req.type]).tolist()
    return {"embeddings": embedding}

# Run with command: uvicorn model_server:app --host 0.0.0.0 --port 8000