from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

app = FastAPI()

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Loads once at server start
    print(f"After loading - Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"After loading - Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
except Exception as e:
    print(f"Error loading model: {e}")

class EncodeRequest(BaseModel):
    type: str

@app.post("/encode")
def encode(req: EncodeRequest):
    embedding = model.encode([req.type]).tolist()
    return {"embeddings": embedding}

# Run with command: uvicorn model_server:app --host 0.0.0.0 --port 8000