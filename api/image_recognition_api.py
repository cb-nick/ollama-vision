from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.data_loader import DataLoader
from services.vision_service import VisionService
from services.semantic_search import SemanticSearchService
from config import AppConfig
from typing import Dict
import base64
import io
from PIL import Image

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load configuration
config = AppConfig()

# Initialize services
data_loader = DataLoader()
data = data_loader.load_all_data()
vision_service = VisionService(config)
semantic_search_service = SemanticSearchService(config, data)

# Cache data in memory
PC_TO_ITEM = data["PC_TO_ITEM"]
ALIAS_TO_PC = data["ALIAS_TO_PC"]
ALIASES = data["ALIASES"]
ALIAS_EMBEDDINGS = data["ALIAS_EMBEDDINGS"]

# Define request model
class ImageRequest(BaseModel):
    image_base64: str

# TODO: Ensure model server is running before calling this endpoint
@app.post("/analyze-image", response_model=Dict)
async def analyze_image(file: UploadFile = File(...)):
    """
    Endpoint to analyze an uploaded image and return recognition results.
    """
    try:
        # Read the uploaded image
        image_bytes = await file.read()
        
        # Validate the image
        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Analyze the image using VisionService
        result = vision_service.analyze_image(image_bytes)
        
        # Ensure result is a dict and not None
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Vision service did not return a dictionary result")
        
        # Enrich results with semantic search
        cb_type, product_code = semantic_search_service.find_closest_match(
            result.get("type", "unknown")
        )
        result["cb_type"] = cb_type
        result["product_code"] = product_code

        return {"success": True, "data": result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/analyze-image-base64", response_model=Dict)
async def analyze_image_base64(request: ImageRequest):
    """
    Endpoint to analyze an image provided as a base64 string.
    """
    try:
        # Decode the base64 image
        try:
            image_data = base64.b64decode(request.image_base64)
            Image.open(io.BytesIO(image_data)).verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image")

        # Analyze the image using VisionService
        result = vision_service.analyze_image(image_data)
        
        # Ensure result is a dict and not None
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Vision service did not return a dictionary result")

        # Capitalize the first letter of each string value in the result dictionary
        for key, value in result.items():
            if isinstance(value, str):
                result[key] = value.capitalize() if not value.lower().startswith('ip') else value

        # Enrich results with semantic search
        cb_type, product_code = semantic_search_service.find_closest_match(
            result.get("type", "unknown")
        )
               
        newItem = {
            "cb_type": cb_type,
            "product_code": product_code,
            "attributes": result
        }

        return {"success": True, "data": newItem}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "image_recognition_api:app",
        host="0.0.0.0",
        port=8505,
        reload=True,
        ssl_keyfile="./certs/api-selfsigned.key",
        ssl_certfile="./certs/api-selfsigned.crt"
    )