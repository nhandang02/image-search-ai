"""
CLIP Image Search Service
FastAPI service for encoding images and text using OpenAI CLIP model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import logging
import base64
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CLIP Image Search Service",
    description="Encode images and text using OpenAI CLIP model for visual search",
    version="1.0.0"
)

# CORS configuration for NestJS backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your NestJS domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model: Optional[CLIPModel] = None
processor: Optional[CLIPProcessor] = None
device: str = "cpu"

# Request/Response models
class TextEncodeRequest(BaseModel):
    text: str

class BatchTextEncodeRequest(BaseModel):
    texts: List[str]

class ImageEncodeRequest(BaseModel):
    image_base64: str

class BatchImageEncodeRequest(BaseModel):
    images_base64: List[str]

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str

@app.on_event("startup")
async def load_model():
    """Load CLIP model on startup"""
    global model, processor, device
    
    try:
        logger.info("Loading CLIP model...")
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load CLIP model and processor
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # Move model to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"CLIP model loaded successfully: {model_name}")
        logger.info(f"Embedding dimensions: 512")
        
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {str(e)}")
        raise

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with service info"""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "device": device,
        "model_name": "openai/clip-vit-base-patch32"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": device,
        "model_name": "openai/clip-vit-base-patch32"
    }

@app.post("/encode/text", response_model=EmbeddingResponse)
async def encode_text(request: TextEncodeRequest):
    """Encode a single text into CLIP embedding"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process text
        inputs = processor(text=[request.text], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize embedding
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embedding = text_features.cpu().numpy().flatten().tolist()
        
        return {
            "embedding": embedding,
            "dimensions": len(embedding)
        }
        
    except Exception as e:
        logger.error(f"Error encoding text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encoding text: {str(e)}")

@app.post("/encode/image", response_model=EmbeddingResponse)
async def encode_image(request: ImageEncodeRequest):
    """Encode a single image (base64) into CLIP embedding"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embedding = image_features.cpu().numpy().flatten().tolist()
        
        return {
            "embedding": embedding,
            "dimensions": len(embedding)
        }
        
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

@app.post("/encode/image/upload", response_model=EmbeddingResponse)
async def encode_image_upload(file: UploadFile = File(...)):
    """Encode an uploaded image file into CLIP embedding"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embedding = image_features.cpu().numpy().flatten().tolist()
        
        return {
            "embedding": embedding,
            "dimensions": len(embedding)
        }
        
    except Exception as e:
        logger.error(f"Error encoding uploaded image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encoding image: {str(e)}")

@app.post("/encode/batch-texts", response_model=BatchEmbeddingResponse)
async def encode_batch_texts(request: BatchTextEncodeRequest):
    """Encode multiple texts into CLIP embeddings"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process texts
        inputs = processor(text=request.texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            # Normalize embeddings
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embeddings = text_features.cpu().numpy().tolist()
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        }
        
    except Exception as e:
        logger.error(f"Error encoding batch texts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encoding texts: {str(e)}")

@app.post("/encode/batch-images", response_model=BatchEmbeddingResponse)
async def encode_batch_images(request: BatchImageEncodeRequest):
    """Encode multiple images (base64) into CLIP embeddings"""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode all images
        images = []
        for img_base64 in request.images_base64:
            image_data = base64.b64decode(img_base64)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            images.append(image)
        
        # Process images
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convert to list
        embeddings = image_features.cpu().numpy().tolist()
        
        return {
            "embeddings": embeddings,
            "dimensions": len(embeddings[0]) if embeddings else 0,
            "count": len(embeddings)
        }
        
    except Exception as e:
        logger.error(f"Error encoding batch images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encoding images: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
