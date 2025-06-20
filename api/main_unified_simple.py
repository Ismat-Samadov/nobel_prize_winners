from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import os
import uuid
import asyncio
import logging
from datetime import datetime
import io
import base64
import random
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Active Learning NER - Unified App",
    description="Full-stack Active Learning for Named Entity Recognition (Demo Version)",
    version="1.0.0"
)

# Serve static files (React frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Pydantic models
class TextRequest(BaseModel):
    text: str
    model_id: Optional[str] = "default"

class PredictionResponse(BaseModel):
    tokens: List[str]
    labels: List[str]
    entities: List[Dict[str, Any]]

class TrainingRequest(BaseModel):
    model_name: str = "bert-base-uncased"
    num_labels: int = 9
    use_crf: bool = False
    strategy: str = "uncertainty"
    uncertainty_method: str = "entropy"
    num_rounds: int = 10
    samples_per_round: int = 100
    epochs_per_round: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    initial_labeled_ratio: float = 0.1

class ActiveLearningSession(BaseModel):
    session_id: str
    model_config: TrainingRequest
    status: str
    current_round: int
    total_rounds: int
    labeled_samples: int
    unlabeled_samples: int
    performance: Dict[str, float]
    created_at: datetime
    updated_at: datetime

class AnnotationRequest(BaseModel):
    session_id: str
    annotations: List[Dict[str, Any]]

# In-memory storage (use database in production)
active_sessions = {}

# Demo NER model (simplified for deployment)
class DemoNERModel:
    def __init__(self):
        # Simple rule-based NER for demo
        self.person_names = ['john', 'smith', 'mary', 'johnson', 'david', 'wilson', 'sarah', 'brown', 'michael', 'davis']
        self.organizations = ['google', 'microsoft', 'apple', 'facebook', 'amazon', 'tesla', 'netflix', 'stanford', 'mit', 'harvard']
        self.locations = ['california', 'new york', 'texas', 'florida', 'washington', 'boston', 'seattle', 'chicago', 'houston']
        
    def predict(self, tokens):
        labels = []
        entities = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            
            if token_lower in self.person_names:
                labels.append('B-PER')
                entities.append({
                    "text": token,
                    "label": "PER",
                    "start": i,
                    "end": i + 1,
                    "confidence": 0.95
                })
            elif token_lower in self.organizations:
                labels.append('B-ORG')
                entities.append({
                    "text": token,
                    "label": "ORG", 
                    "start": i,
                    "end": i + 1,
                    "confidence": 0.92
                })
            elif token_lower in self.locations:
                labels.append('B-LOC')
                entities.append({
                    "text": token,
                    "label": "LOC",
                    "start": i,
                    "end": i + 1,
                    "confidence": 0.88
                })
            elif token.istitle() and len(token) > 2:
                # Likely proper noun
                labels.append('B-MISC')
                entities.append({
                    "text": token,
                    "label": "MISC",
                    "start": i,
                    "end": i + 1,
                    "confidence": 0.75
                })
            else:
                labels.append('O')
        
        return labels, entities

# Initialize demo model
demo_model = DemoNERModel()

# Serve React app for all non-API routes
@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the React frontend."""
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        return {
            "message": "Active Learning NER - Unified Application (Demo)",
            "version": "1.0.0",
            "status": "Backend only (frontend not built)",
            "api_docs": "/docs",
            "note": "This is a demo version with simplified NER model"
        }

# Catch-all route for React Router (SPA routing)
@app.get("/{path:path}", include_in_schema=False)
async def serve_frontend_routes(path: str):
    """Serve React app for all frontend routes."""
    # Check if it's an API route
    if path.startswith("api/") or path.startswith("docs") or path.startswith("openapi"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Check if it's a static file
    file_path = f"static/{path}"
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Default to serving index.html for SPA routing
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    else:
        raise HTTPException(status_code=404, detail="Frontend not available")

# API Routes with /api prefix
@app.get("/api/")
async def api_root():
    """API information endpoint."""
    return {
        "message": "Active Learning NER API (Demo Version)",
        "version": "1.0.0",
        "note": "This demo uses a simplified rule-based NER model",
        "endpoints": {
            "predict": "/api/predict",
            "upload": "/api/upload", 
            "train": "/api/train",
            "sessions": "/api/sessions",
            "annotate": "/api/annotate"
        }
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_ner(request: TextRequest):
    """Predict named entities in text using demo model."""
    try:
        # Tokenize input
        tokens = request.text.split()
        
        # Make prediction using demo model
        labels, entities = demo_model.predict(tokens)
        
        return PredictionResponse(
            tokens=tokens,
            labels=labels,
            entities=entities
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload training data file."""
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_path = f"uploads/{file_id}_{file.filename}"
        
        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Basic validation
        return {
            "file_id": file_id,
            "filename": file.filename,
            "status": "success",
            "stats": {
                "num_sentences": random.randint(800, 1200),
                "num_tokens": random.randint(12000, 18000),
                "label_distribution": {
                    "O": random.randint(8000, 12000),
                    "B-PER": random.randint(500, 800),
                    "I-PER": random.randint(200, 400),
                    "B-ORG": random.randint(300, 600),
                    "I-ORG": random.randint(100, 300),
                    "B-LOC": random.randint(200, 500),
                    "I-LOC": random.randint(50, 200)
                }
            }
        }
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start active learning training session."""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Create session
        session = ActiveLearningSession(
            session_id=session_id,
            model_config=request,
            status="initializing",
            current_round=0,
            total_rounds=request.num_rounds,
            labeled_samples=int(1000 * request.initial_labeled_ratio),
            unlabeled_samples=int(1000 * (1 - request.initial_labeled_ratio)),
            performance={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        active_sessions[session_id] = session
        
        # Start training simulation in background
        background_tasks.add_task(simulate_active_learning, session_id)
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Active learning session started (demo mode)"
        }
        
    except Exception as e:
        logger.error(f"Training start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def simulate_active_learning(session_id: str):
    """Simulate active learning training."""
    try:
        session = active_sessions[session_id]
        
        # Wait a bit to simulate initialization
        await asyncio.sleep(2)
        
        session.status = "active_learning"
        session.updated_at = datetime.now()
        
        logger.info(f"Session {session_id} ready for annotation")
        
    except Exception as e:
        logger.error(f"Active learning simulation error for session {session_id}: {str(e)}")
        if session_id in active_sessions:
            active_sessions[session_id].status = "error"

@app.get("/api/sessions")
async def get_sessions():
    """Get all active learning sessions."""
    return {
        "sessions": [
            {
                "session_id": session.session_id,
                "status": session.status,
                "current_round": session.current_round,
                "total_rounds": session.total_rounds,
                "labeled_samples": session.labeled_samples,
                "unlabeled_samples": session.unlabeled_samples,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }
            for session in active_sessions.values()
        ]
    }

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific session details."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Generate demo samples for annotation
    sample_texts = [
        "John Smith works at Google in California",
        "Microsoft announced a partnership with OpenAI",
        "Sarah Johnson from MIT will present research", 
        "The conference will be held in Boston next week",
        "Apple Inc was founded by Steve Jobs in Cupertino"
    ]
    
    samples_for_annotation = []
    if session.status == "active_learning":
        for i, text in enumerate(sample_texts[:3]):  # Show 3 samples
            tokens = text.split()
            samples_for_annotation.append({
                "id": i,
                "tokens": tokens,
                "text": text
            })
    
    return {
        "session": {
            "session_id": session.session_id,
            "status": session.status,
            "current_round": session.current_round,
            "total_rounds": session.total_rounds,
            "labeled_samples": session.labeled_samples,
            "unlabeled_samples": session.unlabeled_samples,
            "performance": session.performance,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        },
        "samples_for_annotation": samples_for_annotation
    }

@app.post("/api/annotate")
async def submit_annotations(request: AnnotationRequest):
    """Submit annotations for active learning."""
    try:
        session_id = request.session_id
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        
        if session.status != "active_learning":
            raise HTTPException(status_code=400, detail="Session not in active learning phase")
        
        # Process annotations (demo simulation)
        session.current_round += 1
        session.labeled_samples += len(request.annotations)
        session.unlabeled_samples -= len(request.annotations)
        session.updated_at = datetime.now()
        
        # Simulate performance improvement
        base_f1 = 0.6
        improvement = session.current_round * 0.05
        session.performance = {
            "f1": min(0.95, base_f1 + improvement),
            "precision": min(0.93, base_f1 + improvement - 0.02),
            "recall": min(0.97, base_f1 + improvement + 0.02)
        }
        
        # Check if training is complete
        if session.current_round >= session.total_rounds:
            session.status = "completed"
        
        return {
            "status": "success",
            "message": f"Processed {len(request.annotations)} annotations (demo)",
            "session_status": session.status,
            "current_round": session.current_round,
            "performance": session.performance
        }
        
    except Exception as e:
        logger.error(f"Annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    """Get training results and visualizations."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Generate demo learning curve data
    rounds = list(range(1, session.current_round + 1))
    f1_scores = [0.6 + (r * 0.05) for r in rounds]
    
    return {
        "session_id": session_id,
        "status": session.status,
        "learning_curve": {
            "rounds": rounds,
            "f1_scores": f1_scores,
            "precision_scores": [f - 0.02 for f in f1_scores],
            "recall_scores": [f + 0.02 for f in f1_scores]
        },
        "final_performance": session.performance,
        "sample_efficiency": {
            "total_samples": session.labeled_samples + session.unlabeled_samples,
            "labeled_samples": session.labeled_samples,
            "efficiency_ratio": session.labeled_samples / (session.labeled_samples + session.unlabeled_samples) if (session.labeled_samples + session.unlabeled_samples) > 0 else 0
        }
    }

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a training session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "frontend_available": os.path.exists("static/index.html"),
        "mode": "demo",
        "note": "Running with simplified NER model for demo purposes"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)