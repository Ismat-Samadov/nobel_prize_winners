from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import os
import uuid
import asyncio
import logging
from datetime import datetime
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Active Learning NER", description="Active Learning for Named Entity Recognition")

# Get the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Current script directory: {current_dir}")
logger.info(f"Working directory: {os.getcwd()}")

# Create directories relative to script location
static_dir = os.path.join(current_dir, "static")
templates_dir = os.path.join(current_dir, "templates")
uploads_dir = os.path.join(current_dir, "uploads")

logger.info(f"Templates directory: {templates_dir}")
logger.info(f"Templates exist: {os.path.exists(templates_dir)}")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(uploads_dir, exist_ok=True)

# Check if templates exist
if os.path.exists(templates_dir):
    template_files = os.listdir(templates_dir)
    logger.info(f"Template files found: {template_files}")
else:
    logger.error(f"Templates directory not found: {templates_dir}")

# Mount static files (only if directory exists)
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory=templates_dir)

# Pydantic models
class TextRequest(BaseModel):
    text: str

class TrainingRequest(BaseModel):
    strategy: str = "uncertainty"
    num_rounds: int = 10
    samples_per_round: int = 100

class AnnotationRequest(BaseModel):
    session_id: str
    annotations: List[Dict[str, Any]]

# In-memory storage
active_sessions = {}

# Demo NER model
class DemoNERModel:
    def __init__(self):
        self.person_names = ['john', 'smith', 'mary', 'johnson', 'david', 'wilson', 'sarah', 'brown', 'michael', 'davis', 'jennifer', 'garcia', 'robert', 'miller', 'lisa', 'anderson']
        self.organizations = ['google', 'microsoft', 'apple', 'facebook', 'amazon', 'tesla', 'netflix', 'stanford', 'mit', 'harvard', 'openai', 'deepmind', 'uber', 'airbnb']
        self.locations = ['california', 'new york', 'texas', 'florida', 'washington', 'boston', 'seattle', 'chicago', 'houston', 'philadelphia', 'phoenix', 'san antonio', 'san diego', 'dallas', 'austin']
        
    def predict(self, text):
        tokens = text.split()
        labels = []
        entities = []
        
        for i, token in enumerate(tokens):
            token_lower = token.lower().strip('.,!?;')
            
            if token_lower in self.person_names:
                labels.append('B-PER')
                entities.append({
                    "text": token,
                    "label": "PERSON",
                    "start": i,
                    "end": i + 1,
                    "confidence": round(random.uniform(0.85, 0.98), 2)
                })
            elif token_lower in self.organizations:
                labels.append('B-ORG')
                entities.append({
                    "text": token,
                    "label": "ORGANIZATION", 
                    "start": i,
                    "end": i + 1,
                    "confidence": round(random.uniform(0.80, 0.95), 2)
                })
            elif token_lower in self.locations:
                labels.append('B-LOC')
                entities.append({
                    "text": token,
                    "label": "LOCATION",
                    "start": i,
                    "end": i + 1,
                    "confidence": round(random.uniform(0.75, 0.92), 2)
                })
            elif token.istitle() and len(token) > 2 and token_lower not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use']:
                labels.append('B-MISC')
                entities.append({
                    "text": token,
                    "label": "MISC",
                    "start": i,
                    "end": i + 1,
                    "confidence": round(random.uniform(0.60, 0.85), 2)
                })
            else:
                labels.append('O')
        
        return tokens, labels, entities

# Initialize demo model
ner_model = DemoNERModel()

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_ner(request: Request, text: str = Form(...)):
    try:
        tokens, labels, entities = ner_model.predict(text)
        
        # Create highlighted text
        highlighted_tokens = []
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label != 'O':
                entity_info = next((e for e in entities if e['start'] == i), None)
                if entity_info:
                    highlighted_tokens.append({
                        'text': token,
                        'label': entity_info['label'],
                        'confidence': entity_info['confidence']
                    })
                else:
                    highlighted_tokens.append({'text': token, 'label': None})
            else:
                highlighted_tokens.append({'text': token, 'label': None})
        
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "text": text,
            "tokens": highlighted_tokens,
            "entities": entities,
            "prediction_made": True
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return templates.TemplateResponse("predict.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse("train.html", {"request": request})

@app.post("/train", response_class=HTMLResponse)
async def start_training(request: Request, strategy: str = Form(...), num_rounds: int = Form(...), samples_per_round: int = Form(...), background_tasks: BackgroundTasks = None):
    try:
        session_id = str(uuid.uuid4())
        
        session = {
            "session_id": session_id,
            "strategy": strategy,
            "status": "initializing",
            "current_round": 0,
            "total_rounds": num_rounds,
            "samples_per_round": samples_per_round,
            "labeled_samples": random.randint(80, 120),
            "unlabeled_samples": random.randint(800, 1200),
            "performance": {},
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        active_sessions[session_id] = session
        
        # Simulate initialization
        await asyncio.sleep(1)
        session["status"] = "active_learning"
        session["updated_at"] = datetime.now()
        
        return RedirectResponse(url=f"/sessions/{session_id}", status_code=303)
        
    except Exception as e:
        logger.error(f"Training start error: {str(e)}")
        return templates.TemplateResponse("train.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request):
    sessions = list(active_sessions.values())
    return templates.TemplateResponse("sessions.html", {
        "request": request,
        "sessions": sessions
    })

@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Generate demo samples for annotation
    sample_texts = [
        "John Smith works at Google in California",
        "Microsoft announced a partnership with OpenAI",
        "Sarah Johnson from MIT will present research", 
        "The conference will be held in Boston next week",
        "Apple Inc was founded by Steve Jobs in Cupertino",
        "Tesla is developing new technology in Austin",
        "Amazon Web Services launched in Seattle",
        "Facebook changed its name to Meta",
        "Netflix streams content worldwide",
        "Harvard University is located in Massachusetts"
    ]
    
    samples_for_annotation = []
    if session["status"] == "active_learning":
        for i, text in enumerate(random.sample(sample_texts, 3)):
            tokens = text.split()
            samples_for_annotation.append({
                "id": i,
                "tokens": tokens,
                "text": text
            })
    
    return templates.TemplateResponse("session_detail.html", {
        "request": request,
        "session": session,
        "samples": samples_for_annotation
    })

@app.post("/annotate/{session_id}", response_class=HTMLResponse)
async def submit_annotations(request: Request, session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    form_data = await request.form()
    
    # Process annotations (simplified)
    num_annotations = len([key for key in form_data.keys() if key.startswith('annotation_')])
    
    session["current_round"] += 1
    session["labeled_samples"] += num_annotations
    session["unlabeled_samples"] -= num_annotations
    session["updated_at"] = datetime.now()
    
    # Simulate performance improvement
    base_f1 = 0.6
    improvement = session["current_round"] * 0.05
    session["performance"] = {
        "f1": min(0.95, base_f1 + improvement),
        "precision": min(0.93, base_f1 + improvement - 0.02),
        "recall": min(0.97, base_f1 + improvement + 0.02)
    }
    
    if session["current_round"] >= session["total_rounds"]:
        session["status"] = "completed"
    
    return RedirectResponse(url=f"/results/{session_id}", status_code=303)

@app.get("/results/{session_id}", response_class=HTMLResponse)
async def results_page(request: Request, session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Generate chart data
    rounds = list(range(1, session["current_round"] + 1))
    chart_data = []
    for r in rounds:
        f1 = 0.6 + (r * 0.05)
        chart_data.append({
            "round": r,
            "f1": round(f1, 3),
            "precision": round(f1 - 0.02, 3),
            "recall": round(f1 + 0.02, 3)
        })
    
    return templates.TemplateResponse("results.html", {
        "request": request,
        "session": session,
        "chart_data": chart_data
    })

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "mode": "demo"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)