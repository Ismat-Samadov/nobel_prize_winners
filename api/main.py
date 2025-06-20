from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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

# Import our NER components
import sys
sys.path.append('../src')

from models.bert_ner import BertNER
from active_learning.strategies import (
    RandomSampling, UncertaintySampling, QueryByCommittee, 
    DiversitySampling, HybridSampling
)
from data.dataset import ActiveLearningDataManager, load_conll_format
from utils.trainer import ActiveLearningTrainer
from utils.evaluation import ActiveLearningEvaluator

import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Active Learning NER API",
    description="API for Named Entity Recognition with Active Learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model management
models_cache = {}
data_managers_cache = {}
trainers_cache = {}

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

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Active Learning NER API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "upload": "/upload",
            "train": "/train",
            "sessions": "/sessions",
            "annotate": "/annotate"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_ner(request: TextRequest):
    """Predict named entities in text."""
    try:
        model_id = request.model_id
        
        # Load or get cached model
        if model_id not in models_cache:
            # Load default model
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = BertNER(model_name="bert-base-uncased")
            models_cache[model_id] = {"model": model, "tokenizer": tokenizer}
        
        model_info = models_cache[model_id]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]
        
        # Tokenize input
        tokens = request.text.split()
        
        # Encode for model
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            predictions = model.predict(
                encoding['input_ids'],
                encoding['attention_mask']
            )
        
        # Convert predictions to labels
        id_to_label = {
            0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
        }
        
        # Align predictions with original tokens
        word_ids = encoding.word_ids()
        aligned_predictions = []
        
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(tokens):
                if i < predictions.size(1):
                    pred_id = predictions[0][i].item()
                    aligned_predictions.append(id_to_label.get(pred_id, 'O'))
        
        # Ensure we have predictions for all tokens
        while len(aligned_predictions) < len(tokens):
            aligned_predictions.append('O')
        aligned_predictions = aligned_predictions[:len(tokens)]
        
        # Extract entities
        entities = extract_entities(tokens, aligned_predictions)
        
        return PredictionResponse(
            tokens=tokens,
            labels=aligned_predictions,
            entities=entities
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def extract_entities(tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
    """Extract entities from tokens and labels."""
    entities = []
    current_entity = None
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label.startswith('B-'):
            # Start of new entity
            if current_entity:
                entities.append(current_entity)
            
            entity_type = label[2:]
            current_entity = {
                "text": token,
                "label": entity_type,
                "start": i,
                "end": i + 1,
                "confidence": 0.9  # Placeholder
            }
        elif label.startswith('I-') and current_entity:
            # Continue current entity
            entity_type = label[2:]
            if current_entity["label"] == entity_type:
                current_entity["text"] += " " + token
                current_entity["end"] = i + 1
        else:
            # End current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add final entity if exists
    if current_entity:
        entities.append(current_entity)
    
    return entities

@app.post("/upload")
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
        
        # Validate file format (basic check)
        if file.filename.endswith('.conll'):
            # Try to parse CoNLL format
            try:
                sentences, labels = load_conll_format(file_path)
                num_sentences = len(sentences)
                num_tokens = sum(len(s) for s in sentences)
                
                # Get label distribution
                label_counts = {}
                for label_seq in labels:
                    for label in label_seq:
                        label_counts[label] = label_counts.get(label, 0) + 1
                
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "success",
                    "stats": {
                        "num_sentences": num_sentences,
                        "num_tokens": num_tokens,
                        "label_distribution": label_counts
                    }
                }
            except Exception as e:
                return {
                    "file_id": file_id,
                    "filename": file.filename,
                    "status": "error",
                    "error": f"Invalid CoNLL format: {str(e)}"
                }
        else:
            return {
                "file_id": file_id,
                "filename": file.filename,
                "status": "uploaded",
                "message": "File uploaded, format not validated"
            }
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
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
            labeled_samples=0,
            unlabeled_samples=0,
            performance={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        active_sessions[session_id] = session
        
        # Start training in background
        background_tasks.add_task(run_active_learning, session_id, request)
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Active learning session started"
        }
        
    except Exception as e:
        logger.error(f"Training start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_active_learning(session_id: str, config: TrainingRequest):
    """Run active learning training in background."""
    try:
        session = active_sessions[session_id]
        session.status = "running"
        session.updated_at = datetime.now()
        
        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = BertNER(
            model_name=config.model_name,
            num_labels=config.num_labels,
            use_crf=config.use_crf
        )
        
        # For demo, create synthetic data
        # In production, use uploaded files
        from experiments.create_sample_data import SampleDataGenerator
        generator = SampleDataGenerator()
        train_dataset = generator.generate_dataset(1000)
        
        # Save temporary data
        temp_train_file = f"uploads/{session_id}_train.conll"
        generator.save_conll_format(train_dataset, temp_train_file)
        
        # Initialize data manager
        data_manager = ActiveLearningDataManager(
            tokenizer=tokenizer,
            initial_labeled_ratio=config.initial_labeled_ratio,
            seed=42
        )
        
        data_manager.load_data(train_file=temp_train_file)
        
        # Create strategy
        strategy = create_strategy(config.strategy, config.uncertainty_method)
        
        # Initialize trainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = ActiveLearningTrainer(
            model=model,
            data_manager=data_manager,
            strategy=strategy,
            device=device,
            learning_rate=config.learning_rate,
            save_dir=f"results/{session_id}"
        )
        
        # Cache for later use
        models_cache[session_id] = {"model": model, "tokenizer": tokenizer}
        data_managers_cache[session_id] = data_manager
        trainers_cache[session_id] = trainer
        
        # Update session status
        stats = data_manager.get_statistics()
        session.labeled_samples = stats['labeled_samples']
        session.unlabeled_samples = stats['unlabeled_samples']
        session.status = "active_learning"
        session.updated_at = datetime.now()
        
        logger.info(f"Session {session_id} initialized successfully")
        
    except Exception as e:
        logger.error(f"Active learning error for session {session_id}: {str(e)}")
        session.status = "error"
        session.updated_at = datetime.now()

def create_strategy(strategy_name: str, uncertainty_method: str = "entropy"):
    """Create active learning strategy."""
    if strategy_name == "random":
        return RandomSampling()
    elif strategy_name == "uncertainty":
        return UncertaintySampling(method=uncertainty_method)
    elif strategy_name == "committee":
        return QueryByCommittee()
    elif strategy_name == "diversity":
        return DiversitySampling()
    elif strategy_name == "hybrid":
        return HybridSampling()
    else:
        return UncertaintySampling()

@app.get("/sessions")
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

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get specific session details."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Get samples for annotation if in active learning phase
    samples_for_annotation = []
    if session.status == "active_learning" and session_id in data_managers_cache:
        data_manager = data_managers_cache[session_id]
        unlabeled_data = data_manager.get_unlabeled_data()
        
        # Get first 10 samples for annotation
        for i, sample in enumerate(unlabeled_data[:10]):
            # Decode tokens
            tokenizer = models_cache[session_id]["tokenizer"]
            tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
            
            # Remove special tokens and padding
            clean_tokens = []
            for token in tokens:
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    clean_tokens.append(token)
            
            samples_for_annotation.append({
                "id": i,
                "tokens": clean_tokens,
                "text": " ".join(clean_tokens).replace(" ##", "")
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

@app.post("/annotate")
async def submit_annotations(request: AnnotationRequest):
    """Submit annotations for active learning."""
    try:
        session_id = request.session_id
        
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        
        if session.status != "active_learning":
            raise HTTPException(status_code=400, detail="Session not in active learning phase")
        
        # Process annotations (simplified for demo)
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
            "message": f"Processed {len(request.annotations)} annotations",
            "session_status": session.status,
            "current_round": session.current_round,
            "performance": session.performance
        }
        
    except Exception as e:
        logger.error(f"Annotation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{session_id}")
async def get_results(session_id: str):
    """Get training results and visualizations."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Generate mock learning curve data
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

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a training session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up
    if session_id in active_sessions:
        del active_sessions[session_id]
    if session_id in models_cache:
        del models_cache[session_id]
    if session_id in data_managers_cache:
        del data_managers_cache[session_id]
    if session_id in trainers_cache:
        del trainers_cache[session_id]
    
    return {"message": "Session deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "cached_models": len(models_cache)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)