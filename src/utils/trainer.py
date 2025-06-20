import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict
import os
import json
from datetime import datetime

from ..models.bert_ner import BertNER
from ..active_learning.strategies import ActiveLearningStrategy
from ..data.dataset import ActiveLearningDataManager


class ActiveLearningTrainer:
    """Trainer for active learning NER experiments."""
    
    def __init__(self,
                 model: BertNER,
                 data_manager: ActiveLearningDataManager,
                 strategy: ActiveLearningStrategy,
                 device: str = "cpu",
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 warmup_steps: int = 0,
                 max_grad_norm: float = 1.0,
                 patience: int = 3,
                 save_dir: str = "./checkpoints"):
        """
        Args:
            model: NER model to train
            data_manager: Data manager for active learning
            strategy: Active learning strategy
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps for scheduler
            max_grad_norm: Maximum gradient norm for clipping
            patience: Patience for early stopping
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.data_manager = data_manager
        self.strategy = strategy
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Track training history
        self.history = {
            'train_loss': [],
            'eval_scores': [],
            'labeled_samples': [],
            'active_learning_rounds': []
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('ActiveLearningTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update parameters
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, 
                eval_loader: DataLoader,
                return_predictions: bool = False) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            eval_loader: Evaluation data loader
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()
                    num_batches += 1
                
                # Get predictions
                predictions = self.model.predict(
                    batch['input_ids'],
                    batch['attention_mask']
                )
                
                # Collect predictions and labels for metric calculation
                mask = batch['attention_mask'].bool()
                labels = batch['labels']
                
                for i in range(predictions.size(0)):
                    seq_len = mask[i].sum().item()
                    pred_seq = predictions[i][:seq_len].cpu().numpy()
                    label_seq = labels[i][:seq_len].cpu().numpy()
                    
                    # Filter out special tokens (-100)
                    valid_indices = label_seq != -100
                    if valid_indices.any():
                        all_predictions.append(pred_seq[valid_indices])
                        all_labels.append(label_seq[valid_indices])
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        
        if num_batches > 0:
            metrics['eval_loss'] = total_loss / num_batches
        
        if return_predictions:
            metrics['predictions'] = all_predictions
            metrics['labels'] = all_labels
        
        return metrics
    
    def _calculate_metrics(self, 
                          predictions: List[np.ndarray],
                          labels: List[np.ndarray]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
        from seqeval.scheme import IOB2
        
        # Convert predictions and labels to string format for seqeval
        id_to_label = self.model.tokenizer.get_vocab() if hasattr(self.model, 'tokenizer') else {
            0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
            5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
        }
        
        # Default mapping if tokenizer doesn't have label mapping
        if not isinstance(list(id_to_label.values())[0], str):
            id_to_label = {
                0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG',
                5: 'B-LOC', 6: 'I-LOC', 7: 'B-MISC', 8: 'I-MISC'
            }
        
        pred_labels = []
        true_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            pred_tags = [id_to_label.get(p, 'O') for p in pred_seq]
            true_tags = [id_to_label.get(l, 'O') for l in label_seq]
            
            pred_labels.append(pred_tags)
            true_labels.append(true_tags)
        
        # Calculate metrics using seqeval
        try:
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, scheme=IOB2)
            recall = recall_score(true_labels, pred_labels, scheme=IOB2)
            f1 = f1_score(true_labels, pred_labels, scheme=IOB2)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        except Exception as e:
            self.logger.warning(f"Error calculating seqeval metrics: {e}")
            
            # Fallback to simple accuracy
            correct = sum(
                sum(p == t for p, t in zip(pred_seq, true_seq))
                for pred_seq, true_seq in zip(predictions, labels)
            )
            total = sum(len(seq) for seq in labels)
            
            return {
                'accuracy': correct / total if total > 0 else 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    def active_learning_loop(self,
                           num_rounds: int = 10,
                           samples_per_round: int = 100,
                           epochs_per_round: int = 3,
                           eval_interval: int = 1) -> Dict[str, List]:
        """Run active learning training loop.
        
        Args:
            num_rounds: Number of active learning rounds
            samples_per_round: Number of samples to select per round
            epochs_per_round: Number of training epochs per round
            eval_interval: Evaluate every N rounds
            
        Returns:
            Training history
        """
        self.logger.info("Starting active learning training loop")
        self.logger.info(f"Rounds: {num_rounds}, Samples per round: {samples_per_round}")
        
        # Initial evaluation
        test_loader = self.data_manager.get_test_loader()
        if test_loader:
            initial_metrics = self.evaluate(test_loader)
            self.history['eval_scores'].append(initial_metrics)
            self.logger.info(f"Initial F1: {initial_metrics.get('f1', 0.0):.4f}")
        
        for round_num in range(num_rounds):
            self.logger.info(f"\n=== Active Learning Round {round_num + 1}/{num_rounds} ===")
            
            # Get current statistics
            stats = self.data_manager.get_statistics()
            self.logger.info(f"Labeled samples: {stats['labeled_samples']}")
            self.logger.info(f"Unlabeled samples: {stats['unlabeled_samples']}")
            
            # Active learning sample selection
            if round_num > 0:  # Skip selection in first round (use initial labeled data)
                unlabeled_data = self.data_manager.get_unlabeled_data()
                
                if len(unlabeled_data) == 0:
                    self.logger.info("No more unlabeled data available")
                    break
                
                n_select = min(samples_per_round, len(unlabeled_data))
                self.logger.info(f"Selecting {n_select} samples using {type(self.strategy).__name__}")
                
                # Select samples using active learning strategy
                selected_indices = self.strategy.select_samples(
                    model=self.model,
                    unlabeled_data=unlabeled_data,
                    n_samples=n_select,
                    device=self.device
                )
                
                # Label selected samples (oracle)
                self.data_manager.label_samples(selected_indices)
                self.logger.info(f"Added {len(selected_indices)} samples to labeled set")
            
            # Train model
            train_loader = self.data_manager.get_training_loader()
            train_loss = self._train_rounds(train_loader, epochs_per_round)
            
            self.history['train_loss'].append(train_loss)
            self.history['labeled_samples'].append(stats['labeled_samples'])
            
            # Evaluate model
            if (round_num + 1) % eval_interval == 0 and test_loader:
                eval_metrics = self.evaluate(test_loader)
                self.history['eval_scores'].append(eval_metrics)
                
                self.logger.info(f"Round {round_num + 1} Results:")
                self.logger.info(f"  F1: {eval_metrics.get('f1', 0.0):.4f}")
                self.logger.info(f"  Precision: {eval_metrics.get('precision', 0.0):.4f}")
                self.logger.info(f"  Recall: {eval_metrics.get('recall', 0.0):.4f}")
                self.logger.info(f"  Accuracy: {eval_metrics.get('accuracy', 0.0):.4f}")
            
            # Save checkpoint
            self._save_checkpoint(round_num + 1)
        
        self.logger.info("Active learning training completed!")
        return self.history
    
    def _train_rounds(self, train_loader: DataLoader, num_epochs: int) -> float:
        """Train the model for specified number of epochs.
        
        Returns:
            Average training loss across epochs
        """
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        ) if self.warmup_steps > 0 else None
        
        total_loss = 0.0
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = self.train_epoch(train_loader, optimizer, scheduler)
            total_loss += epoch_loss
            self.logger.info(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")
        
        return total_loss / num_epochs
    
    def _save_checkpoint(self, round_num: int):
        """Save model checkpoint and training history."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'round': round_num,
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_round_{round_num}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save history as JSON
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint.
        
        Returns:
            Round number of loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        
        round_num = checkpoint.get('round', 0)
        self.logger.info(f"Loaded checkpoint from round {round_num}")
        
        return round_num
    
    def compare_strategies(self,
                          strategies: List[Tuple[str, ActiveLearningStrategy]],
                          num_rounds: int = 10,
                          samples_per_round: int = 100,
                          epochs_per_round: int = 3,
                          num_runs: int = 3) -> Dict[str, Dict]:
        """Compare multiple active learning strategies.
        
        Args:
            strategies: List of (name, strategy) tuples
            num_rounds: Number of active learning rounds
            samples_per_round: Samples to select per round
            epochs_per_round: Training epochs per round
            num_runs: Number of runs for averaging
            
        Returns:
            Comparison results for each strategy
        """
        results = {}
        
        for strategy_name, strategy in strategies:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Evaluating strategy: {strategy_name}")
            self.logger.info(f"{'='*50}")
            
            strategy_results = []
            
            for run in range(num_runs):
                self.logger.info(f"\nRun {run + 1}/{num_runs}")
                
                # Reset data manager for each run
                # Note: This would require re-initializing with original data
                self.strategy = strategy
                
                # Run active learning loop
                history = self.active_learning_loop(
                    num_rounds=num_rounds,
                    samples_per_round=samples_per_round,
                    epochs_per_round=epochs_per_round
                )
                
                strategy_results.append(history)
            
            results[strategy_name] = strategy_results
        
        return results