import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np


class BertNER(nn.Module):
    """BERT-based Named Entity Recognition model with uncertainty estimation."""
    
    def __init__(self, 
                 model_name: str = "bert-base-uncased",
                 num_labels: int = 9,
                 dropout_rate: float = 0.1,
                 use_crf: bool = False):
        super(BertNER, self).__init__()
        
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Optional CRF layer
        if use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(num_labels, batch_first=True)
            except ImportError:
                print("Warning: torchcrf not installed. Install with: pip install pytorch-crf")
                self.crf = None
                self.use_crf = False
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Get BERT embeddings
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Get logits
        logits = self.classifier(sequence_output)
        
        result = {"logits": logits}
        
        if labels is not None:
            if self.use_crf and self.crf is not None:
                # Use CRF loss
                log_likelihood = self.crf(logits, labels, mask=attention_mask.byte())
                loss = -log_likelihood
            else:
                # Use cross-entropy loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            result["loss"] = loss
        
        return result
    
    def predict(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Make predictions without computing gradients."""
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)
            logits = outputs["logits"]
            
            if self.use_crf and self.crf is not None:
                predictions = self.crf.decode(logits, mask=attention_mask.byte())
                # Convert list of predictions to tensor
                max_len = logits.size(1)
                batch_size = logits.size(0)
                pred_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)
                
                for i, pred in enumerate(predictions):
                    pred_tensor[i, :len(pred)] = torch.tensor(pred)
                
                return pred_tensor
            else:
                return torch.argmax(logits, dim=-1)
    
    def get_uncertainty(self, 
                       input_ids: torch.Tensor,
                       attention_mask: torch.Tensor,
                       token_type_ids: Optional[torch.Tensor] = None,
                       method: str = "entropy") -> torch.Tensor:
        """Calculate prediction uncertainty for active learning."""
        
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, token_type_ids)
            logits = outputs["logits"]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            if method == "entropy":
                # Calculate entropy-based uncertainty
                log_probs = torch.log(probs + 1e-8)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
                # Mask out padding tokens
                entropy = entropy * attention_mask.float()
                
                # Return mean entropy per sequence
                return entropy.sum(dim=1) / attention_mask.sum(dim=1).float()
            
            elif method == "max_prob":
                # Use 1 - max_probability as uncertainty
                max_probs, _ = torch.max(probs, dim=-1)
                uncertainty = 1 - max_probs
                
                # Mask out padding tokens
                uncertainty = uncertainty * attention_mask.float()
                
                # Return mean uncertainty per sequence
                return uncertainty.sum(dim=1) / attention_mask.sum(dim=1).float()
            
            elif method == "margin":
                # Use margin between top two predictions
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                margin = sorted_probs[:, :, 0] - sorted_probs[:, :, 1]
                uncertainty = 1 - margin
                
                # Mask out padding tokens
                uncertainty = uncertainty * attention_mask.float()
                
                # Return mean uncertainty per sequence
                return uncertainty.sum(dim=1) / attention_mask.sum(dim=1).float()
            
            else:
                raise ValueError(f"Unknown uncertainty method: {method}")
    
    def enable_dropout(self):
        """Enable dropout for Monte Carlo dropout uncertainty estimation."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def get_mc_uncertainty(self, 
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor,
                          token_type_ids: Optional[torch.Tensor] = None,
                          num_samples: int = 10) -> torch.Tensor:
        """Calculate uncertainty using Monte Carlo dropout."""
        
        self.enable_dropout()
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                outputs = self.forward(input_ids, attention_mask, token_type_ids)
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                predictions.append(probs)
        
        # Stack predictions and calculate variance
        predictions = torch.stack(predictions)  # [num_samples, batch_size, seq_len, num_labels]
        
        # Calculate mean and variance across samples
        mean_probs = predictions.mean(dim=0)
        var_probs = predictions.var(dim=0)
        
        # Use mean variance across labels as uncertainty
        uncertainty = var_probs.mean(dim=-1)
        
        # Mask out padding tokens
        uncertainty = uncertainty * attention_mask.float()
        
        # Return mean uncertainty per sequence
        return uncertainty.sum(dim=1) / attention_mask.sum(dim=1).float()