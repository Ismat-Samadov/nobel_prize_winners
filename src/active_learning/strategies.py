import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import random


class ActiveLearningStrategy(ABC):
    """Base class for active learning strategies."""
    
    @abstractmethod
    def select_samples(self, 
                      model,
                      unlabeled_data: List[Dict],
                      n_samples: int,
                      **kwargs) -> List[int]:
        """Select samples for labeling.
        
        Args:
            model: The trained model
            unlabeled_data: List of unlabeled samples
            n_samples: Number of samples to select
            
        Returns:
            List of indices of selected samples
        """
        pass


class RandomSampling(ActiveLearningStrategy):
    """Random sampling baseline."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def select_samples(self, 
                      model,
                      unlabeled_data: List[Dict],
                      n_samples: int,
                      **kwargs) -> List[int]:
        """Randomly select samples."""
        indices = list(range(len(unlabeled_data)))
        return random.sample(indices, min(n_samples, len(indices)))


class UncertaintySampling(ActiveLearningStrategy):
    """Uncertainty-based sampling strategies."""
    
    def __init__(self, method: str = "entropy"):
        """
        Args:
            method: Uncertainty method ('entropy', 'max_prob', 'margin', 'mc_dropout')
        """
        self.method = method
    
    def select_samples(self, 
                      model,
                      unlabeled_data: List[Dict],
                      n_samples: int,
                      device: str = "cpu",
                      **kwargs) -> List[int]:
        """Select samples with highest uncertainty."""
        
        model.eval()
        uncertainties = []
        
        for i, sample in enumerate(unlabeled_data):
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
            
            if self.method == "mc_dropout":
                uncertainty = model.get_mc_uncertainty(
                    input_ids, attention_mask, 
                    num_samples=kwargs.get('mc_samples', 10)
                )
            else:
                uncertainty = model.get_uncertainty(
                    input_ids, attention_mask, method=self.method
                )
            
            uncertainties.append((i, uncertainty.item()))
        
        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n_samples indices
        return [idx for idx, _ in uncertainties[:n_samples]]


class QueryByCommittee(ActiveLearningStrategy):
    """Query by Committee active learning strategy."""
    
    def __init__(self, 
                 committee_size: int = 3,
                 disagreement_measure: str = "vote_entropy"):
        """
        Args:
            committee_size: Number of models in the committee
            disagreement_measure: How to measure disagreement ('vote_entropy', 'consensus_entropy')
        """
        self.committee_size = committee_size
        self.disagreement_measure = disagreement_measure
    
    def select_samples(self, 
                      model,
                      unlabeled_data: List[Dict],
                      n_samples: int,
                      device: str = "cpu",
                      committee_models: Optional[List] = None,
                      **kwargs) -> List[int]:
        """Select samples with highest committee disagreement."""
        
        if committee_models is None:
            # Create committee by using different dropout seeds
            committee_models = [model] * self.committee_size
        
        disagreements = []
        
        for i, sample in enumerate(unlabeled_data):
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
            
            # Get predictions from all committee members
            predictions = []
            for committee_model in committee_models:
                committee_model.eval()
                if committee_model == model:
                    # Use MC dropout for diversity
                    pred = self._get_mc_prediction(
                        committee_model, input_ids, attention_mask
                    )
                else:
                    pred = committee_model.predict(input_ids, attention_mask)
                predictions.append(pred.cpu().numpy())
            
            # Calculate disagreement
            disagreement = self._calculate_disagreement(predictions, attention_mask)
            disagreements.append((i, disagreement))
        
        # Sort by disagreement (descending)
        disagreements.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n_samples indices
        return [idx for idx, _ in disagreements[:n_samples]]
    
    def _get_mc_prediction(self, model, input_ids, attention_mask):
        """Get prediction using Monte Carlo dropout."""
        model.enable_dropout()
        with torch.no_grad():
            outputs = model.forward(input_ids, attention_mask)
            logits = outputs["logits"]
            return torch.argmax(logits, dim=-1)
    
    def _calculate_disagreement(self, predictions, attention_mask):
        """Calculate disagreement between committee predictions."""
        predictions = np.array(predictions)  # [committee_size, batch_size, seq_len]
        mask = attention_mask.cpu().numpy().astype(bool)
        
        if self.disagreement_measure == "vote_entropy":
            # Calculate vote entropy
            disagreement = 0
            seq_len = predictions.shape[2]
            
            for pos in range(seq_len):
                if not mask[0, pos]:  # Skip padding
                    continue
                
                votes = predictions[:, 0, pos]
                unique_votes, counts = np.unique(votes, return_counts=True)
                
                if len(unique_votes) > 1:
                    probs = counts / len(votes)
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    disagreement += entropy
            
            return disagreement / mask[0].sum()  # Normalize by sequence length
        
        else:
            raise ValueError(f"Unknown disagreement measure: {self.disagreement_measure}")


class DiversitySampling(ActiveLearningStrategy):
    """Diversity-based sampling using clustering."""
    
    def __init__(self, method: str = "kmeans"):
        """
        Args:
            method: Diversity method ('kmeans', 'max_distance')
        """
        self.method = method
    
    def select_samples(self, 
                      model,
                      unlabeled_data: List[Dict],
                      n_samples: int,
                      device: str = "cpu",
                      **kwargs) -> List[int]:
        """Select diverse samples using embeddings."""
        
        model.eval()
        embeddings = []
        
        # Get embeddings for all unlabeled samples
        for sample in unlabeled_data:
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                # Use [CLS] token embedding or mean pooling
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
                embeddings.append(embedding.flatten())
        
        embeddings = np.array(embeddings)
        
        if self.method == "kmeans":
            return self._kmeans_selection(embeddings, n_samples)
        elif self.method == "max_distance":
            return self._max_distance_selection(embeddings, n_samples)
        else:
            raise ValueError(f"Unknown diversity method: {self.method}")
    
    def _kmeans_selection(self, embeddings: np.ndarray, n_samples: int) -> List[int]:
        """Select samples using K-means clustering."""
        n_clusters = min(n_samples, len(embeddings))
        
        if n_clusters == 1:
            return [0]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select one sample from each cluster (closest to centroid)
        selected_indices = []
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            if len(cluster_indices) > 0:
                centroid = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(embeddings[cluster_indices] - centroid, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)
        
        return selected_indices[:n_samples]
    
    def _max_distance_selection(self, embeddings: np.ndarray, n_samples: int) -> List[int]:
        """Select samples to maximize pairwise distances."""
        if n_samples >= len(embeddings):
            return list(range(len(embeddings)))
        
        selected_indices = [0]  # Start with first sample
        
        for _ in range(n_samples - 1):
            max_min_distance = -1
            best_candidate = -1
            
            for candidate in range(len(embeddings)):
                if candidate in selected_indices:
                    continue
                
                # Calculate minimum distance to already selected samples
                min_distance = float('inf')
                for selected_idx in selected_indices:
                    distance = np.linalg.norm(embeddings[candidate] - embeddings[selected_idx])
                    min_distance = min(min_distance, distance)
                
                # Keep track of candidate with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate != -1:
                selected_indices.append(best_candidate)
        
        return selected_indices


class HybridSampling(ActiveLearningStrategy):
    """Combine uncertainty and diversity sampling."""
    
    def __init__(self, 
                 uncertainty_weight: float = 0.7,
                 diversity_weight: float = 0.3,
                 uncertainty_method: str = "entropy",
                 diversity_method: str = "kmeans"):
        """
        Args:
            uncertainty_weight: Weight for uncertainty component
            diversity_weight: Weight for diversity component
            uncertainty_method: Uncertainty sampling method
            diversity_method: Diversity sampling method
        """
        self.uncertainty_weight = uncertainty_weight
        self.diversity_weight = diversity_weight
        self.uncertainty_sampler = UncertaintySampling(uncertainty_method)
        self.diversity_sampler = DiversitySampling(diversity_method)
    
    def select_samples(self, 
                      model,
                      unlabeled_data: List[Dict],
                      n_samples: int,
                      device: str = "cpu",
                      **kwargs) -> List[int]:
        """Select samples combining uncertainty and diversity."""
        
        # Get uncertainty scores
        model.eval()
        uncertainty_scores = {}
        
        for i, sample in enumerate(unlabeled_data):
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
            
            uncertainty = model.get_uncertainty(input_ids, attention_mask, method="entropy")
            uncertainty_scores[i] = uncertainty.item()
        
        # Get embeddings for diversity calculation
        embeddings = []
        for sample in unlabeled_data:
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(device)
            attention_mask = torch.tensor(sample['attention_mask']).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding.flatten())
        
        embeddings = np.array(embeddings)
        
        # Calculate diversity scores
        diversity_scores = {}
        for i in range(len(unlabeled_data)):
            # Calculate average distance to all other samples
            distances = []
            for j in range(len(unlabeled_data)):
                if i != j:
                    distance = np.linalg.norm(embeddings[i] - embeddings[j])
                    distances.append(distance)
            diversity_scores[i] = np.mean(distances) if distances else 0
        
        # Normalize scores
        max_uncertainty = max(uncertainty_scores.values()) if uncertainty_scores else 1
        max_diversity = max(diversity_scores.values()) if diversity_scores else 1
        
        # Combine scores
        combined_scores = []
        for i in range(len(unlabeled_data)):
            uncertainty_norm = uncertainty_scores[i] / max_uncertainty
            diversity_norm = diversity_scores[i] / max_diversity
            
            combined_score = (self.uncertainty_weight * uncertainty_norm + 
                            self.diversity_weight * diversity_norm)
            combined_scores.append((i, combined_score))
        
        # Sort by combined score (descending)
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n_samples indices
        return [idx for idx, _ in combined_scores[:n_samples]]