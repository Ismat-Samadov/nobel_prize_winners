import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer
import json
import os
from collections import defaultdict


class NERDataset(Dataset):
    """Dataset class for Named Entity Recognition with active learning support."""
    
    def __init__(self, 
                 tokenizer: AutoTokenizer,
                 max_length: int = 128,
                 label_to_id: Optional[Dict[str, int]] = None):
        """
        Args:
            tokenizer: Pre-trained tokenizer
            max_length: Maximum sequence length
            label_to_id: Mapping from label names to IDs
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id or self._get_default_label_mapping()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        self.texts = []
        self.labels = []
        self.encodings = []
        self.is_labeled = []  # Track which samples are labeled
    
    def _get_default_label_mapping(self) -> Dict[str, int]:
        """Get default BIO label mapping for NER."""
        return {
            'O': 0,
            'B-PER': 1, 'I-PER': 2,
            'B-ORG': 3, 'I-ORG': 4,
            'B-LOC': 5, 'I-LOC': 6,
            'B-MISC': 7, 'I-MISC': 8
        }
    
    def add_samples(self, 
                   texts: List[List[str]], 
                   labels: Optional[List[List[str]]] = None,
                   is_labeled: bool = True):
        """Add samples to the dataset.
        
        Args:
            texts: List of tokenized texts (list of tokens)
            labels: List of corresponding labels (optional for unlabeled data)
            is_labeled: Whether the samples are labeled
        """
        for i, text in enumerate(texts):
            self.texts.append(text)
            
            if labels is not None and i < len(labels):
                self.labels.append(labels[i])
            else:
                self.labels.append(['O'] * len(text))  # Dummy labels for unlabeled data
            
            self.is_labeled.append(is_labeled)
        
        # Re-encode all samples
        self._encode_samples()
    
    def _encode_samples(self):
        """Encode all text samples using the tokenizer."""
        self.encodings = []
        
        for text, labels in zip(self.texts, self.labels):
            encoding = self._encode_sample(text, labels)
            self.encodings.append(encoding)
    
    def _encode_sample(self, tokens: List[str], labels: List[str]) -> Dict:
        """Encode a single sample."""
        # Tokenize and align labels
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        # Align labels with tokenized input
        word_ids = encoded.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 label
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword of a word gets the label
                if word_idx < len(labels):
                    aligned_labels.append(self.label_to_id.get(labels[word_idx], 0))
                else:
                    aligned_labels.append(-100)
            else:
                # Subsequent subwords get -100 (or could use same label)
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }
    
    def get_labeled_indices(self) -> List[int]:
        """Get indices of labeled samples."""
        return [i for i, is_lab in enumerate(self.is_labeled) if is_lab]
    
    def get_unlabeled_indices(self) -> List[int]:
        """Get indices of unlabeled samples."""
        return [i for i, is_lab in enumerate(self.is_labeled) if not is_lab]
    
    def mark_as_labeled(self, indices: List[int], new_labels: List[List[str]]):
        """Mark samples as labeled and update their labels.
        
        Args:
            indices: Indices of samples to mark as labeled
            new_labels: New labels for the samples
        """
        for i, idx in enumerate(indices):
            if idx < len(self.is_labeled):
                self.is_labeled[idx] = True
                if i < len(new_labels):
                    self.labels[idx] = new_labels[i]
        
        # Re-encode affected samples
        for idx in indices:
            if idx < len(self.texts):
                encoding = self._encode_sample(self.texts[idx], self.labels[idx])
                self.encodings[idx] = encoding
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict:
        if idx >= len(self.encodings):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.encodings)}")
        
        return self.encodings[idx]
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in labeled samples."""
        label_counts = defaultdict(int)
        
        for i, is_lab in enumerate(self.is_labeled):
            if is_lab and i < len(self.labels):
                for label in self.labels[i]:
                    label_counts[label] += 1
        
        return dict(label_counts)


def load_conll_format(file_path: str, 
                     encoding: str = 'utf-8',
                     sep: str = '\t') -> Tuple[List[List[str]], List[List[str]]]:
    """Load CoNLL format data.
    
    Args:
        file_path: Path to CoNLL format file
        encoding: File encoding
        sep: Column separator
        
    Returns:
        Tuple of (sentences, labels) where each is a list of lists
    """
    sentences = []
    labels = []
    
    current_tokens = []
    current_labels = []
    
    with open(file_path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            
            if not line:  # Empty line indicates sentence boundary
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split(sep)
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[-1]  # Last column is usually the label
                    
                    current_tokens.append(token)
                    current_labels.append(label)
    
    # Add last sentence if file doesn't end with empty line
    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)
    
    return sentences, labels


def create_data_loaders(dataset: NERDataset,
                       batch_size: int = 16,
                       labeled_only: bool = False,
                       unlabeled_only: bool = False,
                       shuffle: bool = True) -> DataLoader:
    """Create data loader for the dataset.
    
    Args:
        dataset: NER dataset
        batch_size: Batch size
        labeled_only: Whether to include only labeled samples
        unlabeled_only: Whether to include only unlabeled samples
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader
    """
    if labeled_only:
        indices = dataset.get_labeled_indices()
        subset = torch.utils.data.Subset(dataset, indices)
    elif unlabeled_only:
        indices = dataset.get_unlabeled_indices()
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset
    
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


class ActiveLearningDataManager:
    """Manages data for active learning experiments."""
    
    def __init__(self, 
                 tokenizer: AutoTokenizer,
                 initial_labeled_ratio: float = 0.1,
                 max_length: int = 128,
                 seed: int = 42):
        """
        Args:
            tokenizer: Tokenizer for encoding text
            initial_labeled_ratio: Ratio of initial labeled data
            max_length: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.initial_labeled_ratio = initial_labeled_ratio
        self.max_length = max_length
        self.seed = seed
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.dataset = NERDataset(tokenizer, max_length)
        self.test_dataset = None
    
    def load_data(self, 
                  train_file: str,
                  test_file: Optional[str] = None,
                  val_file: Optional[str] = None):
        """Load training and test data.
        
        Args:
            train_file: Path to training data file
            test_file: Path to test data file (optional)
            val_file: Path to validation data file (optional)
        """
        # Load training data
        train_texts, train_labels = load_conll_format(train_file)
        
        # Split into initial labeled and unlabeled sets
        n_initial = int(len(train_texts) * self.initial_labeled_ratio)
        indices = np.random.permutation(len(train_texts))
        
        initial_indices = indices[:n_initial]
        unlabeled_indices = indices[n_initial:]
        
        # Add initial labeled samples
        initial_texts = [train_texts[i] for i in initial_indices]
        initial_labels = [train_labels[i] for i in initial_indices]
        self.dataset.add_samples(initial_texts, initial_labels, is_labeled=True)
        
        # Add unlabeled samples
        unlabeled_texts = [train_texts[i] for i in unlabeled_indices]
        self.dataset.add_samples(unlabeled_texts, is_labeled=False)
        
        # Load test data if provided
        if test_file:
            test_texts, test_labels = load_conll_format(test_file)
            self.test_dataset = NERDataset(self.tokenizer, self.max_length)
            self.test_dataset.add_samples(test_texts, test_labels, is_labeled=True)
    
    def get_training_loader(self, batch_size: int = 16) -> DataLoader:
        """Get data loader for labeled training data."""
        return create_data_loaders(
            self.dataset, 
            batch_size=batch_size, 
            labeled_only=True, 
            shuffle=True
        )
    
    def get_unlabeled_data(self) -> List[Dict]:
        """Get unlabeled data for active learning selection."""
        unlabeled_indices = self.dataset.get_unlabeled_indices()
        return [self.dataset.encodings[i] for i in unlabeled_indices]
    
    def get_test_loader(self, batch_size: int = 16) -> Optional[DataLoader]:
        """Get test data loader."""
        if self.test_dataset is None:
            return None
        
        return create_data_loaders(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    def label_samples(self, 
                     selected_indices: List[int], 
                     oracle_labels: Optional[List[List[str]]] = None):
        """Label selected samples (simulate oracle labeling).
        
        Args:
            selected_indices: Indices of selected unlabeled samples
            oracle_labels: True labels (if None, uses ground truth from dataset)
        """
        unlabeled_indices = self.dataset.get_unlabeled_indices()
        
        # Convert selected indices from unlabeled space to full dataset space
        actual_indices = [unlabeled_indices[i] for i in selected_indices 
                         if i < len(unlabeled_indices)]
        
        if oracle_labels is None:
            # Use ground truth labels (simulate perfect oracle)
            oracle_labels = [self.dataset.labels[i] for i in actual_indices]
        
        self.dataset.mark_as_labeled(actual_indices, oracle_labels)
    
    def get_statistics(self) -> Dict[str, Union[int, float, Dict]]:
        """Get dataset statistics."""
        total_samples = len(self.dataset)
        labeled_samples = len(self.dataset.get_labeled_indices())
        unlabeled_samples = len(self.dataset.get_unlabeled_indices())
        
        label_dist = self.dataset.get_label_distribution()
        
        return {
            'total_samples': total_samples,
            'labeled_samples': labeled_samples,
            'unlabeled_samples': unlabeled_samples,
            'labeled_ratio': labeled_samples / total_samples if total_samples > 0 else 0,
            'label_distribution': label_dist
        }