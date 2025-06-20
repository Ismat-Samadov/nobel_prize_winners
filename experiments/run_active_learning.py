#!/usr/bin/env python3
"""
Active Learning for Named Entity Recognition - Main Experiment Script

This script demonstrates how to use the active learning NER system to reduce
labeled data requirements.

Usage:
    python run_active_learning.py --config config.json
    python run_active_learning.py --strategy uncertainty --rounds 10
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import AutoTokenizer

from models.bert_ner import BertNER
from active_learning.strategies import (
    RandomSampling, UncertaintySampling, QueryByCommittee, 
    DiversitySampling, HybridSampling
)
from data.dataset import ActiveLearningDataManager
from utils.trainer import ActiveLearningTrainer
from utils.evaluation import ActiveLearningEvaluator


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('active_learning.log')
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_strategy(strategy_name: str, **kwargs):
    """Create active learning strategy based on name."""
    strategies = {
        'random': RandomSampling,
        'uncertainty': UncertaintySampling,
        'committee': QueryByCommittee,
        'diversity': DiversitySampling,
        'hybrid': HybridSampling
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Active Learning for NER")
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pre-trained model name')
    parser.add_argument('--num_labels', type=int, default=9,
                       help='Number of NER labels')
    parser.add_argument('--use_crf', action='store_true',
                       help='Use CRF layer')
    
    # Data arguments
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to training data file (CoNLL format)')
    parser.add_argument('--test_file', type=str,
                       help='Path to test data file (CoNLL format)')
    parser.add_argument('--initial_labeled_ratio', type=float, default=0.1,
                       help='Initial ratio of labeled data')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    
    # Active learning arguments
    parser.add_argument('--strategy', type=str, default='uncertainty',
                       choices=['random', 'uncertainty', 'committee', 'diversity', 'hybrid'],
                       help='Active learning strategy')
    parser.add_argument('--uncertainty_method', type=str, default='entropy',
                       choices=['entropy', 'max_prob', 'margin', 'mc_dropout'],
                       help='Uncertainty estimation method')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of active learning rounds')
    parser.add_argument('--samples_per_round', type=int, default=100,
                       help='Number of samples to select per round')
    
    # Training arguments
    parser.add_argument('--epochs_per_round', type=int, default=3,
                       help='Training epochs per round')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Warmup steps')
    
    # Device and output
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    # Evaluation
    parser.add_argument('--compare_strategies', action='store_true',
                       help='Compare multiple strategies')
    parser.add_argument('--num_runs', type=int, default=3,
                       help='Number of runs for comparison')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    if args.config:
        config = load_config(args.config)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Initialize model
    logger.info(f"Initializing model: {args.model_name}")
    model = BertNER(
        model_name=args.model_name,
        num_labels=args.num_labels,
        use_crf=args.use_crf
    )
    
    # Initialize data manager
    logger.info("Initializing data manager")
    data_manager = ActiveLearningDataManager(
        tokenizer=tokenizer,
        initial_labeled_ratio=args.initial_labeled_ratio,
        max_length=args.max_length,
        seed=args.seed
    )
    
    # Load data
    logger.info(f"Loading data from {args.train_file}")
    data_manager.load_data(
        train_file=args.train_file,
        test_file=args.test_file
    )
    
    # Print initial statistics
    stats = data_manager.get_statistics()
    logger.info(f"Initial statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Labeled samples: {stats['labeled_samples']}")
    logger.info(f"  Unlabeled samples: {stats['unlabeled_samples']}")
    logger.info(f"  Labeled ratio: {stats['labeled_ratio']:.2%}")
    
    if args.compare_strategies:
        # Compare multiple strategies
        logger.info("Comparing multiple active learning strategies")
        
        strategies = [
            ('Random', RandomSampling(seed=args.seed)),
            ('Uncertainty (Entropy)', UncertaintySampling(method='entropy')),
            ('Uncertainty (Max Prob)', UncertaintySampling(method='max_prob')),
            ('Uncertainty (Margin)', UncertaintySampling(method='margin')),
            ('Diversity (K-means)', DiversitySampling(method='kmeans')),
            ('Hybrid', HybridSampling(uncertainty_weight=0.7, diversity_weight=0.3))
        ]
        
        # Initialize evaluator
        evaluator = ActiveLearningEvaluator(save_dir=args.output_dir)
        
        # Run comparison
        all_results = {}
        
        for strategy_name, strategy in strategies:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running strategy: {strategy_name}")
            logger.info(f"{'='*50}")
            
            strategy_results = []
            
            for run in range(args.num_runs):
                logger.info(f"Run {run + 1}/{args.num_runs}")
                
                # Create fresh model and data manager for each run
                model = BertNER(
                    model_name=args.model_name,
                    num_labels=args.num_labels,
                    use_crf=args.use_crf
                )
                
                data_manager = ActiveLearningDataManager(
                    tokenizer=tokenizer,
                    initial_labeled_ratio=args.initial_labeled_ratio,
                    max_length=args.max_length,
                    seed=args.seed + run  # Different seed for each run
                )
                
                data_manager.load_data(
                    train_file=args.train_file,
                    test_file=args.test_file
                )
                
                # Initialize trainer
                trainer = ActiveLearningTrainer(
                    model=model,
                    data_manager=data_manager,
                    strategy=strategy,
                    device=device,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    warmup_steps=args.warmup_steps,
                    save_dir=os.path.join(args.output_dir, strategy_name.replace(' ', '_').lower(), f'run_{run}')
                )
                
                # Run active learning
                history = trainer.active_learning_loop(
                    num_rounds=args.num_rounds,
                    samples_per_round=args.samples_per_round,
                    epochs_per_round=args.epochs_per_round
                )
                
                strategy_results.append(history)
            
            all_results[strategy_name] = strategy_results
        
        # Evaluate and save results
        logger.info("Evaluating results and creating plots")
        
        # Learning curves
        learning_curves = evaluator.evaluate_learning_curves(all_results)
        
        # Sample efficiency
        efficiency_stats = evaluator.evaluate_sample_efficiency(all_results)
        
        # Comparison table
        comparison_df = evaluator.create_comparison_table(all_results)
        
        # Save comprehensive results
        evaluator.save_experiment_summary(
            experiment_config=vars(args),
            results={
                'learning_curves': learning_curves,
                'efficiency_stats': efficiency_stats,
                'comparison_table': comparison_df.to_dict()
            }
        )
        
        logger.info(f"Comparison results saved to {args.output_dir}")
        
    else:
        # Single strategy experiment
        logger.info(f"Running single strategy experiment: {args.strategy}")
        
        # Create strategy
        strategy_kwargs = {}
        if args.strategy == 'uncertainty':
            strategy_kwargs['method'] = args.uncertainty_method
        elif args.strategy == 'random':
            strategy_kwargs['seed'] = args.seed
        
        strategy = create_strategy(args.strategy, **strategy_kwargs)
        
        # Initialize trainer
        trainer = ActiveLearningTrainer(
            model=model,
            data_manager=data_manager,
            strategy=strategy,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            save_dir=args.output_dir
        )
        
        # Run active learning
        history = trainer.active_learning_loop(
            num_rounds=args.num_rounds,
            samples_per_round=args.samples_per_round,
            epochs_per_round=args.epochs_per_round
        )
        
        # Evaluate results
        evaluator = ActiveLearningEvaluator(save_dir=args.output_dir)
        evaluator.plot_training_progress(history)
        
        # Save results
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Experiment completed. Results saved to {args.output_dir}")
        
        # Print final statistics
        if history['eval_scores']:
            final_scores = history['eval_scores'][-1]
            logger.info("Final Performance:")
            logger.info(f"  F1: {final_scores.get('f1', 0.0):.4f}")
            logger.info(f"  Precision: {final_scores.get('precision', 0.0):.4f}")
            logger.info(f"  Recall: {final_scores.get('recall', 0.0):.4f}")
            logger.info(f"  Accuracy: {final_scores.get('accuracy', 0.0):.4f}")


if __name__ == "__main__":
    main()