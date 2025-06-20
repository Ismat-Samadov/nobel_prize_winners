import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from collections import defaultdict
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report as seq_classification_report
from seqeval.scheme import IOB2


class ActiveLearningEvaluator:
    """Evaluator for active learning experiments."""
    
    def __init__(self, save_dir: str = "./results"):
        """
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def evaluate_learning_curves(self, 
                                histories: Dict[str, List[Dict]],
                                metric: str = 'f1',
                                save_plot: bool = True) -> Dict[str, np.ndarray]:
        """Evaluate and plot learning curves for different strategies.
        
        Args:
            histories: Dictionary mapping strategy names to list of training histories
            metric: Metric to plot ('f1', 'precision', 'recall', 'accuracy')
            save_plot: Whether to save the plot
            
        Returns:
            Dictionary mapping strategy names to metric arrays
        """
        plt.figure(figsize=(10, 6))
        
        results = {}
        
        for strategy_name, strategy_histories in histories.items():
            # Extract metric values across rounds for all runs
            all_curves = []
            
            for history in strategy_histories:
                if 'eval_scores' in history and history['eval_scores']:
                    curve = [score.get(metric, 0.0) for score in history['eval_scores']]
                    all_curves.append(curve)
            
            if all_curves:
                # Pad curves to same length
                max_len = max(len(curve) for curve in all_curves)
                padded_curves = []
                
                for curve in all_curves:
                    padded = curve + [curve[-1]] * (max_len - len(curve))
                    padded_curves.append(padded)
                
                # Calculate mean and std
                mean_curve = np.mean(padded_curves, axis=0)
                std_curve = np.std(padded_curves, axis=0)
                
                # Plot
                x = range(len(mean_curve))
                plt.plot(x, mean_curve, label=strategy_name, marker='o')
                plt.fill_between(x, 
                               mean_curve - std_curve, 
                               mean_curve + std_curve, 
                               alpha=0.2)
                
                results[strategy_name] = mean_curve
        
        plt.xlabel('Active Learning Round')
        plt.ylabel(f'{metric.capitalize()}')
        plt.title(f'Active Learning Performance Comparison ({metric.capitalize()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, f'learning_curves_{metric}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return results
    
    def evaluate_sample_efficiency(self, 
                                 histories: Dict[str, List[Dict]],
                                 target_performance: float = 0.85,
                                 metric: str = 'f1') -> Dict[str, Dict]:
        """Evaluate sample efficiency of different strategies.
        
        Args:
            histories: Dictionary mapping strategy names to training histories
            target_performance: Target performance threshold
            metric: Metric to evaluate
            
        Returns:
            Sample efficiency statistics for each strategy
        """
        efficiency_stats = {}
        
        for strategy_name, strategy_histories in histories.items():
            samples_to_target = []
            max_performances = []
            
            for history in strategy_histories:
                if 'eval_scores' not in history or not history['eval_scores']:
                    continue
                
                performances = [score.get(metric, 0.0) for score in history['eval_scores']]
                labeled_samples = history.get('labeled_samples', [])
                
                # Find samples needed to reach target performance
                target_reached = False
                for i, perf in enumerate(performances):
                    if perf >= target_performance:
                        if i < len(labeled_samples):
                            samples_to_target.append(labeled_samples[i])
                        target_reached = True
                        break
                
                if not target_reached and labeled_samples:
                    # If target not reached, use maximum samples
                    samples_to_target.append(labeled_samples[-1])
                
                # Track maximum performance
                max_performances.append(max(performances) if performances else 0.0)
            
            # Calculate statistics
            efficiency_stats[strategy_name] = {
                'mean_samples_to_target': np.mean(samples_to_target) if samples_to_target else np.inf,
                'std_samples_to_target': np.std(samples_to_target) if samples_to_target else 0.0,
                'target_reached_ratio': len([s for s in samples_to_target if s < np.inf]) / len(samples_to_target) if samples_to_target else 0.0,
                'mean_max_performance': np.mean(max_performances) if max_performances else 0.0,
                'std_max_performance': np.std(max_performances) if max_performances else 0.0
            }
        
        return efficiency_stats
    
    def create_comparison_table(self, 
                              histories: Dict[str, List[Dict]],
                              metrics: List[str] = ['f1', 'precision', 'recall', 'accuracy'],
                              save_csv: bool = True) -> pd.DataFrame:
        """Create comparison table of final performance for different strategies.
        
        Args:
            histories: Dictionary mapping strategy names to training histories
            metrics: List of metrics to include
            save_csv: Whether to save as CSV
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for strategy_name, strategy_histories in histories.items():
            final_performances = {metric: [] for metric in metrics}
            
            for history in strategy_histories:
                if 'eval_scores' in history and history['eval_scores']:
                    final_scores = history['eval_scores'][-1]  # Last evaluation
                    
                    for metric in metrics:
                        final_performances[metric].append(final_scores.get(metric, 0.0))
            
            # Calculate mean and std for each metric
            row_data = {'Strategy': strategy_name}
            
            for metric in metrics:
                if final_performances[metric]:
                    mean_val = np.mean(final_performances[metric])
                    std_val = np.std(final_performances[metric])
                    row_data[f'{metric.capitalize()} (Mean)'] = mean_val
                    row_data[f'{metric.capitalize()} (Std)'] = std_val
                else:
                    row_data[f'{metric.capitalize()} (Mean)'] = 0.0
                    row_data[f'{metric.capitalize()} (Std)'] = 0.0
            
            comparison_data.append(row_data)
        
        df = pd.DataFrame(comparison_data)
        
        if save_csv:
            csv_path = os.path.join(self.save_dir, 'strategy_comparison.csv')
            df.to_csv(csv_path, index=False)
        
        return df
    
    def analyze_label_distribution(self, 
                                 histories: Dict[str, List[Dict]],
                                 save_plot: bool = True) -> Dict[str, Dict]:
        """Analyze how label distribution changes during active learning.
        
        Args:
            histories: Dictionary mapping strategy names to training histories
            save_plot: Whether to save distribution plots
            
        Returns:
            Label distribution analysis for each strategy
        """
        # This would require tracking label distributions in training history
        # For now, return placeholder
        return {}
    
    def plot_uncertainty_distribution(self, 
                                    model,
                                    data_loader,
                                    device: str = "cpu",
                                    save_plot: bool = True):
        """Plot distribution of prediction uncertainties.
        
        Args:
            model: Trained model
            data_loader: Data loader for evaluation
            device: Device to use
            save_plot: Whether to save the plot
        """
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                uncertainty = model.get_uncertainty(
                    batch['input_ids'],
                    batch['attention_mask'],
                    method='entropy'
                )
                
                uncertainties.extend(uncertainty.cpu().numpy())
        
        plt.figure(figsize=(10, 6))
        plt.hist(uncertainties, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Uncertainty (Entropy)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Uncertainties')
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, 'uncertainty_distribution.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_confusion_matrix(self, 
                              predictions: List[List[int]],
                              labels: List[List[int]],
                              label_names: Optional[List[str]] = None,
                              save_plot: bool = True) -> np.ndarray:
        """Create confusion matrix for NER predictions.
        
        Args:
            predictions: List of prediction sequences
            labels: List of true label sequences
            label_names: Names of labels
            save_plot: Whether to save the plot
            
        Returns:
            Confusion matrix
        """
        # Flatten predictions and labels
        flat_predictions = []
        flat_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            flat_predictions.extend(pred_seq)
            flat_labels.extend(label_seq)
        
        # Remove special tokens (-100)
        valid_indices = [i for i, label in enumerate(flat_labels) if label != -100]
        flat_predictions = [flat_predictions[i] for i in valid_indices]
        flat_labels = [flat_labels[i] for i in valid_indices]
        
        # Create confusion matrix
        cm = confusion_matrix(flat_labels, flat_predictions)
        
        if save_plot:
            plt.figure(figsize=(10, 8))
            
            if label_names is None:
                label_names = [f'Label_{i}' for i in range(cm.shape[0])]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=label_names, yticklabels=label_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            
            plot_path = os.path.join(self.save_dir, 'confusion_matrix.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return cm
    
    def generate_classification_report(self, 
                                     predictions: List[List[str]],
                                     labels: List[List[str]],
                                     save_report: bool = True) -> str:
        """Generate detailed classification report.
        
        Args:
            predictions: List of prediction sequences (string labels)
            labels: List of true label sequences (string labels)
            save_report: Whether to save the report
            
        Returns:
            Classification report string
        """
        try:
            report = seq_classification_report(labels, predictions, scheme=IOB2)
        except Exception as e:
            # Fallback to sklearn classification report
            flat_predictions = [pred for seq in predictions for pred in seq]
            flat_labels = [label for seq in labels for label in seq]
            report = classification_report(flat_labels, flat_predictions)
        
        if save_report:
            report_path = os.path.join(self.save_dir, 'classification_report.txt')
            with open(report_path, 'w') as f:
                f.write(report)
        
        return report
    
    def save_experiment_summary(self, 
                              experiment_config: Dict,
                              results: Dict,
                              filename: str = 'experiment_summary.json'):
        """Save comprehensive experiment summary.
        
        Args:
            experiment_config: Configuration used for the experiment
            results: Results from the experiment
            filename: Name of the summary file
        """
        summary = {
            'experiment_config': experiment_config,
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        summary_path = os.path.join(self.save_dir, filename)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def plot_training_progress(self, 
                             history: Dict[str, List],
                             save_plot: bool = True):
        """Plot training progress over active learning rounds.
        
        Args:
            history: Training history dictionary
            save_plot: Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training loss
        if 'train_loss' in history and history['train_loss']:
            axes[0, 0].plot(history['train_loss'], marker='o')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot F1 score
        if 'eval_scores' in history and history['eval_scores']:
            f1_scores = [score.get('f1', 0.0) for score in history['eval_scores']]
            axes[0, 1].plot(f1_scores, marker='o', color='green')
            axes[0, 1].set_title('F1 Score')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('F1')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot labeled samples
        if 'labeled_samples' in history and history['labeled_samples']:
            axes[1, 0].plot(history['labeled_samples'], marker='o', color='orange')
            axes[1, 0].set_title('Labeled Samples')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Number of Samples')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot multiple metrics
        if 'eval_scores' in history and history['eval_scores']:
            metrics = ['precision', 'recall', 'accuracy']
            for metric in metrics:
                values = [score.get(metric, 0.0) for score in history['eval_scores']]
                axes[1, 1].plot(values, marker='o', label=metric.capitalize())
            
            axes[1, 1].set_title('Multiple Metrics')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = os.path.join(self.save_dir, 'training_progress.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()