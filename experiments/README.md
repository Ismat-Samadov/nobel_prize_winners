# Active Learning NER Experiments

This directory contains scripts and configurations for running active learning experiments with Named Entity Recognition.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

2. **Generate sample data:**
   ```bash
   python create_sample_data.py
   ```

3. **Run basic active learning experiment:**
   ```bash
   python run_active_learning.py --train_file ../data/train.conll --test_file ../data/test.conll
   ```

4. **Compare multiple strategies:**
   ```bash
   python run_active_learning.py --train_file ../data/train.conll --test_file ../data/test.conll --compare_strategies
   ```

## Configuration

### Using Configuration File
```bash
python run_active_learning.py --config config.json
```

### Command Line Arguments

#### Model Parameters
- `--model_name`: Pre-trained model (default: bert-base-uncased)
- `--num_labels`: Number of NER labels (default: 9)
- `--use_crf`: Add CRF layer for sequence labeling

#### Data Parameters
- `--train_file`: Training data in CoNLL format (required)
- `--test_file`: Test data in CoNLL format
- `--initial_labeled_ratio`: Initial ratio of labeled data (default: 0.1)
- `--max_length`: Maximum sequence length (default: 128)

#### Active Learning Parameters
- `--strategy`: Active learning strategy (random, uncertainty, committee, diversity, hybrid)
- `--uncertainty_method`: Uncertainty method (entropy, max_prob, margin, mc_dropout)
- `--num_rounds`: Number of active learning rounds (default: 10)
- `--samples_per_round`: Samples to select per round (default: 100)

#### Training Parameters
- `--epochs_per_round`: Training epochs per round (default: 3)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)

## Active Learning Strategies

### 1. Random Sampling
Baseline strategy that randomly selects samples for labeling.
```bash
python run_active_learning.py --strategy random --train_file ../data/train.conll
```

### 2. Uncertainty Sampling
Selects samples with highest prediction uncertainty.
```bash
# Entropy-based uncertainty
python run_active_learning.py --strategy uncertainty --uncertainty_method entropy

# Maximum probability uncertainty
python run_active_learning.py --strategy uncertainty --uncertainty_method max_prob

# Margin-based uncertainty
python run_active_learning.py --strategy uncertainty --uncertainty_method margin

# Monte Carlo dropout uncertainty
python run_active_learning.py --strategy uncertainty --uncertainty_method mc_dropout
```

### 3. Query by Committee
Uses multiple models to identify samples with highest disagreement.
```bash
python run_active_learning.py --strategy committee --train_file ../data/train.conll
```

### 4. Diversity Sampling
Selects diverse samples to maximize coverage of the feature space.
```bash
python run_active_learning.py --strategy diversity --train_file ../data/train.conll
```

### 5. Hybrid Sampling
Combines uncertainty and diversity sampling.
```bash
python run_active_learning.py --strategy hybrid --train_file ../data/train.conll
```

## Experiment Examples

### Basic Uncertainty Sampling Experiment
```bash
python run_active_learning.py \
    --train_file ../data/train.conll \
    --test_file ../data/test.conll \
    --strategy uncertainty \
    --uncertainty_method entropy \
    --num_rounds 15 \
    --samples_per_round 50 \
    --epochs_per_round 3 \
    --output_dir ./results/uncertainty_entropy
```

### Strategy Comparison Experiment
```bash
python run_active_learning.py \
    --train_file ../data/train.conll \
    --test_file ../data/test.conll \
    --compare_strategies \
    --num_rounds 10 \
    --samples_per_round 100 \
    --num_runs 5 \
    --output_dir ./results/strategy_comparison
```

### Custom Model Experiment
```bash
python run_active_learning.py \
    --model_name distilbert-base-uncased \
    --use_crf \
    --train_file ../data/train.conll \
    --test_file ../data/test.conll \
    --strategy hybrid \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --output_dir ./results/distilbert_crf
```

## Output and Results

The experiments generate several types of output:

### 1. Training Logs
- Console output with training progress
- `active_learning.log` file with detailed logs

### 2. Model Checkpoints
- `checkpoint_round_X.pt`: Model state after each round
- `training_history.json`: Training metrics history

### 3. Evaluation Results
- `learning_curves_f1.png`: F1 score progression
- `strategy_comparison.csv`: Performance comparison table
- `training_progress.png`: Multi-metric training plots
- `experiment_summary.json`: Comprehensive results

### 4. Analysis Files
- `classification_report.txt`: Detailed performance metrics
- `confusion_matrix.png`: Confusion matrix visualization

## Data Format

The system expects CoNLL format data with tokens and labels separated by tabs:

```
John    B-PER
Smith   I-PER
works   O
at      O
Google  B-ORG
in      O
California  B-LOC
.       O

The     O
company B-ORG
...
```

## Customization

### Adding New Strategies
1. Create a new strategy class inheriting from `ActiveLearningStrategy`
2. Implement the `select_samples` method
3. Add the strategy to the `create_strategy` function

### Custom Data Preprocessing
Modify the `ActiveLearningDataManager` class to handle different data formats or add custom preprocessing steps.

### Custom Evaluation Metrics
Extend the `ActiveLearningEvaluator` class to add domain-specific evaluation metrics.

## Performance Tips

1. **Use GPU**: Set `--device cuda` for faster training
2. **Batch Processing**: Increase `--batch_size` for better GPU utilization
3. **Early Stopping**: Monitor validation loss to avoid overfitting
4. **Hyperparameter Tuning**: Experiment with learning rates and model architectures

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or sequence length
2. **Slow training**: Use smaller models or reduce data size for testing
3. **Poor performance**: Check data quality and label consistency
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode
Add `--log_level DEBUG` for detailed debugging information.

## Citation

If you use this active learning NER system in your research, please cite:

```bibtex
@software{active_learning_ner,
  title={Active Learning for Named Entity Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/active-learning-ner}
}
```