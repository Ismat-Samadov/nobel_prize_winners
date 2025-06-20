# Active Learning for Named Entity Recognition

A comprehensive system for reducing labeled data requirements in Named Entity Recognition (NER) using active learning techniques.

## Overview

This project implements multiple active learning strategies to minimize the amount of labeled data needed for training effective NER models. By intelligently selecting the most informative samples for human annotation, the system can achieve competitive performance with significantly less labeled data.

## Features

- **Multiple Active Learning Strategies:**
  - Uncertainty Sampling (entropy, max probability, margin)
  - Query by Committee
  - Diversity Sampling (K-means, max distance)
  - Hybrid Sampling (uncertainty + diversity)
  - Monte Carlo Dropout for uncertainty estimation

- **Flexible NER Models:**
  - BERT-based architecture with optional CRF layer
  - Support for any HuggingFace transformer model
  - Uncertainty estimation capabilities

- **Comprehensive Evaluation:**
  - Learning curve analysis
  - Strategy comparison framework
  - Sample efficiency metrics
  - Performance visualization

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate sample data:**
   ```bash
   cd experiments
   python create_sample_data.py
   ```

3. **Run basic experiment:**
   ```bash
   python run_active_learning.py --train_file ../data/train.conll --test_file ../data/test.conll
   ```

4. **Compare strategies:**
   ```bash
   python run_active_learning.py --train_file ../data/train.conll --test_file ../data/test.conll --compare_strategies
   ```

## Project Structure

```
NER/
├── src/
│   ├── models/
│   │   └── bert_ner.py           # BERT-based NER model
│   ├── active_learning/
│   │   └── strategies.py         # Active learning strategies
│   ├── data/
│   │   └── dataset.py           # Data loading and management
│   └── utils/
│       ├── trainer.py           # Training loop with active learning
│       └── evaluation.py        # Evaluation and visualization
├── experiments/
│   ├── run_active_learning.py   # Main experiment script
│   ├── create_sample_data.py    # Sample data generation
│   ├── config.json             # Configuration file
│   └── README.md               # Detailed experiment guide
├── data/                       # Data directory
├── requirements.txt            # Dependencies
└── README.md                  # This file
```

## Active Learning Strategies

### 1. Uncertainty Sampling
Selects samples where the model is most uncertain:
- **Entropy**: High prediction entropy indicates uncertainty
- **Max Probability**: Low maximum probability indicates uncertainty  
- **Margin**: Small margin between top predictions indicates uncertainty
- **Monte Carlo Dropout**: Uses dropout at inference for uncertainty estimation

### 2. Query by Committee
Uses multiple models to identify samples with highest disagreement:
- Creates committee of models with different initializations
- Selects samples where committee members disagree most

### 3. Diversity Sampling
Ensures selected samples are diverse in the feature space:
- **K-means**: Clusters data and selects representatives
- **Max Distance**: Maximizes pairwise distances between selected samples

### 4. Hybrid Sampling
Combines uncertainty and diversity for balanced selection:
- Weighted combination of uncertainty and diversity scores
- Prevents selecting redundant uncertain samples

## Results and Benefits

Active learning typically achieves:
- **50-80% reduction** in labeling effort
- **Faster convergence** to target performance
- **Better generalization** through diverse sample selection
- **Cost-effective** annotation process

## Usage Examples

### Basic Uncertainty Sampling
```bash
python run_active_learning.py \
    --strategy uncertainty \
    --uncertainty_method entropy \
    --num_rounds 10 \
    --samples_per_round 100 \
    --train_file data/train.conll
```

### Strategy Comparison
```bash
python run_active_learning.py \
    --compare_strategies \
    --num_runs 3 \
    --output_dir results/comparison
```

### Custom Configuration
```bash
python run_active_learning.py --config experiments/config.json
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.12+
- scikit-learn
- seqeval
- matplotlib
- seaborn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{active_learning_ner_2024,
  title={Active Learning for Named Entity Recognition: Reducing Labeled Data Requirements},
  author={},
  year={2024},
  url={https://github.com/username/active-learning-ner}
}
```
