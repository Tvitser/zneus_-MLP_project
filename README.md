# Speed Dating MLP Experiments

A comprehensive machine learning project for analyzing and predicting outcomes in speed dating datasets using multi-layer perceptrons (MLPs). This project includes robust data preprocessing, configurable neural network experiments, and comprehensive hyperparameter search capabilities.

## 📋 Project Overview

This repository implements an end-to-end machine learning pipeline for the speed dating domain, featuring:

- **Robust Data Pipeline**: Advanced preprocessing with range parsing, percent handling, and intelligent imputation
- **Flexible MLP Architectures**: Configurable neural networks with dropout, batch/layer normalization, skip connections, and bottleneck layers
- **Comprehensive Experiment Tracking**: Integration with Weights & Biases (WandB) and local CSV logging
- **Hyperparameter Optimization**: Grid search and random search capabilities
- **Modular Code Structure**: Organized into reusable tools, models, and scripts

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**: torch, scikit-learn, pandas, numpy, seaborn, matplotlib, wandb (optional)

### 2. Run a Single Experiment

```bash
python -m tools.experiment --config examples/speeddating_config.json --csv project/speeddating.csv
```

### 3. Run Local Grid Search

```bash
python -m tools.sweeper --mode grid \
  --base-cfg examples/speeddating_config.json \
  --grid examples/grid_params.json \
  --out runs/grid_results.csv
```

### 4. Run WandB Sweep (Optional)

```bash
pip install wandb
wandb login
python -m tools.sweeper --mode wandb \
  --base-cfg examples/speeddating_config.json \
  --sweep examples/wandb_sweep.yaml
```

## 📁 Project Structure

```
├── tools/
│   ├── main.py              # SpeedDatingEDA class with data cleaning & preprocessing
│   ├── experiment.py        # Experiment runner with train/val/test splits & tracking
│   └── sweeper.py          # Grid search and random search implementations
├── models/
│   └── mlp.py              # Configurable MLP model with advanced features
├── examples/
│   ├── speeddating_config.json   # Base experiment configuration
│   ├── grid_params.json          # Grid search parameters
│   └── wandb_sweep.yaml          # WandB sweep configuration
├── project/
│   └── speeddating.csv      # Speed dating dataset
├── runs/                    # Experiment outputs (logs, metrics, results)
├── notebooks/               # Jupyter notebooks for analysis & visualization
├── docs/
│   └── consultation_notes.md # Project consultation & iteration notes
└── requirements.txt         # Python dependencies
```

## 🔧 Key Features

### Data Preprocessing
- **Range Parsing**: Converts range strings (e.g., "1-5") to midpoints
- **Percent Handling**: Normalizes percentage values
- **Smart Imputation**: Context-aware missing value handling
- **Duplicate Removal**: Identifies and removes duplicate records
- **Column Cleanup**: Removes high-missingness columns (configurable threshold)

**Entry Point**: `tools/main.py` → `SpeedDatingEDA` class

### Experiment Configuration

Experiments are defined via JSON configuration files:

```json
{
  "model": {
    "hidden_layers": [128, 64, 32],
    "dropout": 0.3,
    "use_batch_norm": true,
    "use_skip_connections": true
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
  },
  "data": {
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15
  },
  "tracking": {
    "use_wandb": false
  }
}
```

### Hyperparameter Search

**Grid Search**: Define parameter combinations in `grid_params.json`
- Local execution via `tools/sweeper.py`
- Results aggregated to CSV for easy analysis

**Random Search**: Programmatic parameter overrides with WandB sweep config
- Distributed sweep support
- Real-time monitoring in WandB dashboard

## 📊 Outputs & Results

Each experiment run generates:

- **history.json**: Training/validation loss curves
- **metrics.json**: Final evaluation metrics (accuracy, loss, etc.)
- **Logs**: Checkpoints and debug logs in run directory

Grid search aggregates results to: `runs/grid_results.csv`

## 📚 Model Architecture Options

The MLP supports:

- **Dropout**: Regularization to prevent overfitting
- **Batch Normalization**: Accelerates training convergence
- **Layer Normalization**: Alternative normalization approach
- **Skip Connections**: Residual connections between layers
- **Bottleneck Layers**: Configurable layer reduction for dimensionality control

## 🔍 Experiment Tracking

### Local Tracking
- CSV logging of grid search results
- JSON output per run

### Weights & Biases Integration
- Enable with `"use_wandb": true` in config
- Monitor training in real-time
- Compare multiple runs easily
- Define sweeps in `examples/wandb_sweep.yaml`

## 📖 Documentation

- **Inline Comments**: Docstrings throughout codebase
- **Consultation Notes**: See `docs/consultation_notes.md` for design decisions and iterations
- **Jupyter Notebooks**: Analysis and visualization examples in `notebooks/`

## 🛠️ Development Workflow

1. Create a feature branch for new experiments
2. Update configuration files in `examples/`
3. Document changes and insights
4. Commit results and findings
5. Create pull request for review

## 📝 Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib / seaborn
- wandb (optional, for advanced tracking)

## 🤝 Contributing

To extend this project:

1. Add new model architectures to `models/`
2. Create new preprocessing steps in `tools/main.py`
3. Update configuration examples
4. Document new features in this README

## 📄 License

See repository for license details.

---

**Last Updated**: November 2025
