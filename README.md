# Neural Process & Normalizing Flow (NPNF)

A research codebase focused on combining Neural Processes with Normalizing Flows for plant phenotyping time series prediction with improved uncertainty quantification.

## Project Overview

This repository contains the implementation of Neural Process models enhanced with Normalizing Flows for time series prediction of plant traits. The key innovation is the integration of normalizing flows for better uncertainty modeling in plant phenotyping applications.

## Key Features

- **Neural Process Models**: Uncertainty-aware time series prediction
- **Normalizing Flows**: Enhanced uncertainty quantification
- **Plant Phenotyping**: Specialized for agricultural trait prediction
- **Hydra Configuration**: Type-safe configuration management with hydra-zen
- **Accelerate Integration**: Distributed training and mixed precision support
- **Wandb Logging**: Comprehensive experiment tracking
- **UV Package Manager**: Fast dependency resolution and environment management

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd npnf

# Install dependencies using uv
uv sync

# Verify installation
uv run python -c "import npnf; print('Success')"
```

## Usage

### Training Models

```bash
# Train Neural Process model
uv run python src/npnf/scripts/neural_process/train.py

# Generate predictions
uv run python src/npnf/scripts/neural_process/predict.py

# Create visualization GIFs
uv run python src/npnf/scripts/neural_process/create_gifs.py

# Calculate metrics
uv run python src/npnf/scripts/neural_process/calculate_metrics.py
```

### Dataset Preparation

```bash
# Download FIP1 dataset
uv run python src/npnf/scripts/fip1/get_fip1.py

# Create soybean dataset
uv run python src/npnf/scripts/fip1/create_soybean_dataset.py
```

### Code Quality

```bash
# Run code formatting and linting
uv run ruff check
uv run ruff format
```

## Architecture

### Core Structure
- `src/npnf/` - Main package containing model configurations and scripts
- `src/npnf/configs/` - Hydra configuration files for neural process models
- `src/npnf/scripts/` - Executable scripts organized by functionality
- `datasets/` - Dataset handling and preprocessing (workspace member)
- `models/` - Model implementations (workspace member)

### Research Focus

#### Neural Process Models with Normalizing Flows
- **Purpose**: Time series prediction of plant traits with improved uncertainty quantification
- **Key Scripts**: `train.py`, `predict.py`, `create_gifs.py`, `calculate_metrics.py`
- **Features**: Uncertainty-based sampling, sequential prediction, GIF generation
- **Research Innovation**: Integration of normalizing flows for better uncertainty modeling

### Configuration Management
- Configurations are defined in `src/npnf/configs/`
- Uses hydra-zen for type-safe parameter management
- Device configurations available for different hardware setups

### Data Handling

#### Supported Datasets
- **FIP1 dataset** - Main focus for current research
- **synthetic dataset** - Synthetic growth data for training and evaluation
- **functions.py dataset** - Used for testing purposes

#### Dataset Notes
- Focus development efforts on the FIP1 dataset primarily
- Outputs stored in `outputs/` directory with timestamped runs

### Reproducibility
- All scripts include seed setting for reproducible experiments
- Checkpoint saving and resuming supported
- Comprehensive logging with wandb integration

## Requirements

- Python >=3.12,<3.12.7
- CUDA 12.6 with PyTorch nightly builds
- UV package manager

## Contributing

This is a research codebase. Please follow the existing code style and run the linting tools before submitting changes.

## License

MIT License. See [LICENSE.txt](LICENSE.txt).

## Authors

Mike Boss - mboss@ethz.ch