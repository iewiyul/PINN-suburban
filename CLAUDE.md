# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Physics-Informed Neural Network (PINN)** project for automatic suburban layout generation. The system takes road network images as input and generates building layouts (circles with x, y, radius) using energy minimization.

**Key distinction from traditional PINNs:** This project uses energy-based optimization (borrowing hPINN techniques) rather than solving differential equations. It's an unsupervised learning approach with hard constraints + soft rewards.

## Common Commands

```bash
# Training
python train.py                    # Standard training
python train_viz.py                # Training with live visualization

# Data Processing
python data/merge_channel.py       # Generate 5-channel features from road images
python data/load_original_roads.py # Load original road network data

# Visualization
python visualize.py                # Generate layout visualizations from checkpoints
python debug_energy.py             # Debug energy function components

# Generate test layouts
python generate_test_layouts.py    # Generate sample layouts for testing
```

## High-Level Architecture

```
Input: Road Features [batch, 5, 256, 256]
  ├─ Channel 0: Binary road mask
  ├─ Channel 1: Distance field (to nearest road)
  ├─ Channel 2: Road density
  ├─ Channel 3: Road type (main/branch)
  └─ Channel 4: Road centrality
          ↓
Network: ResNet Feature Extractor (net/models.py)
  ├─ Conv7x7 + MaxPool (256→64)
  ├─ ResBlock layers (64→128→256 channels)
  ├─ Global Average Pooling
  └─ MLP (256→512→256→90 outputs)
          ↓
Hard Constraints (output_transform)
  ├─ x, y ∈ (0, 1) via sigmoid
  └─ r ∈ [0.025, 0.12] via sigmoid scaling
          ↓
Output: Layout [batch, 30, 3] = (x, y, radius)
          ↓
Energy Function (energy/energy_function.py)
  ├─ Constraints (penalties): boundary, overlap, space_to_road, radius
  └─ Rewards (incentives): road_distance, road_coverage
          ↓
Training: Minimize E(layout) via gradient descent
```

## Key Module Organization

### Network Architecture (`net/models.py`)
- **Current model:** ResNet-based `RoadFeatureExtractor`
- **Alternative versions (commented out):** VGG-style, spatial attention, conditional batch normalization
- The model uses **hard constraints** in the output layer to guarantee x,y ∈ (0,1) and r ∈ [0.025, 0.12]

### Energy System (`energy/`)
- **Constraint modules** (`energy/constraint/`): Penalize constraint violations
  - `constraint_boundary.py`: Buildings must stay within [0,1] bounds
  - `constraint_overlap.py`: Buildings shouldn't overlap
  - `constraint_space_to_road.py`: Buildings shouldn't overlap roads
  - `constraint_radius.py`: Radius size preferences
- **Reward modules** (`energy/reward/`): Encourage desirable layouts
  - `reward_road_distance_relationship.py`: Keep buildings near roads
  - `reward_road_coverage.py`: Uniform spatial distribution

### Data Pipeline (`data/`)
- **Input:** Original road network images from `data/original mask/`
- **Processing:**
  1. `load_original_roads.py`: Load grayscale road images
  2. `data_augment.py`: Apply rotations/flips (8× augmentation)
  3. `channel_process.py`: Extract 5-channel features
  4. `merge_channel.py`: Orchestrate full pipeline + save features
- **Output:** `data/processed_features/` contains `all_features.npy` and `all_circles.npy`

### Training Configuration (`train.py`)
Key hyperparameters in `TrainingConfig`:
- Energy weights in `energy/energy_function.py` (CONSTRAINT_WEIGHTS, REWARD_WEIGHTS)
- Learning rate scheduling: `reduce_on_plateau`, `cosine`, or `none`
- Early stopping with configurable patience
- Gradient clipping to prevent explosion

## Important Implementation Details

### Road Circle Representation
Road networks are represented as **circles** `(x, y, r)` for efficient constraint checking. Each road pixel becomes a circle with small radius `r=0.01`. This is pre-computed and stored in `all_circles.npy` to avoid runtime overhead.

### Energy Function Philosophy
- **Total Energy = Σ(weight × component)**
- Constraint violations increase energy (penalty)
- Good layout properties decrease energy (reward)
- Training minimizes total energy via gradient descent
- **No ground truth labels needed** - fully unsupervised

### Model Versions
The codebase contains multiple architecture iterations (all in `net/models.py`):
1. **VGG-style** (original, commented)
2. **Spatial Attention** (adds attention mechanism)
3. **Conditional Batch Normalization** (CBN, conditions BN on road features)
4. **ResNet** (current default - most stable)

To switch versions, uncomment the desired class and comment out the current `RoadFeatureExtractor`.

### Data Augmentation
Each original road image generates **8 augmented variants** via `data_augment.py`. This significantly expands the training dataset and improves generalization.

## File Naming Conventions

- `all_features.npy`: Batch of all road feature tensors [N, 5, 256, 256]
- `all_circles.npy`: Padded batch of road circle tensors [N, max_road_pixels, 3]
- `best_model.pth`: Checkpoint with best validation energy
- `checkpoint_epoch_N.pth`: Periodic checkpoints every 50 epochs
- `training_history.npy`: Training curves (train_energy, val_energy, lr)

## Typical Workflow

1. **Prepare data:** Run `python data/merge_channel.py` to generate features
2. **Train:** Run `python train.py` and monitor energy convergence
3. **Visualize:** Run `python visualize.py` to generate layout samples
4. **Debug:** Run `python debug_energy.py` to analyze energy components
5. **Iterate:** Adjust energy weights in `energy/energy_function.py` and retrain

## GPU Support
The code automatically detects CUDA availability. Training will use GPU if available, otherwise falls back to CPU. Use `config.device` to check.
