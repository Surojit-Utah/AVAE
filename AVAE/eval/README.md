# AVAE Evaluation Pipeline

This directory contains all evaluation scripts and results for the AVAE (Aggregate Variational Autoencoder) paper published at ICPR 2024.

## Directory Structure

```
eval/
├── study_fid/                  # FID (Fréchet Inception Distance) evaluation
│   ├── fid_stats_data/        # Pre-computed FID statistics for datasets
│   ├── generate_samples/      # Sample generation scripts
│   └── compute_fid/           # FID computation scripts
│
├── study_prd/                 # Precision-Recall evaluation
│   ├── PRD/                   # Official NeurIPS 2018 implementation (customized)
│   └── logs/                  # PRD results
│
├── entropy/                   # Entropy of aggregate posterior evaluation
│   ├── compute_entropy.py     # Entropy computation script
│   └── logs/                  # Entropy results
│
└── MSE/                       # Mean Squared Error (reconstruction) evaluation
    ├── compute_mse.py         # MSE computation script
    └── logs/                  # MSE results
```

## Evaluation Metrics

This evaluation pipeline computes four key metrics reported in the paper:

### 1. **FID (Fréchet Inception Distance)**
- **Measures**: Quality and diversity of generated samples
- **Lower is better**
- **Paper Results**: MNIST 13.27±0.34, CelebA 46.0±0.42, CIFAR10 90.93±6.65
- **Script**: `study_fid/compute_fid/compute_n_plot_fid.py`

### 2. **Precision & Recall**
- **Precision**: Quality/fidelity of generated samples
- **Recall**: Coverage/diversity of generation
- **Paper Results**: 
  - MNIST: P=0.92±0.02, R=0.98±0.00
  - CelebA: P=0.88±0.02, R=0.85±0.02
  - CIFAR10: P=0.72±0.05, R=0.67±0.04
- **Script**: `study_prd/PRD/precision-recall-distributions/prd_from_image_folders.py`

### 3. **Entropy of Aggregate Posterior**
- **Measures**: How well latent distribution matches prior Gaussian (after whitening)
- **Higher is better** (Gaussian has maximum entropy)
- **Paper Results**: MNIST 7.56±0.10, CelebA 30.96±0.02, CIFAR10 55.64±0.00
- **Script**: `entropy/compute_entropy.py`

### 4. **MSE (Mean Squared Error)**
- **Measures**: Reconstruction quality
- **Lower is better**
- **Paper Results**: MNIST 0.0041±0.0004, CIFAR10 0.0062±0.0002
- **Script**: `MSE/compute_mse.py`

## Quick Start

### Prerequisites

```bash
# Required dependencies
pip install tensorflow==2.8.0  # or tensorflow-gpu==2.8.0
pip install numpy scipy matplotlib scikit-learn imageio
```

### Dataset Setup

**MNIST & CIFAR10**: Automatically downloaded via TensorFlow/Keras

**CelebA**: Set environment variable:
```bash
# PowerShell
$env:CELEBA_DATA_DIR = "C:\path\to\celeba\data"

# Bash
export CELEBA_DATA_DIR="/path/to/celeba/data"
```

Expected file: `train_images_npy.npy` in the specified directory.

### Running Evaluations

#### 1. Generate Samples (Required First)

```bash
cd study_fid/generate_samples

# MNIST (config_id=0)
python generate_samples.py --config_id 0 --seed 0

# CelebA (config_id=1) - requires CELEBA_DATA_DIR
python generate_samples.py --config_id 1 --seed 0

# CIFAR10 (config_id=2)
python generate_samples.py --config_id 2 --seed 0
```

This generates samples for all 5 training runs (run_id_1 through run_id_5).

#### 2. Compute FID Scores

```bash
cd study_fid/compute_fid

# MNIST
python compute_n_plot_fid.py --config_id 0 --seed 0 --gen_type generation

# CelebA
python compute_n_plot_fid.py --config_id 1 --seed 0 --gen_type generation

# CIFAR10
python compute_n_plot_fid.py --config_id 2 --seed 0 --gen_type generation
```

Results saved to: `study_fid/compute_fid/logs/{dataset}/generation/fid_stat.txt`

#### 3. Compute Precision-Recall

```bash
cd study_prd/PRD/precision-recall-distributions

# MNIST
python prd_from_image_folders.py --dataset_name MNIST --method_name AVAE --eval_mode generation

# CelebA (requires CELEBA_DATA_DIR)
python prd_from_image_folders.py --dataset_name CelebA --method_name AVAE --eval_mode generation

# CIFAR10
python prd_from_image_folders.py --dataset_name CIFAR10 --method_name AVAE --eval_mode generation
```

Results saved to: `study_prd/logs/{dataset}/generation/prd_scores.txt`

#### 4. Compute Entropy

```bash
cd entropy

# MNIST
python compute_entropy.py --config_id 0 --seed 0

# CelebA
python compute_entropy.py --config_id 1 --seed 0

# CIFAR10
python compute_entropy.py --config_id 2 --seed 0
```

Results saved to: `entropy/logs/{dataset}/entropy_stat.txt`

#### 5. Compute MSE

```bash
cd MSE

# MNIST
python compute_mse.py --config_id 0

# CelebA
python compute_mse.py --config_id 1

# CIFAR10
python compute_mse.py --config_id 2
```

Results saved to: `MSE/logs/{dataset}/mse_error.txt`

## Verification Status

All evaluation results in the logs have been verified against the ICPR 2024 paper:

- ✅ **FID**: Exact match for all datasets
- ✅ **Entropy**: Exact match for all datasets
- ✅ **Precision-Recall**: Close match (differences < 0.004, due to rounding)
- ✅ **MSE**: Exact match for MNIST and CIFAR10

## Paper Citation

If you use this evaluation pipeline, please cite:

```bibtex
@inproceedings{saha2024avae,
  title={Matching Aggregate Posteriors in the Variational Autoencoder},
  author={Saha, Surojit and Joshi, Sarang and Whitaker, Ross},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2024}
}
```

## Additional Documentation

- **FID**: See `study_fid/compute_fid/README.md` (if exists)
- **Precision-Recall**: See `study_prd/README.md`
- **Entropy**: Mathematical details in paper Section 4.2.3
- **MSE**: Standard reconstruction loss

## Troubleshooting

### TensorFlow not found
```bash
pip install tensorflow==2.8.0
```

### CelebA path not configured
```bash
$env:CELEBA_DATA_DIR = "C:\your\path\to\celeba"
```

### Generated samples not found
Run `generate_samples.py` first (step 1 above).

### FID stats not found
FID statistics should be in `study_fid/fid_stats_data/`:
- `fid_stats_mnist.npz`
- `fid_stats_celeba.npz`
- `fid_stats_cifar10_train.npz`

These files contain pre-computed Inception statistics for the real datasets.

## Technical Notes

1. **All evaluations use 5 independent training runs** (run_id 1-5)
2. **Results reported as mean ± std** across these 5 runs
3. **FID**: Uses InceptionV3 features at pool_3 layer
4. **PRD**: Uses official NeurIPS 2018 implementation with 20 clusters
5. **Entropy**: KDE-based estimation with bias-corrected bandwidth
6. **MSE**: Per-pixel reconstruction error

---

**Repository**: https://github.com/Surojit-Utah/AVAE  
**Paper**: ICPR 2024
