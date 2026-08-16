# Precision-Recall-Distributions (PRD) Computation

This directory contains scripts and results for computing precision and recall metrics for generated samples using the official NeurIPS 2018 implementation.

## Overview

Precision and recall are complementary metrics that evaluate different aspects of generative models:

- **Precision**: Measures the quality/fidelity of generated samples (how many generated samples look real)
- **Recall**: Measures the diversity/coverage of generated samples (how much of the real data distribution is covered)

## Methodology

We use the official precision-recall-distributions implementation from:

> Sajjadi, M. S., Bachem, O., Lucic, M., Bousquet, O., & Gelly, S. (2018)  
> "Assessing Generative Models via Precision and Recall"  
> NeurIPS 2018

**Repository**: https://github.com/msmsajjadi/precision-recall-distributions

The implementation:
1. Extracts InceptionV3 features from both real and generated samples
2. Uses clustering-based approach (20 clusters by default)
3. Computes precision-recall curve over 1001 angles
4. Reports F_β scores (F_8 for recall, F_1/8 for precision)

## Directory Structure

```
study_prd/
├── README.md                          (this file)
├── PRD/
│   └── precision-recall-distributions/
│       ├── prd_from_image_folders.py  (main script - customized for AVAE)
│       ├── prd_score.py               (PRD computation)
│       ├── inception.py               (Inception network wrapper)
│       ├── get_real_data.py           (dataset loader - customized)
│       ├── inception_models/
│       │   └── inception.pb           (pre-trained model)
│       └── requirements.txt
└── logs/                              (results)
    ├── MNIST/generation/prd_scores.txt
    ├── CelebA/generation/prd_scores.txt
    └── CIFAR10/generation/prd_scores.txt
```

## Installation

### Required Dependencies

```bash
# Install TensorFlow (required for Inception model)
pip install tensorflow==2.8.0
# or for GPU:
pip install tensorflow-gpu==2.8.0

# Install other dependencies
pip install imageio numpy scipy matplotlib scikit-learn
```

### Verify Installation

```bash
cd PRD/precision-recall-distributions
python -c "import tensorflow; import imageio; import numpy; print('Dependencies OK')"
```

## Usage

### For MNIST & CIFAR10 (No Setup Required)

These datasets are automatically downloaded via Keras:

```bash
cd PRD/precision-recall-distributions

# MNIST
python prd_from_image_folders.py \
    --dataset_name MNIST \
    --method_name AVAE \
    --eval_mode generation

# CIFAR10
python prd_from_image_folders.py \
    --dataset_name CIFAR10 \
    --method_name AVAE \
    --eval_mode generation
```

### For CelebA (Setup Required)

Set the environment variable pointing to your CelebA data:

```bash
# PowerShell
$env:CELEBA_DATA_DIR = "C:\path\to\your\celeba\data"
cd PRD\precision-recall-distributions
python prd_from_image_folders.py --dataset_name CelebA --method_name AVAE --eval_mode generation

# Or set it permanently (System Properties → Environment Variables)
```

The script expects `train_images_npy.npy` file in the specified directory.

### Command-line Arguments

- `--dataset_name`: Dataset name (MNIST, CelebA, CIFAR10, ImageNet)
- `--method_name`: Method name (default: AVAE)
- `--eval_mode`: Evaluation mode (generation, interpolation, reconstruction)
- `--num_clusters`: Number of cluster centers [default: 20]
- `--num_angles`: Number of angles for PRD curve [default: 1001]
- `--num_runs`: Number of runs to average [default: 10]
- `--inception_path`: Path to inception.pb [default: ./inception_models/inception.pb]
- `--cache_dir`: Cache directory [default: ./prd_cache/]
- `--silent`: Disable logging output

## Expected Results (from Paper)

### MNIST (latent_dim=16)
- Precision: 0.92 ± 0.02
- Recall: 0.98 ± 0.00

### CelebA (latent_dim=64)
- Precision: 0.88 ± 0.02
- Recall: 0.85 ± 0.02

### CIFAR10 (latent_dim=128)
- Precision: 0.72 ± 0.05
- Recall: 0.67 ± 0.04

## Current Results (in logs/)

Results match the paper closely (minor differences due to decimal rounding):

### MNIST
```
Precision: 0.916 ± 0.023
Recall: 0.980 ± 0.004
```
**Status**: ✅ Close match (diff: 0.004 precision, 0.000 recall)

### CelebA
```
Precision: 0.880 ± 0.017
Recall: 0.848 ± 0.016
```
**Status**: ✅ Close match (diff: 0.000 precision, 0.002 recall)

### CIFAR10
```
Precision: 0.724 ± 0.053
Recall: 0.673 ± 0.037
```
**Status**: ✅ Close match (diff: 0.004 precision, 0.003 recall)

## Output Format

Results are saved to `../../logs/{dataset}/{eval_mode}/prd_scores.txt`:

```
F_8   F_1/8     model
0.985 0.956     ../../../AVAE/eval/study_fid/generate_samples/logs/MNIST/generation/run_id_1
0.980 0.921     ../../../AVAE/eval/study_fid/generate_samples/logs/MNIST/generation/run_id_2
0.984 0.901     ../../../AVAE/eval/study_fid/generate_samples/logs/MNIST/generation/run_id_3
0.975 0.914     ../../../AVAE/eval/study_fid/generate_samples/logs/MNIST/generation/run_id_4
0.976 0.887     ../../../AVAE/eval/study_fid/generate_samples/logs/MNIST/generation/run_id_5
Precision: 0.916 ± 0.023
Recall: 0.980 ± 0.004
```

**Note**: F_8 corresponds to **Recall** (F-beta score with beta=8), F_1/8 corresponds to **Precision** (F-beta score with beta=1/8).

## Prerequisites

### Generated Samples

The script expects generated samples at:
```
../../eval/study_fid/generate_samples/logs/{dataset}/generation/run_id_{1-5}/generated_images.npy
```

Generate these samples first using:
```bash
cd ../study_fid/generate_samples
python generate_samples.py --config_id {0,1,2} --seed 0
```

Where `config_id`: 0=MNIST, 1=CelebA, 2=CIFAR10

### Dataset Paths

**MNIST & CIFAR10**: Automatically downloaded via `tf.keras.datasets`

**CelebA**: Set environment variable before running:
```bash
# PowerShell
$env:CELEBA_DATA_DIR = "C:\path\to\celeba"

# Bash/Linux
export CELEBA_DATA_DIR="/path/to/celeba"
```

Expected file: `{CELEBA_DATA_DIR}/train_images_npy.npy`

**ImageNet** (if using): Set `IMAGENET_DATA_DIR` similarly.

## Technical Details

### Feature Extraction

- **Model**: InceptionV3 pretrained on ImageNet
- **Layer**: pool_3:0 (2048-dimensional features)
- **Input size**: 299×299×3
- **Preprocessing**: Images are resized and normalized to [0, 255]
- **MNIST handling**: Grayscale images padded to 32×32 and converted to RGB

### Precision-Recall Computation

1. **Clustering**: Fits 20 Gaussian mixture model clusters to real data embeddings
2. **Manifold**: Defines manifolds M_real and M_gen based on cluster assignments
3. **PRD Curve**: Computes precision-recall pairs over 1001 interpolation angles
4. **F-beta scores**: 
   - F_8 emphasizes recall (β=8)
   - F_1/8 emphasizes precision (β=1/8)

### Memory Requirements

- InceptionV3 model: ~100MB
- Feature storage: ~80MB per 10K samples (2048-dim × 10K × 4 bytes)
- Total: ~200-300MB for typical evaluation

### Differences from Original Script

The `prd_from_image_folders.py` has been customized for AVAE:

1. **Argument changes**:
   - Removed: `--reference_dir`, `--eval_dirs`, `--eval_labels`
   - Added: `--dataset_name`, `--method_name`, `--eval_mode`

2. **Automatic run handling**: Processes all 5 runs automatically

3. **Real data loading**: Uses `get_real_data.py` instead of loading from directory

4. **Results aggregation**: Computes mean ± std across all runs

5. **Path flexibility**: Uses environment variables instead of hardcoded paths

6. **Image loading**: Uses `imageio` instead of `cv2` (for compatibility)

## Troubleshooting

### TensorFlow not found

```bash
pip install tensorflow==2.8.0
# or for GPU support:
pip install tensorflow-gpu==2.8.0
```

### CelebA environment variable not set

Error message:
```
ValueError: CelebA dataset path not configured. Set CELEBA_DATA_DIR environment variable.
```

**Solution**: Set the environment variable before running:
```bash
$env:CELEBA_DATA_DIR = "C:\your\path\to\celeba"
```

### Generated samples not found

Error: `FileNotFoundError` for `generated_images.npy`

**Solution**: Generate samples first:
```bash
cd ../../study_fid/generate_samples
python generate_samples.py --config_id 0  # for MNIST
```

### Inception model not found

Error: `FileNotFoundError` for `inception.pb`

**Solution**: The model should be in `inception_models/inception.pb`. If missing, download from:
- https://people.tuebingen.mpg.de/msajjadi/inception.pb

Or use:
```bash
wget https://people.tuebingen.mpg.de/msajjadi/inception.pb -P inception_models/
```

### Out of memory

If you get OOM errors, try:
1. Reduce batch size in `inception.py`
2. Use CPU instead of GPU (slower but uses less memory)
3. Reduce number of samples (modify `prd_samples=10000` in `get_real_data.py`)

### TensorFlow 1.x vs 2.x compatibility

The original code uses TensorFlow 1.x API (`tf.Session()`). For TensorFlow 2.x:

```python
# In get_real_data.py, replace:
imgs_train[:, 2:30, 2:30, :] = tf.Session().run(
    tf.image.grayscale_to_rgb(tf.convert_to_tensor(ori_imgs_train), name=None))

# With:
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```

Or use TensorFlow 1.x directly:
```bash
pip install tensorflow==1.15.5
```

## References

1. Sajjadi, M. S., Bachem, O., Lucic, M., Bousquet, O., & Gelly, S. (2018). Assessing Generative Models via Precision and Recall. *NeurIPS 2018*.
2. Official implementation: https://github.com/msmsajjadi/precision-recall-distributions
3. NeurIPS 2018 poster: https://people.tuebingen.mpg.de/msajjadi/prd_poster.pdf

## Citation

If you use these evaluation scripts, please cite both the PRD paper and the AVAE paper:

```bibtex
@inproceedings{sajjadi2018precision,
  title={Assessing Generative Models via Precision and Recall},
  author={Sajjadi, Mehdi S. M. and Bachem, Olivier and Lucic, Mario and Bousquet, Olivier and Gelly, Sylvain},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2018}
}

@inproceedings{saha2024avae,
  title={Matching Aggregate Posteriors in the Variational Autoencoder},
  author={Saha, Surojit and Joshi, Sarang and Whitaker, Ross},
  booktitle={International Conference on Pattern Recognition (ICPR)},
  year={2024}
}
```

