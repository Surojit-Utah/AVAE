"""
Master script to run all AVAE evaluations.

This script runs the complete evaluation pipeline:
1. Generate samples from trained models
2. Compute FID scores
3. Compute Precision-Recall scores
4. Compute Entropy scores
5. Compute MSE (reconstruction error)

Usage:
    # Run all evaluations for MNIST
    python run_all_evaluations.py --dataset MNIST
    
    # Run all evaluations for all datasets
    python run_all_evaluations.py --dataset all
    
    # Run specific evaluation only
    python run_all_evaluations.py --dataset CIFAR10 --eval_type fid
"""

import os
import sys
import argparse
import subprocess
import time


# Dataset configurations
DATASET_CONFIG = {
    'MNIST': {
        'config_id': 0,
        'latent_dim': 16,
        'num_filter': 64,
    },
    'CelebA': {
        'config_id': 1,
        'latent_dim': 64,
        'num_filter': 64,
    },
    'CIFAR10': {
        'config_id': 2,
        'latent_dim': 128,
        'num_filter': 128,
    }
}


def run_command(cmd, description, cwd=None):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {description} failed after {elapsed:.1f}s")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Script not found for: {description}")
        return False


def generate_samples(dataset, config_id, seed=0):
    """Step 1: Generate samples from trained models."""
    script_dir = os.path.join('study_fid', 'generate_samples')
    cmd = [
        sys.executable,
        'generate_samples.py',
        '--config_id', str(config_id),
        '--seed', str(seed)
    ]
    return run_command(
        cmd,
        f"Generating samples for {dataset}",
        cwd=script_dir
    )


def compute_fid(dataset, config_id, seed=0):
    """Step 2: Compute FID scores."""
    script_dir = os.path.join('study_fid', 'compute_fid')
    cmd = [
        sys.executable,
        'compute_n_plot_fid.py',
        '--config_id', str(config_id),
        '--gen_type', 'generation',
        '--seed', str(seed)
    ]
    return run_command(
        cmd,
        f"Computing FID for {dataset}",
        cwd=script_dir
    )


def compute_prd(dataset, seed=0):
    """Step 3: Compute Precision-Recall scores."""
    script_dir = 'study_prd'
    cmd = [
        sys.executable,
        'compute_prd.py',
        '--dataset', dataset,
        '--gen_type', 'generation',
        '--seed', str(seed)
    ]
    return run_command(
        cmd,
        f"Computing Precision-Recall for {dataset}",
        cwd=script_dir
    )


def compute_entropy(dataset, config_id, seed=0):
    """Step 4: Compute Entropy scores."""
    script_dir = 'entropy'
    cmd = [
        sys.executable,
        'compute_entropy.py',
        '--config_id', str(config_id),
        '--seed', str(seed)
    ]
    return run_command(
        cmd,
        f"Computing Entropy for {dataset}",
        cwd=script_dir
    )


def compute_mse(dataset, config_id, seed=0):
    """Step 5: Compute MSE (reconstruction error)."""
    script_dir = 'MSE'
    cmd = [
        sys.executable,
        'compute_mse.py',
        '--config_id', str(config_id),
        '--seed', str(seed)
    ]
    return run_command(
        cmd,
        f"Computing MSE for {dataset}",
        cwd=script_dir
    )


def print_summary(results):
    """Print summary of all evaluations."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    for dataset, evals in results.items():
        print(f"\n{dataset}:")
        for eval_name, success in evals.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {eval_name:20s}: {status}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run complete AVAE evaluation pipeline"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['MNIST', 'CelebA', 'CIFAR10', 'all'],
        help='Dataset to evaluate (or "all" for all datasets)'
    )
    parser.add_argument(
        '--eval_type',
        type=str,
        default='all',
        choices=['all', 'samples', 'fid', 'prd', 'entropy', 'mse'],
        help='Type of evaluation to run'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed'
    )
    parser.add_argument(
        '--skip_samples',
        action='store_true',
        help='Skip sample generation (if samples already exist)'
    )
    args = parser.parse_args()
    
    # Determine which datasets to evaluate
    if args.dataset == 'all':
        datasets = ['MNIST', 'CelebA', 'CIFAR10']
    else:
        datasets = [args.dataset]
    
    # Store results
    results = {}
    
    # Run evaluations for each dataset
    for dataset in datasets:
        print(f"\n{'#'*70}")
        print(f"# EVALUATING {dataset}")
        print(f"{'#'*70}\n")
        
        config = DATASET_CONFIG[dataset]
        config_id = config['config_id']
        
        results[dataset] = {}
        
        # Step 1: Generate samples (unless skipped)
        if not args.skip_samples and args.eval_type in ['all', 'samples']:
            results[dataset]['Sample Generation'] = generate_samples(
                dataset, config_id, args.seed
            )
        else:
            print(f"\n⏭️  Skipping sample generation for {dataset}")
            results[dataset]['Sample Generation'] = None
        
        # Step 2: FID
        if args.eval_type in ['all', 'fid']:
            results[dataset]['FID'] = compute_fid(
                dataset, config_id, args.seed
            )
        
        # Step 3: Precision-Recall
        if args.eval_type in ['all', 'prd']:
            results[dataset]['Precision-Recall'] = compute_prd(
                dataset, args.seed
            )
        
        # Step 4: Entropy
        if args.eval_type in ['all', 'entropy']:
            results[dataset]['Entropy'] = compute_entropy(
                dataset, config_id, args.seed
            )
        
        # Step 5: MSE
        if args.eval_type in ['all', 'mse']:
            results[dataset]['MSE'] = compute_mse(
                dataset, config_id, args.seed
            )
    
    # Print summary
    print_summary(results)
    
    print("\n✅ All evaluations completed!")
    print("\nResults are saved in the respective logs/ directories:")
    print("  - FID:      eval/study_fid/compute_fid/logs/{dataset}/generation/")
    print("  - PRD:      eval/study_prd/logs/{dataset}/generation/")
    print("  - Entropy:  eval/entropy/logs/{dataset}/")
    print("  - MSE:      eval/MSE/logs/{dataset}/")
    print("\nSee eval/EVAL_VERIFICATION.md for comparison with paper results.\n")


if __name__ == "__main__":
    main()
