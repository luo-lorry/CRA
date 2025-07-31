"""
Basic usage example for Adaptive Conformal Prediction
"""

import numpy as np
import torch
from src.experiments.aa_cra import run_aa_cra_experiment
from src.experiments.aa_crc import run_aa_crc_experiment
from src.experiments.traditional_methods import run_traditional_crc
from src.utils.metrics import calculate_all_metrics


def generate_synthetic_data(n_samples=1000, image_size=64):
    """Generate synthetic medical segmentation data for demonstration"""
    
    # Generate synthetic prediction probabilities
    phat = np.random.beta(2, 5, size=(n_samples, image_size, image_size))
    
    # Generate synthetic ground truth (correlated with predictions)
    labels = (phat + np.random.normal(0, 0.1, phat.shape)) > 0.3
    labels = labels.astype(bool)
    
    # Generate synthetic RGB images
    images = np.random.rand(n_samples, 3, 224, 224).astype(np.float32)
    
    return {
        'phat_raw': phat.astype(np.float32),
        'phat_calibrated': phat.astype(np.float32),  # Same as raw for demo
        'labels': labels,
        'images': images
    }


def main():
    """Demonstrate basic usage of adaptive conformal prediction"""
    
    print("Adaptive Conformal Prediction - Basic Usage Example")
    print("=" * 55)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data = generate_synthetic_data(n_samples=500)
    n_samples = len(data['labels'])
    
    # Split data
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    n_test = int(0.4 * n_samples)
    n_train = int(0.3 * (n_samples - n_test))
    n_calib = n_samples - n_test - n_train
    
    train_idx = indices[:n_train]
    calib_idx = indices[n_train:n_train + n_calib]
    test_idx = indices[n_train + n_calib:]
    
    print(f"Data split: {len(train_idx)} train, {len(calib_idx)} calib, {len(test_idx)} test")
    
    # Set target coverage
    alpha = 0.1  # 90% target coverage
    target_coverage = 1 - alpha
    
    print(f"\nTarget coverage: {target_coverage:.1%}")
    print("-" * 30)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run AA-CRA (our proposed method)
    print("\n1. Running AA-CRA...")
    try:
        aa_cra_results = run_aa_cra_experiment(
            data, train_idx, calib_idx, test_idx, alpha, device
        )
        
        if aa_cra_results:
            aa_cra_coverage = np.mean(aa_cra_results['coverage'])
            aa_cra_gap = np.mean(aa_cra_results['gap'])
            aa_cra_precision = np.mean(aa_cra_results['precision'])
            aa_cra_size = np.mean(aa_cra_results['size'])
            
            print(f"   Coverage: {aa_cra_coverage:.3f} (Gap: {aa_cra_gap:.3f})")
            print(f"   Precision: {aa_cra_precision:.3f}")
            print(f"   Avg Set Size: {aa_cra_size:.3f}")
        else:
            print("   Failed to run AA-CRA")
            aa_cra_results = None
    except Exception as e:
        print(f"   Error in AA-CRA: {e}")
        aa_cra_results = None
    
    # Run AA-CRC
    print("\n2. Running AA-CRC...")
    try:
        aa_crc_results = run_aa_crc_experiment(
            data, train_idx, calib_idx, test_idx, alpha, device
        )
        
        if aa_crc_results:
            aa_crc_coverage = np.mean(aa_crc_results['coverage'])
            aa_crc_gap = np.mean(aa_crc_results['gap'])
            aa_crc_precision = np.mean(aa_crc_results['precision'])
            aa_crc_size = np.mean(aa_crc_results['size'])
            
            print(f"   Coverage: {aa_crc_coverage:.3f} (Gap: {aa_crc_gap:.3f})")
            print(f"   Precision: {aa_crc_precision:.3f}")
            print(f"   Avg Set Size: {aa_crc_size:.3f}")
        else:
            print("   Failed to run AA-CRC")
            aa_crc_results = None
    except Exception as e:
        print(f"   Error in AA-CRC: {e}")
        aa_crc_results = None
    
    # Run traditional CRC for comparison
    print("\n3. Running Traditional CRC...")
    try:
        crc_results = run_traditional_crc(
            data, calib_idx, test_idx, alpha
        )
        
        if crc_results:
            crc_coverage = np.mean(crc_results['coverage'])
            crc_gap = np.mean(crc_results['gap'])
            crc_precision = np.mean(crc_results['precision'])
            crc_size = np.mean(crc_results['size'])
            
            print(f"   Coverage: {crc_coverage:.3f} (Gap: {crc_gap:.3f})")
            print(f"   Precision: {crc_precision:.3f}")
            print(f"   Avg Set Size: {crc_size:.3f}")
        else:
            print("   Failed to run CRC")
            crc_results = None
    except Exception as e:
        print(f"   Error in CRC: {e}")
        crc_results = None
    
    # Compare results
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"{'Method':<15} {'Coverage':<10} {'Gap':<8} {'Precision':<10} {'Size':<8}")
    print("-" * 50)
    
    if aa_cra_results:
        print(f"{'AA-CRA':<15} {aa_cra_coverage:<10.3f} {aa_cra_gap:<8.3f} {aa_cra_precision:<10.3f} {aa_cra_size:<8.3f}")
    
    if aa_crc_results:
        print(f"{'AA-CRC':<15} {aa_crc_coverage:<10.3f} {aa_crc_gap:<8.3f} {aa_crc_precision:<10.3f} {aa_crc_size:<8.3f}")
    
    if crc_results:
        print(f"{'CRC':<15} {crc_coverage:<10.3f} {crc_gap:<8.3f} {crc_precision:<10.3f} {crc_size:<8.3f}")
    
    print(f"\nTarget Coverage: {target_coverage:.3f}")
    print("\nNote: This is a demonstration with synthetic data.")
    print("Real performance should be evaluated on actual medical imaging datasets.")


if __name__ == "__main__":
    main()