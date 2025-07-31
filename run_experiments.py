#!/usr/bin/env python3
"""
Main script to run adaptive conformal prediction experiments
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiments.aa_crc import run_aa_crc_experiment
from experiments.aa_cra import run_aa_cra_experiment
from experiments.traditional_methods import run_all_traditional_methods
from utils.data_utils import load_dataset
from utils.visualization import create_all_plots
from utils.metrics import summarize_metrics


def set_seeds(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run Adaptive Conformal Prediction Experiments'
    )
    
    parser.add_argument(
        '--input_folder', 
        type=str, 
        default='data/POLYPS',
        help='Path to input data folder'
    )
    
    parser.add_argument(
        '--alpha_values',
        nargs='+',
        type=float,
        default=[0.05, 0.1, 0.2],
        help='Alpha values to test (space separated)'
    )
    
    parser.add_argument(
        '--n_repeats',
        type=int,
        default=10,
        help='Number of experimental repeats'
    )
    
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.5,
        help='Fraction of data for testing'
    )
    
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.3,
        help='Fraction of non-test data for training (rest for calibration)'
    )
    
    parser.add_argument(
        '--output_folder',
        type=str,
        default=None,
        help='Output folder (default: input_folder/AAAI26)'
    )
    
    parser.add_argument(
        '--methods',
        nargs='+',
        default=['all'],
        help='Methods to run (default: all)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (auto, cpu, cuda)'
    )
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup computation device"""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def main():
    """Main experimental pipeline"""
    args = parse_arguments()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Setup output folder
    if args.output_folder is None:
        output_folder = os.path.join(args.input_folder, 'AAAI26')
    else:
        output_folder = args.output_folder
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input_folder}...")
    data = load_dataset(args.input_folder)
    
    # Run experiments
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for alpha in args.alpha_values:
        print(f"\n{'='*60}")
        print(f"Running experiments with α={alpha:.2f}")
        print(f"{'='*60}")
        
        alpha_results = {}
        
        # Run multiple repeats
        for repeat in range(args.n_repeats):
            print(f"\nRepeat {repeat + 1}/{args.n_repeats}")
            
            # Split data
            indices = np.random.permutation(len(data['labels']))
            n_test = int(len(indices) * args.test_ratio)
            n_train = int((len(indices) - n_test) * args.train_ratio)
            
            train_idx = indices[:n_train]
            calib_idx = indices[n_train:n_train + (len(indices) - n_test - n_train)]
            test_idx = indices[-n_test:]
            
            # Run methods
            if 'all' in args.methods or 'AA-CRC' in args.methods:
                print("Running AA-CRC...")
                result = run_aa_crc_experiment(
                    data, train_idx, calib_idx, test_idx, alpha, device
                )
                if result:
                    if 'AA-CRC' not in alpha_results:
                        alpha_results['AA-CRC'] = {k: [] for k in result.keys()}
                    for k, v in result.items():
                        alpha_results['AA-CRC'][k].extend(v)
            
            if 'all' in args.methods or 'AA-CRA' in args.methods:
                print("Running AA-CRA...")
                result = run_aa_cra_experiment(
                    data, train_idx, calib_idx, test_idx, alpha, device
                )
                if result:
                    if 'AA-CRA' not in alpha_results:
                        alpha_results['AA-CRA'] = {k: [] for k in result.keys()}
                    for k, v in result.items():
                        alpha_results['AA-CRA'][k].extend(v)
            
            # Run traditional methods
            traditional_results = run_all_traditional_methods(
                data, calib_idx, test_idx, alpha
            )
            
            for method_name, result in traditional_results.items():
                if result:
                    if method_name not in alpha_results:
                        alpha_results[method_name] = {k: [] for k in result.keys()}
                    for k, v in result.items():
                        alpha_results[method_name][k].extend(v)
        
        all_results[f"alpha_{int(alpha * 100)}"] = alpha_results
        
        # Generate plots for this alpha
        create_all_plots(alpha_results, alpha, output_folder)
        
        # Save results
        results_file = os.path.join(
            output_folder, 
            f'results_alpha_{int(alpha * 100)}_{timestamp}.npz'
        )
        
        save_data = {'alpha': alpha}
        for method in alpha_results:
            for metric in alpha_results[method]:
                save_data[f'{method}_{metric}'] = np.array(alpha_results[method][metric])
        
        np.savez(results_file, **save_data)
        print(f"Results saved to {results_file}")
    
    # Generate summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for alpha in args.alpha_values:
        alpha_key = f"alpha_{int(alpha * 100)}"
        if alpha_key in all_results:
            print(f"\nα = {alpha:.2f} (Target Coverage: {1-alpha:.1%})")
            print("-" * 40)
            
            for method in all_results[alpha_key]:
                if len(all_results[alpha_key][method]['coverage']) > 0:
                    summary = summarize_metrics(all_results[alpha_key][method])
                    print(f"{method:25s} | "
                          f"Coverage: {summary['coverage_mean']:.3f}±{summary['coverage_std']:.3f} | "
                          f"Gap: {summary['gap_mean']:.3f}±{summary['gap_std']:.3f} | "
                          f"Precision: {summary['precision_mean']:.3f}±{summary['precision_std']:.3f}")
    
    print(f"\nResults saved to: {output_folder}")
    print("Experiment completed successfully!")


if __name__ == "__main__":
    main()