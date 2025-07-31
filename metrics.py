"""
Evaluation metrics for conformal prediction
"""

import numpy as np
from typing import Tuple, List


def calculate_coverage(pred_mask: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Calculate coverage for each sample
    
    Args:
        pred_mask: Binary prediction mask (N, D)
        true_labels: Binary true labels (N, D)
        
    Returns:
        Coverage values for each sample
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        coverage = np.sum(pred_mask & true_labels, axis=1) / np.sum(true_labels, axis=1)
    
    # Filter out NaN values (samples with no positive labels)
    valid_coverage = coverage[~np.isnan(coverage)]
    return valid_coverage


def calculate_precision(pred_mask: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """
    Calculate precision for each sample
    
    Args:
        pred_mask: Binary prediction mask (N, D)
        true_labels: Binary true labels (N, D)
        
    Returns:
        Precision values for each sample
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.sum(pred_mask & true_labels, axis=1) / np.sum(pred_mask, axis=1)
    
    # Filter out NaN values (samples with no predictions)
    valid_precision = precision[~np.isnan(precision)]
    return valid_precision


def calculate_set_size(pred_mask: np.ndarray) -> np.ndarray:
    """
    Calculate relative prediction set size
    
    Args:
        pred_mask: Binary prediction mask (N, D)
        
    Returns:
        Relative set sizes for each sample
    """
    total_pixels = pred_mask.shape[1]
    set_sizes = np.sum(pred_mask, axis=1) / total_pixels
    return set_sizes


def calculate_all_metrics(pred_mask: np.ndarray, true_labels: np.ndarray, 
                         target_coverage: float) -> dict:
    """
    Calculate all evaluation metrics
    
    Args:
        pred_mask: Binary prediction mask (N, D)
        true_labels: Binary true labels (N, D)
        target_coverage: Target coverage level
        
    Returns:
        Dictionary containing all metrics
    """
    coverage = calculate_coverage(pred_mask, true_labels)
    gap = np.abs(coverage - target_coverage)
    precision = calculate_precision(pred_mask, true_labels)
    size = calculate_set_size(pred_mask)
    
    return {
        'coverage': coverage.tolist(),
        'gap': gap.tolist(),
        'precision': precision.tolist(),
        'size': size.tolist()
    }


def summarize_metrics(metrics: dict) -> dict:
    """
    Compute summary statistics for metrics
    
    Args:
        metrics: Dictionary of metric lists
        
    Returns:
        Dictionary of summary statistics
    """
    summary = {}
    for metric_name, values in metrics.items():
        if len(values) > 0:
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_median'] = np.median(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
        else:
            summary[f'{metric_name}_mean'] = np.nan
            summary[f'{metric_name}_std'] = np.nan
            summary[f'{metric_name}_median'] = np.nan
            summary[f'{metric_name}_min'] = np.nan
            summary[f'{metric_name}_max'] = np.nan
    
    return summary