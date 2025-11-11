#!/usr/bin/env python
"""
Global Normalization for Distributed Hyperparameter Optimization

This script performs global normalization across all methods after distributed
optimization has completed. It should be run AFTER all hyperopt.py runs finish.

Usage:
    python normalize_results.py --dataset swiss_roll --data-type continuous
    python normalize_results.py --dataset mnist --data-type categorical
"""

import numpy as np
import pandas as pd
import os
import json
import glob
import argparse
from pathlib import Path

# Import functions from hyperopt
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hyperopt import (
    normalize_metrics_globally,
    calculate_composite_score,
    find_best_config,
    clean_for_json,
    create_heatmap,
    create_composite_score_heatmap,
    create_mds_comparison,
    HYPERPARAM_DIR
)


def load_all_results(dataset_name, data_type='continuous'):
    """
    Load all optimization results for a dataset from JSON files.

    Args:
        dataset_name: Name of the dataset
        data_type: 'continuous' or 'categorical'

    Returns:
        all_results: Dict of {method_name: [results]}
    """
    base_dir = os.path.join(HYPERPARAM_DIR, data_type, dataset_name)

    if not os.path.exists(base_dir):
        print(f"✗ No results found at: {base_dir}")
        return {}

    print(f"Loading results from: {base_dir}")

    all_results = {}
    methods = ['tsne', 'umap', 'phate', 'pacmap', 'mds', 'autoencoder']

    for method in methods:
        method_dir = os.path.join(base_dir, method)

        if not os.path.exists(method_dir):
            continue

        # Find all result JSON files
        result_files = glob.glob(os.path.join(method_dir, 'result_*.json'))

        if len(result_files) == 0:
            continue

        print(f"  {method}: Found {len(result_files)} result files")

        method_results = []
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)

                # Load embedding if needed (for top 5 later)
                embedding_file = result.get('embedding_file')
                if embedding_file and os.path.exists(embedding_file):
                    embedding_data = np.load(embedding_file)
                    result['embedding'] = embedding_data['embedding']

                method_results.append(result)

            except Exception as e:
                print(f"    ⚠️  Error loading {result_file}: {e}")
                continue

        if len(method_results) > 0:
            all_results[method] = method_results
            print(f"    ✓ Loaded {len(method_results)} configs")

    return all_results


def save_normalized_results(all_results, dataset_name, data_type='continuous'):
    """
    Save normalized results and generate visualizations.

    Args:
        all_results: Dict of {method_name: [results]} with normalized_metrics
        dataset_name: Name of the dataset
        data_type: 'continuous' or 'categorical'
    """
    print(f"\n{'='*80}")
    print("SAVING NORMALIZED RESULTS AND VISUALIZATIONS")
    print(f"{'='*80}")

    # Create visualizations
    if 'tsne' in all_results:
        create_heatmap(all_results['tsne'], 'tsne', dataset_name,
                     'perplexity', 'max_iter', 'T&C_AUC', data_type)
        create_composite_score_heatmap(all_results['tsne'], 'tsne', dataset_name,
                                      'perplexity', 'max_iter', data_type)
        print("  ✓ Created t-SNE heatmaps")

    if 'umap' in all_results:
        create_heatmap(all_results['umap'], 'umap', dataset_name,
                     'n_neighbors', 'min_dist', 'T&C_AUC', data_type)
        create_composite_score_heatmap(all_results['umap'], 'umap', dataset_name,
                                      'n_neighbors', 'min_dist', data_type)
        print("  ✓ Created UMAP heatmaps")

    if 'pacmap' in all_results:
        create_heatmap(all_results['pacmap'], 'pacmap', dataset_name,
                     'MN_ratio', 'FP_ratio', 'T&C_AUC', data_type)
        create_composite_score_heatmap(all_results['pacmap'], 'pacmap', dataset_name,
                                      'MN_ratio', 'FP_ratio', data_type)
        print("  ✓ Created PaCMAP heatmaps")

    if 'mds' in all_results:
        create_mds_comparison(all_results['mds'], dataset_name, data_type)
        print("  ✓ Created MDS comparison")

    if 'phate' in all_results:
        create_heatmap(all_results['phate'], 'phate', dataset_name,
                     'knn', 'decay', 'T&C_AUC', data_type)
        create_composite_score_heatmap(all_results['phate'], 'phate', dataset_name,
                                      'knn', 'decay', data_type)
        print("  ✓ Created PHATE heatmaps")

    # Save summary
    summary_file = os.path.join(HYPERPARAM_DIR, data_type, dataset_name, 'optimization_summary.json')
    summary = {}

    # Save top 5 embeddings for each method
    for method, results in all_results.items():
        if len(results) > 0:
            # Find best config
            best_result = find_best_config(results, data_type=data_type)

            if best_result is None:
                print(f"  ✗ {method}: Could not find best config")
                continue

            results_sorted = sorted(results, key=lambda x: x.get('composite_score', 0), reverse=True)

            # Save embeddings for top 5
            output_dir = os.path.join(HYPERPARAM_DIR, data_type, dataset_name, method)
            for idx, r in enumerate(results_sorted[:5]):
                if 'embedding' in r:
                    param_str = '_'.join([f"{k}={v}" for k, v in sorted(r['params'].items())])
                    param_str = param_str.replace('/', '-').replace('.', 'p').replace('[', '').replace(']', '').replace(',', '_')[:100]
                    embedding_file = os.path.join(output_dir, f"embedding_top{idx+1}_{param_str}.npz")
                    np.savez_compressed(embedding_file, embedding=r['embedding'], params=r['params'])
                    r['embedding_file'] = embedding_file
                    del r['embedding']

            top_5 = results_sorted[:5]

            summary[method] = clean_for_json({
                'num_configs': len(results),
                'best_config': {
                    'params': best_result['params'],
                    'metrics': best_result['metrics'],
                    'normalized_metrics': best_result['normalized_metrics'],
                    'composite_score': best_result['composite_score'],
                    'runtime': best_result['runtime']
                },
                'top_5_configs': [
                    {
                        'rank': idx + 1,
                        'params': r['params'],
                        'composite_score': r.get('composite_score', 0),
                        'metrics': r['metrics'],
                        'normalized_metrics': r.get('normalized_metrics', {})
                    }
                    for idx, r in enumerate(top_5)
                ]
            })

            print(f"\n  ✓ {method}: {len(results)} configs")
            print(f"      Best composite score: {best_result['composite_score']:.4f}")
            print(f"      Best params: {best_result['params']}")
            print(f"      Metrics: T&C={best_result['metrics']['T&C_AUC']:.3f}, "
                  f"ND={best_result['metrics']['Neighbor_Dissimilarity']:.3f}, "
                  f"Stress={best_result['metrics']['Stress']:.3f}")
            if data_type == 'categorical':
                dbi_val = best_result['metrics'].get('DBI')
                nmi_val = best_result['metrics'].get('NMI')
                dbi_str = f"{dbi_val:.3f}" if dbi_val is not None else "N/A"
                nmi_str = f"{nmi_val:.3f}" if nmi_val is not None else "N/A"
                print(f"               DBI={dbi_str}, NMI={nmi_str}")

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n✓ Summary saved: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Perform global normalization on distributed optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normalize results for continuous dataset:
  python normalize_results.py --dataset swiss_roll --data-type continuous

  # Normalize results for categorical dataset:
  python normalize_results.py --dataset mnist --data-type categorical

This script should be run AFTER all hyperopt.py runs complete in distributed mode.
        """
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., swiss_roll, mnist)')
    parser.add_argument('--data-type', type=str, default='continuous',
                       choices=['continuous', 'categorical'],
                       help='Type of dataset')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("GLOBAL NORMALIZATION - POST-PROCESSING")
    print(f"{'='*80}")
    print(f"Dataset: {args.dataset}")
    print(f"Data type: {args.data_type}")
    print(f"{'='*80}\n")

    # Load all results
    print("STEP 1: Loading Results")
    print(f"{'='*80}")
    all_results = load_all_results(args.dataset, args.data_type)

    if len(all_results) == 0:
        print(f"\n✗ No results found for {args.dataset} ({args.data_type})")
        print(f"   Make sure hyperopt.py has been run with --skip-normalization")
        return

    total_configs = sum(len(v) for v in all_results.values())
    print(f"\n✓ Loaded {len(all_results)} methods, {total_configs} total configs")

    # Global normalization
    print(f"\n{'='*80}")
    print("STEP 2: Global Normalization")
    print(f"{'='*80}")
    all_results = normalize_metrics_globally(all_results, args.data_type)

    # Calculate composite scores
    for method, results in all_results.items():
        for result in results:
            result['composite_score'] = calculate_composite_score(result, data_type=args.data_type)

    print(f"\n✓ Global normalization complete")

    # Save results and create visualizations
    print(f"\n{'='*80}")
    print("STEP 3: Saving and Visualizing")
    print(f"{'='*80}")
    save_normalized_results(all_results, args.dataset, args.data_type)

    print(f"\n{'='*80}")
    print("NORMALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults directory: {os.path.join(HYPERPARAM_DIR, args.data_type, args.dataset)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
