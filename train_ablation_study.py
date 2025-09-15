import argparse
import os
import sys
import json
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm

from config import (
    create_full_spectrumnet_config,
    create_roberta_baseline_config,
    create_roberta_han_config,
    create_roberta_gru_config,
    create_roberta_han_gru_no_fusion_config
)
from train_hierarchical import train_hierarchical_single_fold


def get_ablation_configurations():
    return {
        'full_spectrumnet': {
            'config_fn': create_full_spectrumnet_config,
            'description': 'Full SpectrumNet: RoBERTa + HAN + GRU + Dynamic Fusion'
        },
        'roberta_baseline': {
            'config_fn': create_roberta_baseline_config,
            'description': 'RoBERTa Baseline: RoBERTa encoder with mean pooling only'
        },
        'roberta_han': {
            'config_fn': create_roberta_han_config,
            'description': 'RoBERTa + HAN: RoBERTa + Hierarchical Attention'
        },
        'roberta_gru': {
            'config_fn': create_roberta_gru_config,
            'description': 'RoBERTa + GRU: RoBERTa + Previous Comments GRU'
        },
        'roberta_han_gru_no_fusion': {
            'config_fn': create_roberta_han_gru_no_fusion_config,
            'description': 'RoBERTa + HAN + GRU (No Fusion): RoBERTa + HAN + GRU with simple mean aggregation'
        }
    }


def run_single_configuration(config_name, config_info, folds, embedding_file, base_output_dir):
    print(f"\n{'='*80}")
    print(f"RUNNING CONFIGURATION: {config_name.upper()}")
    print(f"Description: {config_info['description']}")
    print(f"{'='*80}")

    config = config_info['config_fn']()
    config_output_dir = os.path.join(base_output_dir, config_name)
    os.makedirs(config_output_dir, exist_ok=True)

    successful_folds = []
    failed_folds = []

    for fold in folds:
        try:
            result = train_hierarchical_single_fold(
                config=config,
                fold_num=fold,
                embedding_file=embedding_file,
                output_dir=config_output_dir
            )
            if result:
                successful_folds.append(fold)
                print(f"Fold {fold} completed successfully")
            else:
                failed_folds.append(fold)
                print(f"Fold {fold} failed")
        except Exception as e:
            print(f"Fold {fold} failed with error: {e}")
            failed_folds.append(fold)

    config_summary = {
        'configuration_name': config_name,
        'description': config_info['description'],
        'successful_folds': successful_folds,
        'failed_folds': failed_folds,
        'total_folds': len(successful_folds) + len(failed_folds),
        'success_rate': len(successful_folds) / (len(successful_folds) + len(failed_folds)) if (len(successful_folds) + len(failed_folds)) > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(config_output_dir, 'configuration_summary.json'), 'w') as f:
        json.dump(config_summary, f, indent=2)

    print(f"\n{config_name.upper()} COMPLETED:")
    print(f"  Successful folds: {successful_folds}")
    print(f"  Failed folds: {failed_folds}")
    print(f"  Success rate: {config_summary['success_rate']:.2%}")

    return successful_folds, failed_folds


def aggregate_results(base_output_dir, configurations):
    all_results = []
    config_summaries = {}

    for config_name in configurations:
        config_dir = os.path.join(base_output_dir, config_name)
        summary_file = os.path.join(config_dir, 'configuration_summary.json')

        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                config_summaries[config_name] = json.load(f)

            for fold in config_summaries[config_name].get('successful_folds', []):
                fold_summary_file = os.path.join(config_dir, f'fold_{fold}_summary.json')
                if os.path.exists(fold_summary_file):
                    with open(fold_summary_file, 'r') as f:
                        fold_data = json.load(f)
                        fold_data['configuration'] = config_name
                        fold_data['fold'] = fold
                        all_results.append(fold_data)

    if all_results:
        comparison_df = pd.DataFrame(all_results)
        comparison_df.to_csv(os.path.join(base_output_dir, 'ablation_comparison.csv'), index=False)

        summary_stats = comparison_df.groupby('configuration').agg({
            'f1_macro': ['mean', 'std'],
            'accuracy': ['mean', 'std'],
            'precision_macro': ['mean', 'std'],
            'recall_macro': ['mean', 'std']
        }).round(4)

        summary_stats.to_csv(os.path.join(base_output_dir, 'ablation_summary_stats.csv'))

    study_summary = {
        'configurations_tested': list(configurations),
        'total_configurations': len(configurations),
        'configuration_summaries': config_summaries,
        'timestamp': datetime.now().isoformat(),
        'total_results_collected': len(all_results)
    }

    with open(os.path.join(base_output_dir, 'ablation_study_summary.json'), 'w') as f:
        json.dump(study_summary, f, indent=2)

    print(f"  Configurations: {len(configurations)}")
    print(f"  Total results: {len(all_results)}")


def main():
    parser = argparse.ArgumentParser(description='Run SpectrumNet Ablation Study')
    parser.add_argument('--embedding_file', required=True, help='Path to precomputed embeddings H5 file')
    parser.add_argument('--output_dir', default='ablation_results', help='Output directory for results')
    parser.add_argument('--folds', default='1,2,3,4,5,6,7,8,9,10', help='Comma-separated list of folds to run')
    parser.add_argument('--configs', default='all', help='Comma-separated list of configs or "all"')

    args = parser.parse_args()

    if not os.path.exists(args.embedding_file):
        print(f"Error: Embedding file not found: {args.embedding_file}")
        print("Please run precompute_embeddings_hierarchical.py first")
        sys.exit(1)

    folds = [int(f.strip()) for f in args.folds.split(',')]
    all_configurations = get_ablation_configurations()

    if args.configs == 'all':
        selected_configs = all_configurations
    else:
        config_names = [c.strip() for c in args.configs.split(',')]
        selected_configs = {name: all_configurations[name] for name in config_names if name in all_configurations}

        if len(selected_configs) != len(config_names):
            missing = set(config_names) - set(selected_configs.keys())
            print(f"Error: Unknown configurations: {missing}")
            print(f"Available configurations: {list(all_configurations.keys())}")
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*80)
    print("SPECTRUMNET ABLATION STUDY")
    print("="*80)
    print(f"Configurations: {list(selected_configs.keys())}")
    print(f"Folds: {folds}")
    print(f"Output directory: {args.output_dir}")
    print(f"Embedding file: {args.embedding_file}")
    print("="*80)

    all_successful = {}
    all_failed = {}

    for config_name, config_info in selected_configs.items():
        successful, failed = run_single_configuration(
            config_name, config_info, folds, args.embedding_file, args.output_dir
        )
        all_successful[config_name] = successful
        all_failed[config_name] = failed

    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    for config_name in selected_configs:
        success_rate = len(all_successful[config_name]) / len(folds) if folds else 0
        print(f"{config_name:30s}: {len(all_successful[config_name]):2d}/{len(folds)} folds successful ({success_rate:.1%})")

    aggregate_results(args.output_dir, selected_configs.keys())

    print(f"\nAll results saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()