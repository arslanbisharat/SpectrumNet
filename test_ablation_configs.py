#!/usr/bin/env python3

from config import (
    create_full_spectrumnet_config,
    create_roberta_baseline_config,
    create_roberta_han_config,
    create_roberta_gru_config,
    create_roberta_han_gru_no_fusion_config
)


def test_config(config_name, config_fn):
    print(f"\n{'='*50}")
    print(f"Testing: {config_name}")
    print(f"{'='*50}")

    try:
        config = config_fn()

        print(f"  use_hierarchical_attention: {config.model.use_hierarchical_attention}")
        print(f"  use_previous_comments: {config.model.use_previous_comments}")
        print(f"  use_dynamic_attention: {config.model.use_dynamic_attention}")
        print(f"  use_embeddings: {config.model.use_embeddings}")
        print(f"  hidden_size: {config.model.hidden_size}")
        print(f"  num_classes: {config.model.num_classes}")
        print(f"  batch_size: {config.training.batch_size}")
        print(f"  learning_rate: {config.training.learning_rate}")
        print(f"  num_epochs: {config.training.num_epochs}")

        return True

    except Exception as e:
        print(f"Error creating config: {e}")
        return False


def main():
    configurations = [
        ("Full SpectrumNet", create_full_spectrumnet_config),
        ("RoBERTa Baseline", create_roberta_baseline_config),
        ("RoBERTa + HAN", create_roberta_han_config),
        ("RoBERTa + GRU", create_roberta_gru_config),
        ("RoBERTa + HAN + GRU (No Fusion)", create_roberta_han_gru_no_fusion_config),
    ]

    results = []
    for config_name, config_fn in configurations:
        success = test_config(config_name, config_fn)
        results.append((config_name, success))

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    all_passed = True
    for config_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{config_name:35s}: {status}")
        if not success:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL CONFIGURATIONS PASSED")
        print("Ready to run ablation study!")
    else:
        print("SOME CONFIGURATIONS FAILED")
        print("Please fix the failed configurations before running the ablation study.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())