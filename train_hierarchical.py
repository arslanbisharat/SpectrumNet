import argparse
import os
import time
import psutil
from config import Config
from dataset_hierarchical import HierarchicalLGBTQDataset, load_data, create_data_loader
from models import RoBERTaWithHAN
from train import train_epoch, evaluate_model
import pandas as pd
import torch
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from losses import FocalLoss, StandardLoss
from metrics import compute_detailed_metrics, save_epoch_results_to_csv, find_best_epoch, save_single_fold_detailed_results
from config import get_class_mapping
from tqdm import tqdm


def train_hierarchical_single_fold(config, fold_num, embedding_file, output_dir=None):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_embeddings = embedding_file is not None
    class_names = list(get_class_mapping().values())

    timing_stats = {
        "data_loading_time": 0,
        "model_setup_time": 0,
        "training_time": 0,
        "evaluation_time": 0,
        "total_time": 0,
        "epoch_times": [],
        "inference_times": []
    }

    train_path = config.data.train_path_template.format(fold=fold_num)
    test_path = config.data.test_path_template.format(fold=fold_num)

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"Data files not found for fold {fold_num}")

    data_load_start = time.time()
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    timing_stats["data_loading_time"] = time.time() - data_load_start

    print(f"Original data: {len(train_df)} train, {len(test_df)} test samples")
    print(f"Data loading time: {timing_stats['data_loading_time']:.2f}s")

    if use_embeddings:
        config.model.use_embeddings = True

        train_dataset = HierarchicalLGBTQDataset(
            train_df,
            embedding_file=embedding_file,
            split_name='train',
            use_previous_comments=config.model.use_previous_comments,
            prev_comment_strategy="last_10"
        )
        test_dataset = HierarchicalLGBTQDataset(
            test_df,
            embedding_file=embedding_file,
            split_name='test',
            use_previous_comments=config.model.use_previous_comments,
            prev_comment_strategy="last_10"
        )

        print(f"Hierarchical samples: {len(train_dataset)} train, {len(test_dataset)} test")
    else:
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(config.model.model_name)
        config.model.use_embeddings = False

        train_dataset = HierarchicalLGBTQDataset(
            train_df,
            tokenizer=tokenizer,
            max_length=config.training.max_length,
            use_previous_comments=config.model.use_previous_comments
        )
        test_dataset = HierarchicalLGBTQDataset(
            test_df,
            tokenizer=tokenizer,
            max_length=config.training.max_length,
            use_previous_comments=config.model.use_previous_comments
        )


    train_loader = create_data_loader(
        train_dataset,
        config.training.batch_size,
        config.training.use_weighted_sampling
    )
    test_loader = create_data_loader(
        test_dataset,
        config.training.batch_size,
        use_weighted_sampling=False,
        shuffle=False
    )

    model_setup_start = time.time()
    model = RoBERTaWithHAN(config.model).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if config.training.use_focal_loss:
        alpha_tensor = torch.tensor(config.training.focal_alpha, device=device)
        criterion = FocalLoss(gamma=config.training.focal_gamma, alpha=alpha_tensor)
    else:
        criterion = StandardLoss()

    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)
    total_steps = len(train_loader) * config.training.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.training.warmup_ratio),
        num_training_steps=total_steps
    )
    timing_stats["model_setup_time"] = time.time() - model_setup_start

    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model setup time: {timing_stats['model_setup_time']:.2f}s")

    fold_results = []
    best_model_state = None
    best_epoch = 0
    best_macro_f1 = -1

    print(f"\nStarting hierarchical training for fold {fold_num}")
    print(f"Strategy: Dynamic previous comment selection (last 10 comments)")

    training_start_time = time.time()

    for epoch in range(config.training.num_epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{config.training.num_epochs} - Fold {fold_num}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, use_embeddings
        )

        inference_start_time = time.time()
        metrics, labels, preds, probs, indices = evaluate_model(
            model, test_loader, device, class_names, use_embeddings
        )
        inference_time = time.time() - inference_start_time
        timing_stats["inference_times"].append(inference_time)

        epoch_time = time.time() - epoch_start_time
        timing_stats["epoch_times"].append(epoch_time)

        metrics["epoch"] = epoch + 1
        metrics["loss"] = train_loss
        fold_results.append(metrics)

        if metrics['macro_avg_f1'] > best_macro_f1:
            best_macro_f1 = metrics['macro_avg_f1']
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

        print(f"Loss: {train_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_avg_f1']:.4f} {'(BEST)' if metrics['macro_avg_f1'] == best_macro_f1 else ''}")
        print(f"Weighted F1: {metrics['weighted_avg_f1']:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s | Inference time: {inference_time:.2f}s")
        if metrics['auroc']:
            print(f"AUROC: {metrics['auroc']:.4f}")
        if metrics['auprc']:
            print(f"AUPRC: {metrics['auprc']:.4f}")

    timing_stats["training_time"] = time.time() - training_start_time
    timing_stats["total_time"] = time.time() - start_time
    timing_stats["avg_epoch_time"] = sum(timing_stats["epoch_times"]) / len(timing_stats["epoch_times"]) if timing_stats["epoch_times"] else 0
    timing_stats["avg_inference_time"] = sum(timing_stats["inference_times"]) / len(timing_stats["inference_times"]) if timing_stats["inference_times"] else 0
    timing_stats["inference_samples_per_second"] = len(test_df) / timing_stats["avg_inference_time"] if timing_stats["avg_inference_time"] > 0 else 0

    model.load_state_dict(best_model_state)
    print(f"\nBest epoch for fold {fold_num}: {best_epoch} (Macro F1: {best_macro_f1:.4f})")
    print(f"Training time: {timing_stats['training_time']:.2f}s | Avg inference: {timing_stats['avg_inference_time']:.2f}s | Throughput: {timing_stats['inference_samples_per_second']:.1f} samples/s")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, f"model_fold_{fold_num}_hierarchical.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

        save_epoch_results_to_csv(fold_results, fold_num, output_dir)
        save_single_fold_detailed_results(fold_results, fold_num, output_dir)

        best_epoch_idx, best_metrics = find_best_epoch(fold_results, 'macro_avg_f1')
        results_summary = {
            "fold": fold_num,
            "storage_type": "hierarchical",
            "best_epoch": best_metrics['epoch'],
            "best_macro_f1": best_metrics['macro_avg_f1'],
            "best_accuracy": best_metrics['accuracy'],
            "best_balanced_accuracy": best_metrics.get('balanced_accuracy', None),
            "embedding_file": embedding_file,
            "strategy": "last_10_comments",
            "best_epoch_confusion_matrix": best_metrics.get('confusion_matrix', None),
            "config": {
                "use_hierarchical_attention": config.model.use_hierarchical_attention,
                "use_previous_comments": config.model.use_previous_comments,
                "use_dynamic_attention": config.model.use_dynamic_attention,
                "use_embeddings": config.model.use_embeddings
            },
            "timing_stats": {
                "total_time": timing_stats["total_time"],
                "data_loading_time": timing_stats["data_loading_time"],
                "model_setup_time": timing_stats["model_setup_time"],
                "training_time": timing_stats["training_time"],
                "avg_epoch_time": timing_stats["avg_epoch_time"],
                "avg_inference_time": timing_stats["avg_inference_time"],
                "inference_samples_per_second": timing_stats["inference_samples_per_second"],
                "total_params": total_params,
                "trainable_params": trainable_params
            }
        }

        import json
        with open(os.path.join(output_dir, f"fold_{fold_num}_summary.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)

    return model, fold_results


def main():
    parser = argparse.ArgumentParser(description='Train with hierarchical embeddings')
    parser.add_argument('--embedding_file', type=str, required=True,
                       help='Path to hierarchical embedding HDF5 file')
    parser.add_argument('--fold', type=int, default=1,
                       help='Fold number to train on')
    parser.add_argument('--output_dir', type=str, default='hierarchical_results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (override config)')

    args = parser.parse_args()

    if not os.path.exists(args.embedding_file):
        raise FileNotFoundError(f"Hierarchical embedding file not found: {args.embedding_file}")

    config = Config()
    config.data.fold = args.fold

    if args.epochs:
        config.training.num_epochs = args.epochs

    print("=" * 60)
    print(f"Embedding file: {args.embedding_file}")
    print(f"Fold: {args.fold}")
    print(f"Output directory: {args.output_dir}")


    model, results = train_hierarchical_single_fold(
        config, args.fold, args.embedding_file, args.output_dir
    )

    print(f"\nHIERARCHICAL TRAINING COMPLETED")
    print(f"Results saved to: {args.output_dir}")

    best_epoch_idx, best_metrics = find_best_epoch(results, 'macro_avg_f1')
    print(f"\nFINAL RESULTS:")
    print(f"Best Epoch: {best_metrics['epoch']}")
    print(f"Best Macro F1: {best_metrics['macro_avg_f1']:.4f}")
    print(f"Best Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Best Balanced Accuracy: {best_metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
