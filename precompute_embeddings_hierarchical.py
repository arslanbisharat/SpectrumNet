import torch
import pandas as pd
import h5py
import numpy as np
from transformers import RobertaTokenizer
from encoders import RobertaEncoder
from dataset import load_data
from tqdm import tqdm
import os
import argparse
from config import Config
from collections import defaultdict


def build_comment_hierarchy(df):
    hierarchy = defaultdict(lambda: {'post': None, 'comments': []})

    for _, row in df.iterrows():
        s_unit_id = str(row['s_unit_id'])

        if hierarchy[s_unit_id]['post'] is None:
            hierarchy[s_unit_id]['post'] = {
                'text': str(row.get('s_owner_comment', '')),
                'target': row['target'] if 's_owner_comment' in row else None
            }

        comment_data = {
            'comment_id': len(hierarchy[s_unit_id]['comments']),
            'text': str(row.get('c_comment_content', '')),
            'target': row['target'],
            'original_index': row.name if hasattr(row, 'name') else len(hierarchy[s_unit_id]['comments'])
        }

        hierarchy[s_unit_id]['comments'].append(comment_data)

    return dict(hierarchy)


def precompute_hierarchical_embeddings_for_fold(fold_num, config, output_dir, device):
    print(f"Processing Fold {fold_num} (Hierarchical)")

    train_path = config.data.train_path_template.format(fold=fold_num)
    test_path = config.data.test_path_template.format(fold=fold_num)

    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return False

    train_df = load_data(train_path)
    test_df = load_data(test_path)

    print(f"Loaded fold {fold_num}: {len(train_df)} train, {len(test_df)} test samples")

    tokenizer = RobertaTokenizer.from_pretrained(config.model.model_name)
    encoder = RobertaEncoder(
        model_name=config.model.model_name,
        freeze_roberta=config.model.freeze_roberta,
        unfreeze_last_n=config.model.unfreeze_last_n
    ).to(device)
    encoder.eval()

    output_file = os.path.join(output_dir, f"embeddings_fold_{fold_num}_hierarchical.h5")

    with h5py.File(output_file, 'w') as f:
        for split_name, df in [('train', train_df), ('test', test_df)]:

            hierarchy = build_comment_hierarchy(df.reset_index())

            split_group = f.create_group(split_name)
            posts_group = split_group.create_group('posts')
            comments_group = split_group.create_group('comments')

            post_count = 0
            comment_count = 0

            with torch.no_grad():
                for s_unit_id, thread_data in tqdm(hierarchy.items(), desc=f"Processing {split_name} threads"):
                    post_text = thread_data['post']['text']
                    comments = thread_data['comments']

                    post_group = posts_group.create_group(s_unit_id)

                    if post_text.strip():
                        post_tokens = tokenizer(
                            post_text,
                            truncation=True,
                            padding="max_length",
                            max_length=config.training.max_length,
                            return_tensors="pt"
                        )

                        post_embedding = encoder(
                            post_tokens["input_ids"].to(device),
                            post_tokens["attention_mask"].to(device)
                        ).cpu().numpy()

                        post_group.create_dataset('embedding', data=post_embedding, compression='gzip', compression_opts=9)
                        post_group.create_dataset('attention_mask', data=post_tokens["attention_mask"].numpy(), compression='gzip')
                        post_group.create_dataset('text', data=post_text, dtype=h5py.string_dtype())

                        if thread_data['post']['target'] is not None:
                            post_group.create_dataset('target', data=thread_data['post']['target'])

                    post_count += 1

                    if comments:
                        comment_thread_group = comments_group.create_group(s_unit_id)

                        for comment_idx, comment_data in enumerate(comments):
                            comment_text = comment_data['text']

                            if comment_text.strip():
                                comment_tokens = tokenizer(
                                    comment_text,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=config.training.max_length,
                                    return_tensors="pt"
                                )

                                comment_embedding = encoder(
                                    comment_tokens["input_ids"].to(device),
                                    comment_tokens["attention_mask"].to(device)
                                ).cpu().numpy()

                                comment_group = comment_thread_group.create_group(str(comment_idx))
                                comment_group.create_dataset('embedding', data=comment_embedding, compression='gzip', compression_opts=9)
                                comment_group.create_dataset('attention_mask', data=comment_tokens["attention_mask"].numpy(), compression='gzip')
                                comment_group.create_dataset('text', data=comment_text, dtype=h5py.string_dtype())
                                comment_group.create_dataset('target', data=comment_data['target'])
                                comment_group.create_dataset('original_index', data=comment_data['original_index'])
                                comment_group.create_dataset('previous_comment_id', data=max(0, comment_idx - 1))

                                comment_count += 1

            split_group.attrs['post_count'] = post_count
            split_group.attrs['comment_count'] = comment_count
            split_group.attrs['thread_count'] = len(hierarchy)

            print(f"Stored {len(hierarchy)} threads, {post_count} posts, {comment_count} comments")

    print(f"Saved hierarchical embeddings for fold {fold_num} to: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Pre-compute hierarchical RoBERTa embeddings')
    parser.add_argument('--output_dir', type=str, default='cached_embeddings_hierarchical',
                       help='Directory to save cached embeddings')
    parser.add_argument('--folds', type=str, default='all',
                       help='Folds to process (e.g., "1,2,3" or "all")')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')

    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    config = Config()

    config_dict = {
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "data": config.data.__dict__,
        "storage_type": "hierarchical",
        "max_length_per_comment": config.training.max_length
    }

    import json
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)

    if args.folds == 'all':
        fold_nums = []
        for i in range(1, 11):
            train_path = config.data.train_path_template.format(fold=i)
            test_path = config.data.test_path_template.format(fold=i)
            if os.path.exists(train_path) and os.path.exists(test_path):
                fold_nums.append(i)
    else:
        fold_nums = [int(f.strip()) for f in args.folds.split(',')]

    print(f"Processing folds: {fold_nums}")

    processed_folds = []
    for fold_num in fold_nums:
        success = precompute_hierarchical_embeddings_for_fold(fold_num, config, args.output_dir, device)
        if success:
            processed_folds.append(fold_num)

    summary = {
        "processed_folds": processed_folds,
        "total_folds": len(processed_folds),
        "output_directory": args.output_dir,
        "device_used": str(device),
        "storage_type": "hierarchical",
        "benefits": [
            "No information loss from comment truncation",
            "Each comment gets full 512-token capacity",
            "Flexible context building during training",
            "Dynamic previous comment selection"
        ],
        "config": config_dict
    }

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary")
    print(f"Successfully processed {len(processed_folds)} folds: {processed_folds}")
    print(f"Hierarchical embeddings saved to: {args.output_dir}")


if __name__ == "__main__":
    main()