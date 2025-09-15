import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from models import RoBERTaWithHAN
from dataset import LGBTQDataset, create_data_loader
from metrics import compute_detailed_metrics
from config import get_class_mapping


def load_model(model_path, config, device):
    model = RoBERTaWithHAN(config.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def evaluate_on_dataset(model, dataset, config, device, use_embeddings=False):
    class_names = list(get_class_mapping().values())
    test_loader = create_data_loader(dataset, config.training.batch_size,
                                    use_weighted_sampling=False, shuffle=False)
    
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            if use_embeddings:
                post_emb = batch["post_embeddings"].to(device)
                post_mask = batch["post_attention_mask"].to(device)
                comment_emb = batch["comment_embeddings"].to(device)
                comment_mask = batch["comment_attention_mask"].to(device)
                cyber_target = batch["cyber_target"].to(device)
                row_idx = batch["row_idx"]
                
                prev_emb = batch.get("prev_embeddings")
                prev_mask = batch.get("prev_attention_mask")
                if prev_emb is not None:
                    prev_emb = prev_emb.to(device)
                    prev_mask = prev_mask.to(device)
                
                logits = model(post_emb, post_mask, comment_emb, comment_mask, prev_emb, prev_mask)
            else:
                post_input_ids = batch["post_input_ids"].to(device)
                post_attention_mask = batch["post_attention_mask"].to(device)
                comment_input_ids = batch["comment_input_ids"].to(device)
                comment_attention_mask = batch["comment_attention_mask"].to(device)
                cyber_target = batch["cyber_target"].to(device)
                row_idx = batch["row_idx"]
                
                prev_input_ids = batch.get("prev_input_ids")
                prev_attention_mask = batch.get("prev_attention_mask")
                if prev_input_ids is not None:
                    prev_input_ids = prev_input_ids.to(device)
                    prev_attention_mask = prev_attention_mask.to(device)
                
                logits = model(post_input_ids, post_attention_mask,
                              comment_input_ids, comment_attention_mask,
                              prev_input_ids, prev_attention_mask)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(cyber_target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            for i, idx in enumerate(row_idx.numpy()):
                predictions.append({
                    "row_idx": int(idx),
                    "true_label": int(cyber_target[i].cpu().numpy()),
                    "predicted_label": int(preds[i].cpu().numpy()),
                    "probabilities": probs[i].cpu().numpy().tolist(),
                    "is_correct": cyber_target[i].cpu().numpy() == preds[i].cpu().numpy()
                })
    
    metrics = compute_detailed_metrics(all_labels, all_preds, all_probs, class_names)
    return metrics, predictions


def evaluate_single_model(model_path, data_path, config, embedding_file=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_embeddings = embedding_file is not None
    
    if use_embeddings:
        config.model.use_embeddings = True
    
    model = load_model(model_path, config, device)
    
    from dataset import load_data
    df = load_data(data_path)
    
    if use_embeddings:
        split_name = 'test' if 'test' in data_path else 'train'
        dataset = LGBTQDataset(df, embedding_file=embedding_file, split_name=split_name,
                              use_previous_comments=config.model.use_previous_comments)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(config.model.model_name)
        dataset = LGBTQDataset(df, tokenizer, config.training.max_length,
                              config.model.use_previous_comments)
    
    metrics, predictions = evaluate_on_dataset(model, dataset, config, device, use_embeddings)
    
    class_mapping = get_class_mapping()
    print(classification_report(
        [p["true_label"] for p in predictions],
        [p["predicted_label"] for p in predictions],
        target_names=list(class_mapping.values())
    ))
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_avg_f1']:.4f}")
    if metrics['auroc']:
        print(f"AUROC: {metrics['auroc']:.4f}")
    
    return metrics, predictions


def analyze_errors(predictions, df):
    errors = [p for p in predictions if not p["is_correct"]]
    class_mapping = get_class_mapping()
    
    error_analysis = []
    for error in errors:
        row = df.iloc[error["row_idx"]]
        error_analysis.append({
            "post_text": row.get("s_owner_comment", ""),
            "comment_text": row.get("c_comment_content", ""),
            "true_label": class_mapping[error["true_label"]],
            "predicted_label": class_mapping[error["predicted_label"]],
            "probabilities": error["probabilities"]
        })
    
    print(f"\nError Analysis: {len(errors)} misclassified out of {len(predictions)}")
    print(f"Error Rate: {len(errors)/len(predictions):.4f}")
    
    return error_analysis


def compare_models(model_paths, data_path, config):
    results = {}
    
    for i, model_path in enumerate(model_paths):
        print(f"\n=== Evaluating Model {i+1}: {model_path} ===")
        metrics, predictions = evaluate_single_model(model_path, data_path, config)
        results[f"model_{i+1}"] = {
            "path": model_path,
            "metrics": metrics,
            "predictions": predictions
        }
    
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(f"{model_name}: Acc={metrics['accuracy']:.4f}, F1={metrics['macro_avg_f1']:.4f}")
    
    return results