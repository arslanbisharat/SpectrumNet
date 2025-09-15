import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, average_precision_score
import pandas as pd
import os
import hashlib
import json
from datetime import datetime


def calculate_per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc


def calculate_auroc(y_true, y_score, num_classes):
    try:
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        if num_classes == 2:
            return roc_auc_score(y_true, y_score[:, 1])
        else:
            unique_classes = np.unique(y_true)
            if len(unique_classes) < num_classes:
                print(f"Warning: only {len(unique_classes)} classes present in validation set.")
            
            valid_mask = (y_true >= 0) & (y_true < num_classes)
            if not np.all(valid_mask):
                print(f"Warning: Invalid class indices found for AUROC. Valid range: [0, {num_classes-1}]")
                y_true = y_true[valid_mask]
                y_score = y_score[valid_mask]
            
            if len(y_true) == 0:
                return None
                
            y_true_bin = np.eye(num_classes)[y_true.astype(int)]
            return roc_auc_score(y_true_bin, y_score, multi_class='ovr')
    except Exception as e:
        print(f"Could not calculate AUROC: {e}")
        return None


def calculate_auprc(y_true, y_score, num_classes):
    try:
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        if num_classes == 2:
            return average_precision_score(y_true, y_score[:, 1])
        else:
            unique_classes = np.unique(y_true)
            if len(unique_classes) < num_classes:
                print(f"Warning: only {len(unique_classes)} classes present for AUPRC calculation.")
            
            valid_mask = (y_true >= 0) & (y_true < num_classes)
            if not np.all(valid_mask):
                print(f"Warning: Invalid class indices found. Valid range: [0, {num_classes-1}]")
                y_true = y_true[valid_mask]
                y_score = y_score[valid_mask]
            
            if len(y_true) == 0:
                return None
                
            y_true_bin = np.eye(num_classes)[y_true.astype(int)]
            auprc_scores = []
            for i in range(num_classes):
                if np.sum(y_true_bin[:, i]) > 0:
                    score = average_precision_score(y_true_bin[:, i], y_score[:, i])
                    auprc_scores.append(score)
            return np.mean(auprc_scores) if auprc_scores else None
    except Exception as e:
        print(f"Could not calculate AUPRC: {e}")
        return None


def get_confusion_matrix_values(cm, class_idx):
    tp = cm[class_idx, class_idx]
    fp = np.sum(cm[:, class_idx]) - tp
    fn = np.sum(cm[class_idx, :]) - tp
    tn = np.sum(cm) - (tp + fp + fn)
    return tp, fp, tn, fn


def compute_detailed_metrics(y_true, y_pred, y_probs, class_names):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        tp, fp, tn, fn = get_confusion_matrix_values(cm, i)
        class_metrics[class_name] = {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Precision": float(report[class_name]["precision"]),
            "Recall": float(report[class_name]["recall"]),
            "F1-Score": float(report[class_name]["f1-score"])
        }
    
    try:
        auroc = calculate_auroc(y_true, y_probs, len(class_names))
    except Exception as e:
        print(f"Could not calculate AUROC: {e}")
        auroc = None
    
    try:
        auprc = calculate_auprc(y_true, y_probs, len(class_names))
    except Exception as e:
        print(f"Could not calculate AUPRC: {e}")
        auprc = None
    
    return {
        "accuracy": float(report["accuracy"]),
        "balanced_accuracy": float(balanced_acc),
        "macro_avg_f1": float(report["macro avg"]["f1-score"]),
        "macro_avg_precision": float(report["macro avg"]["precision"]),
        "macro_avg_recall": float(report["macro avg"]["recall"]),
        "weighted_avg_f1": float(report["weighted avg"]["f1-score"]),
        "weighted_avg_precision": float(report["weighted avg"]["precision"]),
        "weighted_avg_recall": float(report["weighted avg"]["recall"]),
        "auroc": auroc,
        "auprc": auprc,
        "class_metrics": class_metrics,
        "confusion_matrix": cm.tolist()
    }



def find_best_epoch(fold_results, metric='macro_avg_f1'):
    best_epoch_idx = 0
    best_score = -1
    
    for i, epoch_result in enumerate(fold_results):
        if epoch_result.get(metric, 0) > best_score:
            best_score = epoch_result[metric]
            best_epoch_idx = i
    
    return best_epoch_idx, fold_results[best_epoch_idx]


def generate_config_hash(config):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    config_hash = generate_config_hash(config)
    
    new_row = {
        'timestamp': timestamp,
        'fold': fold_num,
        'best_epoch': best_metrics['epoch'],
        'accuracy': best_metrics['accuracy'],
        'balanced_accuracy': best_metrics['balanced_accuracy'],
        'macro_f1': best_metrics['macro_avg_f1'],
        'macro_precision': best_metrics['macro_avg_precision'],
        'macro_recall': best_metrics['macro_avg_recall'],
        'weighted_f1': best_metrics['weighted_avg_f1'],
        'weighted_precision': best_metrics['weighted_avg_precision'],
        'weighted_recall': best_metrics['weighted_avg_recall'],
        'auroc': best_metrics.get('auroc', None),
        'auprc': best_metrics.get('auprc', None),
        'confusion_matrix': str(best_metrics['confusion_matrix']),
        'class_0_precision': best_metrics['class_metrics']['Non-Bullying']['Precision'],
        'class_0_recall': best_metrics['class_metrics']['Non-Bullying']['Recall'],
        'class_0_f1': best_metrics['class_metrics']['Non-Bullying']['F1-Score'],
        'class_1_precision': best_metrics['class_metrics']['LGBTQ+ Bullying']['Precision'],
        'class_1_recall': best_metrics['class_metrics']['LGBTQ+ Bullying']['Recall'],
        'class_1_f1': best_metrics['class_metrics']['LGBTQ+ Bullying']['F1-Score'],
        'class_2_precision': best_metrics['class_metrics']['Non-LGBTQ Bullying']['Precision'],
        'class_2_recall': best_metrics['class_metrics']['Non-LGBTQ Bullying']['Recall'],
        'class_2_f1': best_metrics['class_metrics']['Non-LGBTQ Bullying']['F1-Score'],
        'config_hash': config_hash
    }
    
    if os.path.exists(summary_csv_path):
        df = pd.read_csv(summary_csv_path)
        
        mask = (df['fold'] == fold_num) & (df['config_hash'] == config_hash)
        
        if mask.any():
            for key, value in new_row.items():
                df.loc[mask, key] = value
            print(f"Updated existing entry for fold {fold_num} in summary CSV")
        else:
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            print(f"Added new entry for fold {fold_num} to summary CSV")
    else:
        df = pd.DataFrame([new_row])
        print(f"Created new summary CSV with fold {fold_num} data")
    
    df = df.sort_values(['config_hash', 'fold']).reset_index(drop=True)
    
    df.to_csv(summary_csv_path, index=False)
    print(f"Summary results saved to: {summary_csv_path}")


def save_epoch_results_to_csv(fold_results, fold_num, results_dir):
    best_epoch_idx, best_metrics = find_best_epoch(fold_results, 'macro_avg_f1')
    
    fold_data = [{
        'fold': fold_num,
        'best_epoch': best_metrics['epoch'],
        'accuracy': best_metrics['accuracy'],
        'balanced_accuracy': best_metrics['balanced_accuracy'],
        'macro_f1': best_metrics['macro_avg_f1'],
        'macro_precision': best_metrics['macro_avg_precision'],
        'macro_recall': best_metrics['macro_avg_recall'],
        'weighted_f1': best_metrics['weighted_avg_f1'],
        'weighted_precision': best_metrics['weighted_avg_precision'],
        'weighted_recall': best_metrics['weighted_avg_recall'],
        'auroc': best_metrics.get('auroc', None),
        'auprc': best_metrics.get('auprc', None),
        'confusion_matrix': str(best_metrics['confusion_matrix']),
        'class_0_precision': best_metrics['class_metrics']['Non-Bullying']['Precision'],
        'class_0_recall': best_metrics['class_metrics']['Non-Bullying']['Recall'],
        'class_0_f1': best_metrics['class_metrics']['Non-Bullying']['F1-Score'],
        'class_0_tp': best_metrics['class_metrics']['Non-Bullying']['TP'],
        'class_0_fp': best_metrics['class_metrics']['Non-Bullying']['FP'],
        'class_0_tn': best_metrics['class_metrics']['Non-Bullying']['TN'],
        'class_0_fn': best_metrics['class_metrics']['Non-Bullying']['FN'],
        'class_1_precision': best_metrics['class_metrics']['LGBTQ+ Bullying']['Precision'],
        'class_1_recall': best_metrics['class_metrics']['LGBTQ+ Bullying']['Recall'],
        'class_1_f1': best_metrics['class_metrics']['LGBTQ+ Bullying']['F1-Score'],
        'class_1_tp': best_metrics['class_metrics']['LGBTQ+ Bullying']['TP'],
        'class_1_fp': best_metrics['class_metrics']['LGBTQ+ Bullying']['FP'],
        'class_1_tn': best_metrics['class_metrics']['LGBTQ+ Bullying']['TN'],
        'class_1_fn': best_metrics['class_metrics']['LGBTQ+ Bullying']['FN'],
        'class_2_precision': best_metrics['class_metrics']['Non-LGBTQ Bullying']['Precision'],
        'class_2_recall': best_metrics['class_metrics']['Non-LGBTQ Bullying']['Recall'],
        'class_2_f1': best_metrics['class_metrics']['Non-LGBTQ Bullying']['F1-Score'],
        'class_2_tp': best_metrics['class_metrics']['Non-LGBTQ Bullying']['TP'],
        'class_2_fp': best_metrics['class_metrics']['Non-LGBTQ Bullying']['FP'],
        'class_2_tn': best_metrics['class_metrics']['Non-LGBTQ Bullying']['TN'],
        'class_2_fn': best_metrics['class_metrics']['Non-LGBTQ Bullying']['FN']
    }]
    
    df = pd.DataFrame(fold_data)
    csv_path = f"{results_dir}/fold_{fold_num}_detailed_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed fold results saved to: {csv_path}")


def save_fold_results_to_csv(all_fold_results, results_dir):
    pass
