import pandas as pd
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer


class LGBTQDataset(Dataset):
    def __init__(self, df, tokenizer=None, max_length=512, use_previous_comments=True, embedding_file=None, split_name=None):
        self.df = df.copy()
        self.embedding_file = embedding_file
        self.split_name = split_name
        self.use_embeddings = embedding_file is not None
        
        if not self.use_embeddings:
            if "s_unit_id" in df.columns and use_previous_comments:
                self.df = self.compute_prev_comments(self.df)
            self.tokenizer = tokenizer
        
        self.df = self.df.reset_index(drop=True)
        self.max_length = max_length
        self.use_previous_comments = use_previous_comments
        
        if self.use_embeddings:
            self.h5_file = h5py.File(embedding_file, 'r')
            self.split_data = self.h5_file[split_name]
            self._validate_embedding_integrity()

    def compute_prev_comments(self, df):
        df['prev_comments'] = ""
        for post_id, group in df.groupby("s_unit_id", sort=False):
            prev_list = []
            for idx in group.index:
                df.at[idx, 'prev_comments'] = " </s> ".join(prev_list)
                prev_list.append(str(df.at[idx, 'c_comment_content']))
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.use_embeddings:
            return self._get_embeddings(idx)
        else:
            return self._get_tokenized(idx)
    
    def _get_embeddings(self, idx):
        result = {
            "post_embeddings": torch.from_numpy(self.split_data['post_embeddings'][idx]),
            "post_attention_mask": torch.from_numpy(self.split_data['post_masks'][idx]),
            "comment_embeddings": torch.from_numpy(self.split_data['comment_embeddings'][idx]),
            "comment_attention_mask": torch.from_numpy(self.split_data['comment_masks'][idx]),
            "cyber_target": torch.from_numpy(self.split_data['targets'][idx]),
            "row_idx": idx
        }
        
        if self.use_previous_comments and 'prev_embeddings' in self.split_data:
            result.update({
                "prev_embeddings": torch.from_numpy(self.split_data['prev_embeddings'][idx]),
                "prev_attention_mask": torch.from_numpy(self.split_data['prev_masks'][idx])
            })
        
        return result
    
    def _validate_embedding_integrity(self):
        if 's_unit_ids' in self.split_data and 'row_indices' in self.split_data:
            stored_s_unit_ids = [s.decode() if isinstance(s, bytes) else str(s) for s in self.split_data['s_unit_ids'][:]]
            stored_row_indices = self.split_data['row_indices'][:]
            
            df_s_unit_ids = [str(row.get("s_unit_id", "")) for _, row in self.df.iterrows()]
            
            if len(stored_s_unit_ids) != len(self.df):
                raise ValueError(f"Embedding count mismatch: stored {len(stored_s_unit_ids)}, DataFrame {len(self.df)}")
            
            mismatches = []
            for i, (stored_id, df_id) in enumerate(zip(stored_s_unit_ids, df_s_unit_ids)):
                if stored_id != df_id:
                    mismatches.append(f"Row {i}: stored='{stored_id}' vs df='{df_id}'")
            
            if mismatches:
                raise ValueError(f"s_unit_id mismatches found:\n" + "\n".join(mismatches[:5]))
            
            print(f"Embedding integrity validated: {len(self.df)} samples with consistent s_unit_id relationships")
        else:
            pass
    
    def _get_tokenized(self, idx):
        row = self.df.iloc[idx]
        post_text = str(row["s_owner_comment"]) if pd.notna(row["s_owner_comment"]) else ""
        comment_text = str(row["c_comment_content"]) if pd.notna(row["c_comment_content"]) else ""
        
        post_enc = self.tokenizer(post_text, truncation=True, padding="max_length",
                                  max_length=self.max_length, return_tensors="pt")
        comment_enc = self.tokenizer(comment_text, truncation=True, padding="max_length",
                                     max_length=self.max_length, return_tensors="pt")
        
        result = {
            "post_input_ids": post_enc["input_ids"].squeeze(0),
            "post_attention_mask": post_enc["attention_mask"].squeeze(0),
            "comment_input_ids": comment_enc["input_ids"].squeeze(0),
            "comment_attention_mask": comment_enc["attention_mask"].squeeze(0),
            "cyber_target": torch.tensor(row["target"], dtype=torch.long),
            "row_idx": idx
        }
        
        if self.use_previous_comments and "prev_comments" in row:
            prev_text = str(row["prev_comments"]) if pd.notna(row["prev_comments"]) else ""
            prev_enc = self.tokenizer(prev_text, truncation=True, padding="max_length",
                                      max_length=self.max_length, return_tensors="pt")
            result.update({
                "prev_input_ids": prev_enc["input_ids"].squeeze(0),
                "prev_attention_mask": prev_enc["attention_mask"].squeeze(0)
            })
        
        return result


def load_data(data_path):
    df = pd.read_csv(data_path)

    if "target" not in df.columns:
        df["target"] = 0
        df.loc[(df["c_cyberbullying_majority"]=="t") & (df["c_topic_gender_majority"]=="t"), "target"] = 1
        df.loc[(df["c_cyberbullying_majority"]=="t") & (df["c_topic_gender_majority"]=="f"), "target"] = 2
    
    return df


def create_weighted_sampler(df):
    target_counts = df["target"].value_counts().to_dict()
    sample_weights = [1.0 / target_counts[t] for t in df["target"].tolist()]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_data_loader(dataset, batch_size=8, use_weighted_sampling=True, shuffle=False):
    if use_weighted_sampling and hasattr(dataset, 'df'):
        sampler = create_weighted_sampler(dataset.df)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)