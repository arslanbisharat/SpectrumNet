import pandas as pd
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer
from abc import ABC, abstractmethod


class PreviousCommentStrategy(ABC):
    @abstractmethod
    def select_previous_comments(self, h5_file, s_unit_id, current_comment_id, split_name):
        pass


class LastNCommentsStrategy(PreviousCommentStrategy):
    def __init__(self, n=5):
        self.n = n
    
    def select_previous_comments(self, h5_file, s_unit_id, current_comment_id, split_name):
        if current_comment_id == 0:
            return None, None
        
        comments_group = h5_file[split_name]['comments'][s_unit_id]
        available_comments = min(current_comment_id, self.n)
        
        if available_comments == 0:
            return None, None
        
        start_idx = max(0, current_comment_id - available_comments)
        selected_embeddings = []
        selected_masks = []
        
        for i in range(start_idx, current_comment_id):
            comment_group = comments_group[str(i)]
            embedding = torch.from_numpy(comment_group['embedding'][:]).squeeze(0)
            mask = torch.from_numpy(comment_group['attention_mask'][:]).squeeze(0)
            selected_embeddings.append(embedding)
            selected_masks.append(mask)
        
        if selected_embeddings:
            combined_embeddings = torch.stack(selected_embeddings).mean(dim=0)
            combined_masks = torch.stack(selected_masks).float().mean(dim=0)
            combined_masks = (combined_masks > 0.5).long()
            return combined_embeddings, combined_masks
        
        return None, None


class WeightedRecentCommentsStrategy(PreviousCommentStrategy):
    def __init__(self, n=10, decay_factor=0.8):
        self.n = n
        self.decay_factor = decay_factor
    
    def select_previous_comments(self, h5_file, s_unit_id, current_comment_id, split_name):
        if current_comment_id == 0:
            return None, None
        
        comments_group = h5_file[split_name]['comments'][s_unit_id]
        available_comments = min(current_comment_id, self.n)
        
        if available_comments == 0:
            return None, None
        
        start_idx = max(0, current_comment_id - available_comments)
        weighted_embeddings = []
        weighted_masks = []
        weights = []
        
        for i in range(start_idx, current_comment_id):
            comment_group = comments_group[str(i)]
            embedding = torch.from_numpy(comment_group['embedding'][:]).squeeze(0)
            mask = torch.from_numpy(comment_group['attention_mask'][:]).squeeze(0)
            
            distance_from_current = current_comment_id - i
            weight = self.decay_factor ** (distance_from_current - 1)
            
            weighted_embeddings.append(embedding * weight)
            weighted_masks.append(mask.float() * weight)
            weights.append(weight)
        
        if weighted_embeddings:
            total_weight = sum(weights)
            combined_embeddings = torch.stack(weighted_embeddings).sum(dim=0) / total_weight
            combined_masks = torch.stack(weighted_masks).sum(dim=0) / total_weight
            combined_masks = (combined_masks > 0.5).long()
            return combined_embeddings, combined_masks
        
        return None, None


class AllPreviousCommentsStrategy(PreviousCommentStrategy):
    def __init__(self, max_comments=50):
        self.max_comments = max_comments
    
    def select_previous_comments(self, h5_file, s_unit_id, current_comment_id, split_name):
        if current_comment_id == 0:
            return None, None
        
        comments_group = h5_file[split_name]['comments'][s_unit_id]
        available_comments = min(current_comment_id, self.max_comments)
        
        if available_comments == 0:
            return None, None
        
        start_idx = max(0, current_comment_id - available_comments)
        all_embeddings = []
        all_masks = []
        
        for i in range(start_idx, current_comment_id):
            comment_group = comments_group[str(i)]
            embedding = torch.from_numpy(comment_group['embedding'][:]).squeeze(0)
            mask = torch.from_numpy(comment_group['attention_mask'][:]).squeeze(0)
            all_embeddings.append(embedding)
            all_masks.append(mask)
        
        if all_embeddings:
            combined_embeddings = torch.stack(all_embeddings).mean(dim=0)
            combined_masks = torch.stack(all_masks).float().mean(dim=0)
            combined_masks = (combined_masks > 0.5).long()
            return combined_embeddings, combined_masks
        
        return None, None


class RelevanceBasedStrategy(PreviousCommentStrategy):
    def __init__(self, n=10, use_target_filtering=True, similarity_threshold=0.3):
        self.n = n
        self.use_target_filtering = use_target_filtering
        self.similarity_threshold = similarity_threshold
    
    def select_previous_comments(self, h5_file, s_unit_id, current_comment_id, split_name):
        if current_comment_id == 0:
            return None, None
        
        comments_group = h5_file[split_name]['comments'][s_unit_id]
        available_comments = min(current_comment_id, len(comments_group))
        
        if available_comments == 0:
            return None, None
        
        current_comment_group = comments_group[str(current_comment_id)]
        current_embedding = torch.from_numpy(current_comment_group['embedding'][:]).squeeze(0)
        current_target = current_comment_group['target'][()]
        
        candidates = []
        for i in range(current_comment_id):
            comment_group = comments_group[str(i)]
            embedding = torch.from_numpy(comment_group['embedding'][:]).squeeze(0)
            target = comment_group['target'][()]
            mask = torch.from_numpy(comment_group['attention_mask'][:]).squeeze(0)
            
            similarity = torch.cosine_similarity(
                current_embedding.mean(dim=0), 
                embedding.mean(dim=0), 
                dim=0
            ).item()
            
            target_match = (target == current_target) if self.use_target_filtering else True
            recency_weight = (current_comment_id - i) / current_comment_id
            
            relevance_score = similarity * 0.6 + recency_weight * 0.4
            if target_match:
                relevance_score += 0.2
            
            if similarity >= self.similarity_threshold:
                candidates.append({
                    'embedding': embedding,
                    'mask': mask,
                    'score': relevance_score,
                    'comment_id': i
                })
        
        if not candidates:
            return self._fallback_to_recent(comments_group, current_comment_id, min(self.n, current_comment_id))
        
        candidates.sort(key=lambda x: x['score'], reverse=True)
        selected = candidates[:min(self.n, len(candidates))]
        
        selected_embeddings = []
        selected_masks = []
        weights = []
        
        for candidate in selected:
            weight = candidate['score']
            selected_embeddings.append(candidate['embedding'] * weight)
            selected_masks.append(candidate['mask'].float() * weight)
            weights.append(weight)
        
        if selected_embeddings:
            total_weight = sum(weights)
            combined_embeddings = torch.stack(selected_embeddings).sum(dim=0) / total_weight
            combined_masks = torch.stack(selected_masks).sum(dim=0) / total_weight
            combined_masks = (combined_masks > 0.5).long()
            return combined_embeddings, combined_masks
        
        return None, None
    
    def _fallback_to_recent(self, comments_group, current_comment_id, n):
        start_idx = max(0, current_comment_id - n)
        embeddings = []
        masks = []
        
        for i in range(start_idx, current_comment_id):
            comment_group = comments_group[str(i)]
            embeddings.append(torch.from_numpy(comment_group['embedding'][:]).squeeze(0))
            masks.append(torch.from_numpy(comment_group['attention_mask'][:]).squeeze(0))
        
        if embeddings:
            combined_embeddings = torch.stack(embeddings).mean(dim=0)
            combined_masks = torch.stack(masks).float().mean(dim=0)
            combined_masks = (combined_masks > 0.5).long()
            return combined_embeddings, combined_masks
        
        return None, None


class HierarchicalLGBTQDataset(Dataset):
    def __init__(self, df, tokenizer=None, max_length=512, use_previous_comments=True, 
                 embedding_file=None, split_name=None, prev_comment_strategy="last_5", 
                 memory_efficient=False, max_thread_size=100):
        
        self.df = df.copy()
        self.embedding_file = embedding_file
        self.split_name = split_name
        self.use_embeddings = embedding_file is not None
        self.use_previous_comments = use_previous_comments
        self.memory_efficient = memory_efficient
        self.max_thread_size = max_thread_size
        
        if not self.use_embeddings:
            if "s_unit_id" in df.columns and use_previous_comments:
                self.df = self.compute_prev_comments(self.df)
            self.tokenizer = tokenizer
        
        self.df = self.df.reset_index(drop=True)
        self.max_length = max_length
        
        if self.use_embeddings:
            self.h5_file = h5py.File(embedding_file, 'r')
            self.split_data = self.h5_file[split_name]
            self._setup_comment_strategy(prev_comment_strategy)
            self._build_sample_index()
            self._cache_targets_for_sampling()
            
            if self.memory_efficient:
                self._embedding_cache = {}
                self._cache_size = 1000
    
    def _setup_comment_strategy(self, strategy_name):
        if strategy_name == "last_5":
            self.prev_strategy = LastNCommentsStrategy(n=5)
        elif strategy_name == "last_10":
            self.prev_strategy = LastNCommentsStrategy(n=10)
        elif strategy_name == "weighted_recent":
            self.prev_strategy = WeightedRecentCommentsStrategy(n=10)
        elif strategy_name == "all_previous":
            self.prev_strategy = AllPreviousCommentsStrategy(max_comments=50)
        elif strategy_name == "relevance_based":
            self.prev_strategy = RelevanceBasedStrategy(n=10, use_target_filtering=True)
        elif strategy_name == "relevance_no_filter":
            self.prev_strategy = RelevanceBasedStrategy(n=10, use_target_filtering=False)
        else:
            self.prev_strategy = LastNCommentsStrategy(n=5)
    
    def _build_sample_index(self):
        self.sample_index = []
        large_threads = 0
        
        if 'comments' in self.split_data:
            for s_unit_id in self.split_data['comments'].keys():
                comment_group = self.split_data['comments'][s_unit_id]
                thread_size = len(comment_group)
                
                if thread_size > self.max_thread_size:
                    large_threads += 1
                    if self.memory_efficient:
                        print(f"Warning:  Large thread {s_unit_id} with {thread_size} comments - enabling chunked processing")
                
                for comment_id in comment_group.keys():
                    self.sample_index.append({
                        's_unit_id': s_unit_id,
                        'comment_id': int(comment_id),
                        'has_post': s_unit_id in self.split_data.get('posts', {}),
                        'original_index': comment_group[comment_id]['original_index'][()],
                        'thread_size': thread_size,
                        'is_large_thread': thread_size > self.max_thread_size
                    })
        
        self.sample_index.sort(key=lambda x: x['original_index'])
        print(f"Built sample index: {len(self.sample_index)} samples from hierarchical storage")
        if large_threads > 0:
            print(f"Found {large_threads} large threads (>{self.max_thread_size} comments)")
    
    def _cache_targets_for_sampling(self):
        self.cached_targets = []
        
        for sample in self.sample_index:
            s_unit_id = sample['s_unit_id']
            comment_id = sample['comment_id']
            try:
                target = self.split_data['comments'][s_unit_id][str(comment_id)]['target'][()]
                self.cached_targets.append(int(target))
            except (KeyError, ValueError) as e:
                print(f"Warning: Could not get target for {s_unit_id}/{comment_id}: {e}")
                self.cached_targets.append(0)
        
        print(f"Cached {len(self.cached_targets)} targets for WRS")
        target_counts = pd.Series(self.cached_targets).value_counts().sort_index()
        print(f"Class distribution: {target_counts.to_dict()}")
    
    def compute_prev_comments(self, df):
        df['prev_comments'] = ""
        for post_id, group in df.groupby("s_unit_id", sort=False):
            prev_list = []
            for idx in group.index:
                df.at[idx, 'prev_comments'] = " </s> ".join(prev_list)
                prev_list.append(str(df.at[idx, 'c_comment_content']))
        return df
    
    def __len__(self):
        if self.use_embeddings:
            return len(self.sample_index)
        return len(self.df)
    
    def __getitem__(self, idx):
        if self.use_embeddings:
            return self._get_hierarchical_embeddings(idx)
        else:
            return self._get_tokenized(idx)
    
    def _get_cached_embedding(self, cache_key, load_func):
        if not self.memory_efficient:
            return load_func()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if len(self._embedding_cache) >= self._cache_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        embedding = load_func()
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def _get_hierarchical_embeddings(self, idx):
        sample_info = self.sample_index[idx]
        s_unit_id = sample_info['s_unit_id']
        comment_id = sample_info['comment_id']
        is_large_thread = sample_info.get('is_large_thread', False)
        
        if sample_info['has_post'] and s_unit_id in self.split_data['posts']:
            if is_large_thread and self.memory_efficient:
                post_cache_key = f"post_{s_unit_id}"
                post_emb, post_mask = self._get_cached_embedding(
                    post_cache_key,
                    lambda: (
                        torch.from_numpy(self.split_data['posts'][s_unit_id]['embedding'][:]).squeeze(0),
                        torch.from_numpy(self.split_data['posts'][s_unit_id]['attention_mask'][:]).squeeze(0)
                    )
                )
            else:
                post_group = self.split_data['posts'][s_unit_id]
                post_emb = torch.from_numpy(post_group['embedding'][:]).squeeze(0)
                post_mask = torch.from_numpy(post_group['attention_mask'][:]).squeeze(0)
        else:
            post_emb = torch.zeros(self.max_length, 768)
            post_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        if is_large_thread and self.memory_efficient:
            comment_cache_key = f"comment_{s_unit_id}_{comment_id}"
            comment_emb, comment_mask, target = self._get_cached_embedding(
                comment_cache_key,
                lambda: (
                    torch.from_numpy(self.split_data['comments'][s_unit_id][str(comment_id)]['embedding'][:]).squeeze(0),
                    torch.from_numpy(self.split_data['comments'][s_unit_id][str(comment_id)]['attention_mask'][:]).squeeze(0),
                    torch.tensor(self.split_data['comments'][s_unit_id][str(comment_id)]['target'][()], dtype=torch.long)
                )
            )
        else:
            comment_group = self.split_data['comments'][s_unit_id][str(comment_id)]
            comment_emb = torch.from_numpy(comment_group['embedding'][:]).squeeze(0)
            comment_mask = torch.from_numpy(comment_group['attention_mask'][:]).squeeze(0)
            target = torch.tensor(comment_group['target'][()], dtype=torch.long)
        
        result = {
            "post_embeddings": post_emb,
            "post_attention_mask": post_mask,
            "comment_embeddings": comment_emb,
            "comment_attention_mask": comment_mask,
            "cyber_target": target,
            "row_idx": idx,
            "s_unit_id": s_unit_id,
            "comment_id": comment_id,
            "thread_size": sample_info.get('thread_size', 0),
            "is_large_thread": is_large_thread
        }
        
        if self.use_previous_comments:
            prev_emb, prev_mask = self.prev_strategy.select_previous_comments(
                self.h5_file, s_unit_id, comment_id, self.split_name
            )
            
            if prev_emb is not None:
                result["prev_embeddings"] = prev_emb
                result["prev_attention_mask"] = prev_mask
            else:
                result["prev_embeddings"] = torch.zeros(self.max_length, 768)
                result["prev_attention_mask"] = torch.zeros(self.max_length, dtype=torch.long)
        
        return result
    
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
    
    def get_thread_info(self, s_unit_id):
        if not self.use_embeddings:
            return None
        
        info = {
            'post_exists': s_unit_id in self.split_data.get('posts', {}),
            'comment_count': len(self.split_data['comments'].get(s_unit_id, {})),
            'comments': []
        }
        
        if s_unit_id in self.split_data.get('comments', {}):
            for comment_id in sorted(self.split_data['comments'][s_unit_id].keys(), key=int):
                comment_group = self.split_data['comments'][s_unit_id][comment_id]
                info['comments'].append({
                    'comment_id': int(comment_id),
                    'target': comment_group['target'][()],
                    'text_length': len(comment_group['text'][()]),
                    'original_index': comment_group['original_index'][()]
                })
        
        return info


def load_data(data_path):
    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        df["target"] = 0
        df.loc[(df["c_cyberbullying_majority"]=="t") & (df["c_topic_gender_majority"]=="t"), "target"] = 1
        df.loc[(df["c_cyberbullying_majority"]=="t") & (df["c_topic_gender_majority"]=="f"), "target"] = 2
    return df


def create_weighted_sampler(dataset):
    if hasattr(dataset, 'cached_targets') and dataset.cached_targets:
        targets = dataset.cached_targets
    elif hasattr(dataset, 'sample_index'):
        targets = []
        for sample in dataset.sample_index:
            s_unit_id = sample['s_unit_id']
            comment_id = sample['comment_id']
            comment_group = dataset.split_data['comments'][s_unit_id][str(comment_id)]
            targets.append(comment_group['target'][()])
    else:
        targets = dataset.df["target"].tolist()
    
    if not targets:
        raise ValueError("No targets found for weighted sampling")
    
    target_counts = pd.Series(targets).value_counts().to_dict()
    
    if len(target_counts) == 1:
        sample_weights = [1.0] * len(targets)
    else:
        sample_weights = [1.0 / target_counts[t] for t in targets]
    
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_data_loader(dataset, batch_size=8, use_weighted_sampling=True, shuffle=False):
    if use_weighted_sampling:
        sampler = create_weighted_sampler(dataset)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)