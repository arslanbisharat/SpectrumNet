import torch
import torch.nn as nn


class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.word_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.sentence_attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
    
    def forward(self, embeddings, mask):
        word_attn, _ = self.word_attention(embeddings, embeddings, embeddings,
                                           key_padding_mask=~mask.bool())
        sent_attn, _ = self.sentence_attention(word_attn, word_attn, word_attn)
        return sent_attn.mean(dim=1)


class DynamicContextualAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
    
    def forward(self, contexts):
        aggregated, _ = self.attn(contexts, contexts, contexts)
        aggregated = aggregated.mean(dim=1)
        return aggregated


class SimplePooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, embeddings, mask=None):
        if mask is not None:
            masked_embeddings = embeddings * mask.unsqueeze(-1)
            return masked_embeddings.mean(dim=1)
        return embeddings.mean(dim=1)