import torch
import torch.nn as nn
from encoders import RobertaEncoder, PreviousCommentsEncoder
from attention import HierarchicalAttention, DynamicContextualAttention


class CyberbullyingClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes=3, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, features):
        return self.classifier(features)


class FeatureFusion(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, features):
        return self.fusion(features)


class RoBERTaWithHAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_embeddings = getattr(config, 'use_embeddings', False)
        
        if not self.use_embeddings:
            self.encoder = RobertaEncoder(
                model_name=config.model_name,
                freeze_roberta=config.freeze_roberta,
                unfreeze_last_n=config.unfreeze_last_n
            )
        
        if config.use_hierarchical_attention:
            self.attention = HierarchicalAttention(config.hidden_size, config.num_heads)
        else:
            self.attention = None
        
        if config.use_previous_comments:
            self.prev_encoder = PreviousCommentsEncoder(
                config.hidden_size, 
                config.gru_layers, 
                config.bidirectional_gru
            )
        else:
            self.prev_encoder = None
        
        if config.use_dynamic_attention:
            num_contexts = 3 if config.use_previous_comments else 2
            self.dynamic_attn = DynamicContextualAttention(config.hidden_size, 1)
        else:
            self.dynamic_attn = None
        
        self.feature_fusion = FeatureFusion(config.hidden_size, config.fusion_dropout)
        self.classifier = CyberbullyingClassifier(
            config.hidden_size, 
            config.num_classes, 
            config.classifier_dropout
        )
    
    def forward(self, *args, **kwargs):
        if self.use_embeddings:
            return self.forward_embeddings(*args, **kwargs)
        else:
            return self.forward_tokens(*args, **kwargs)
    
    def forward_embeddings(self, post_emb, post_mask, comment_emb, comment_mask, prev_emb=None, prev_mask=None):
        if self.attention:
            post_repr = self.attention(post_emb, post_mask)
            comment_repr = self.attention(comment_emb, comment_mask)
        else:
            post_repr = post_emb.mean(dim=1)
            comment_repr = comment_emb.mean(dim=1)
        
        contexts = [post_repr, comment_repr]
        
        if self.prev_encoder and prev_emb is not None:
            prev_repr = self.prev_encoder(prev_emb, prev_mask)
            contexts.append(prev_repr)
        
        if self.dynamic_attn and len(contexts) > 1:
            contexts = torch.stack(contexts, dim=1)
            aggregated_context = self.dynamic_attn(contexts)
        else:
            aggregated_context = torch.stack(contexts, dim=0).mean(dim=0)
        
        fused = self.feature_fusion(aggregated_context)
        logits = self.classifier(fused)
        return logits
    
    def forward_tokens(self, post_ids, post_mask, comment_ids, comment_mask, prev_ids=None, prev_mask=None):
        post_output = self.encoder(post_ids, post_mask)
        comment_output = self.encoder(comment_ids, comment_mask)
        
        if self.attention:
            post_repr = self.attention(post_output, post_mask)
            comment_repr = self.attention(comment_output, comment_mask)
        else:
            post_repr = post_output.mean(dim=1)
            comment_repr = comment_output.mean(dim=1)
        
        contexts = [post_repr, comment_repr]
        
        if self.prev_encoder and prev_ids is not None:
            prev_output = self.encoder(prev_ids, prev_mask)
            prev_repr = self.prev_encoder(prev_output, prev_mask)
            contexts.append(prev_repr)
        
        if self.dynamic_attn and len(contexts) > 1:
            contexts = torch.stack(contexts, dim=1)
            aggregated_context = self.dynamic_attn(contexts)
        else:
            aggregated_context = torch.stack(contexts, dim=0).mean(dim=0)
        
        fused = self.feature_fusion(aggregated_context)
        logits = self.classifier(fused)
        return logits