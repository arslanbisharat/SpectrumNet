import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", freeze_roberta=True, unfreeze_last_n=0):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        if freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
            
            if unfreeze_last_n > 0:
                for i in range(12 - unfreeze_last_n, 12): 
                    for param in self.roberta.encoder.layer[i].parameters():
                        param.requires_grad = True
    
    def forward(self, input_ids, attention_mask):
        return self.roberta(input_ids, attention_mask=attention_mask)[0]


class PreviousCommentsEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers,
                          batch_first=True, bidirectional=bidirectional)
        out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(out_size, hidden_size)
    
    def forward(self, embeddings, mask):
        lengths = mask.sum(dim=1).cpu()
        
        lengths = torch.clamp(lengths, min=1)
        
        lengths, perm_idx = lengths.sort(0, descending=True)
        embeddings = embeddings[perm_idx]
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths,
                                                   batch_first=True, enforce_sorted=True)
        packed_out, hidden = self.gru(packed)
        hidden = hidden[-1]
        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[unperm_idx]
        hidden = self.fc(hidden)
        return hidden


class SimpleEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, embeddings, mask=None):
        return self.linear(embeddings.mean(dim=1))