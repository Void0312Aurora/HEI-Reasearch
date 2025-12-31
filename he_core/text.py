import torch
import torch.nn as nn
from typing import Optional

class SimpleTextEncoder(nn.Module):
    """
    A simple semantic encoder mapping token sequences to a semantic drive u.
    Architecture: Embedding -> GRU -> Final State.
    This acts as the "Ear" or "Language Center" interface for the Entity.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim) # align with u dim
        
        # Initialize weights
        nn.init.xavier_uniform_(self.embedding.weight)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, L) LongTensor of token IDs.
            lengths: (B,) LongTensor of valid sequence lengths (optional).
        Returns:
            u: (B, hidden_dim) Semantic drive vector.
        """
        # Embed: (B, L, D)
        h = self.embedding(x)
        
        # RNN Processing
        # RNN Processing
        # We use the final hidden state as the "summary" drive
        if lengths is not None:
             # Sort by length (required for pack_padded_sequence)
             # But here assuming x is already sorted? No, DataLoaders shuffle.
             # We must sort, pack, run, unsort. 
             # Simpler: just enforce enforce_sorted=False (inefficient but easy) in newer pytorch.
             # Or sort here.
             lengths_cpu = lengths.cpu()
             total_length = x.size(1)
             
             packed = nn.utils.rnn.pack_padded_sequence(h, lengths_cpu, batch_first=True, enforce_sorted=False)
             out, h_n = self.gru(packed)
             # Unpack if needed, but we only need h_n
        else:
             out, h_n = self.gru(h)
             
        # h_n shape: (num_layers, B, H) -> (1, B, H)
        u = h_n.squeeze(0) # (B, H)
        u = self.out_proj(u)
        
        return u
