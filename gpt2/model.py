import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class Transformer:
    def __init__(self):
        pass

# Based on gpt2 specs
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size = 50257
    n_layer: int = 12 # TODO: I think there are many hidden layers
    n_embed: int = 768
    n_head: int = 12
    hidden_dim = 4 * n_embed

class GPT2(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token and Positional Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embed) # (502557, 768)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embed) # (1024, 768)

        # Attention
        self.multihead_attn = nn.MultiheadAttention(config.n_embed, config.n_head, batch_first=True)

        # LayerNorm
        self.layer_norm1 = nn.LayerNorm(config.n_embed) # (N, 768)
        self.layer_norm2 = nn.LayerNorm(config.n_embed)

        # Feed Forward Layer
        self.fc1 = nn.Linear(config.n_embed, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.n_embed)
        self.relu = nn.ReLU()

        # Linear and softmax output
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        
    def forward(self, seq: torch.Tensor, targets: torch.Tensor=None):
        B, T = seq.size()
        device = seq.device

        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=device) # Positional info
        x = self.tok_emb(seq) + self.pos_emb(pos) 

        # Layer Norm, Attention, Add residual
        x = self.layer_norm1(x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, is_causal=True, attn_mask=attn_mask)
        x = x + attn_output

        # Norm
        norm = self.layer_norm2(x)

        # TODO: I think use n_layer for many hidden layers
        ffn = self.fc2(self.relu(self.fc1(norm)))

        # Add residuals and output values
        output = self.lm_head(x + ffn)

        # Calcluate loss if given target labels. We need to flatten tensor for cross_entropy
        loss = None
        if targets is not None:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))

        return output, loss
    
    @torch.no_grad()
    def generate(self, seq, max_tokens=16):
        for _ in range(max_tokens):
            # Stay in block size
            if seq.size(1) > self.config.block_size:
                seq = seq[:, -self.config.block_size:]

            # Get probs for next token
            output, _ = self(seq)
            logits = output[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Sample and concat to seq
            out_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat((seq, out_token), dim=1)
        
        return seq