import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

class Transformer:
    def __init__(self):
        pass

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()

        vocab_size, embedding_dim = 50257, 768 # Based on gpt2 specs
        hidden_dim = 4 * embedding_dim
        num_heads = 6
        # context_size = 1024

        # Token and Positional Embeddings
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) # (502557, 768)
        self.pos_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        # self.attn_mask = torch.triu(torch.ones(size=(context_size, context_size)), diagonal=1).bool() # Triangle; True means don't attend

        # LayerNorm
        self.layer_norm1 = nn.LayerNorm(embedding_dim) # (N, 768)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # Feed Forward Layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)
        self.relu = nn.ReLU()

        # Linear and softmax output
        self.lm_head = nn.Linear(embedding_dim, vocab_size)
        
        
    def forward(self, seq, targets=None):
        B, T = seq.size()

        # Embeddings
        pos = torch.arange(0, T, dtype=torch.long) # Positional info
        x = self.tok_emb(seq) + self.pos_emb(pos) 

        # Layer Norm, Attention, Add residual
        x = self.layer_norm1(x)
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, is_causal=True, attn_mask=attn_mask)
        x = x + attn_output

        # Norm
        norm = self.layer_norm2(x)
        ffn = self.fc2(self.relu(self.fc1(norm)))

        # Add residuals and output values
        output = self.lm_head(x + ffn)

        # Calcluate loss if given target labels. We need to flatten tensor for cross_entropy
        loss = None
        if targets is not None:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))

        return output, loss
    