import torch
import torch.nn as nn
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

class Transformer:
    def __init__(self):
        pass

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        vocab_size, embedding_dim = 50257, 768 # Based on gpt2 specs
        num_heads = 6
        context_size = 1024

        # Token and Positional Embeddings
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim) 
        self.pos_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.attn_mask = torch.triu(torch.ones(size=(context_size, context_size)), diagonal=1).bool() # Triangle; True means don't attend
        
        
    def forward(self, x):
        x = self.tok_emb(x) + self.pos_emb(x) # Give some positional info. Gradients flow through both, so we learn both (instead of using sin/cos wave method)
        attn_output = self.multihead_attn.forward(query=x, key=x, value=x,attn_mask=self.attn_mask)
        






input_tensor = torch.tensor(tokenizer.encode("Dylan's GPT2"))
model = Model()
print(model)
print(model(input_tensor))