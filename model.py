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

        self.attn = nn.MultiheadAttention(embedding_dim, num_heads)
        self.W = torch.randn()
        
        
    def forward(self, x):
        hidden = self.tok_emb(x) + self.pos_emb(x) # Token embedding and position embedding. Gradients flow through both, so we learn both
        attn_output = self.attn.forward()





input_tensor = torch.tensor(tokenizer.encode("Dylan's GPT2"))
model = Model()
print(model)
print(model(input_tensor))