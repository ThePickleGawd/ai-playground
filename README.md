# Attention is All You Need

PyTorch implementation of "Attention is All You Need"

## Notes for me

Downloading dataset:

- Download raw text file. Split is 90/10 by character, not token
- Then tokenize
- Then store in np array and save as .bin file

Pipeline

- Tokenize text. Pass the encoding to model
- Context length is 1024
- We need an embedding (seems to be 768 per token)

Goals:

- Recreate GPT-2 without looking back at the Karpathy videos or github
