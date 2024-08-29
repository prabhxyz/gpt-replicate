import torch
import torch.nn as nn
from  torch.nn import functional as F

# Hyperparameters
batch_size = 32 # Number of sequences that will be processed in parallel
block_size = 8 # Maximum context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337) # For reproducibility

# Shakespeare dataset
with open('dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# All unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoder to map characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # Encoder: Convert a string to a list of integers.
decode = lambda l: ''.join(itos[i] for i in l) # Decoder: Convert a list of integers to a string.

# Train & Test split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% of the data for training; 10% for val
train_data = data[:n]
val_data = data[n:]

# Data loader
def get_batch(split):
    # Generate a small batch of data for inputs x, and targets y.
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are (B, T) tensor of integers
        logits = self.token_embedding_table(idx)

        if targets is not None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
"""
def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        logits, loss = self(idx)
        logi

    def forward(self, x, y):
        emb = self.emb(x)
        logits = self.fc(emb)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        return logits, loss"""