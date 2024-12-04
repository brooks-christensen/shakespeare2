import os
import subprocess
import yaml
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F

# set random seed
torch.manual_seed = 1337

# import config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# set up config
batch_size = config['hyperparameters']['batch_size']
block_size = config['hyperparameters']['block_size']
max_iters = config['hyperparameters']['max_iters']
eval_interval = config['hyperparameters']['eval_interval']
learning_rate = float(config['hyperparameters']['learning_rate'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = config['hyperparameters']['eval_iters']
train_val_split = config['hyperparameters']['train_val_split']
max_new_tokens = config['generation']['max_new_tokens']
n_embd = config['vocabulary']['n_embd']
attention_heads = config['hyperparameters']['attention_heads']
n_layer = config['hyperparameters']['n_layers']
dropout = config['hyperparameters']['dropout']

# download text data if not present
if not os.path.exists('input.txt'):
    url = config['paths']['data_url']
    try:
        subprocess.run(['wget', url], check=True)
        print('File downloaded successfully')
    except subprocess.CalledProcessError as e:
        print('Error downloading file:', e)

# load text
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
def encode(s: str) -> List[int]:
    return [stoi[ch] for ch in s]

# decoder: take a list of integers, output a string
def decode(li: List[int]) -> str:
    return ''.join([itos[i] for i in li])

# tokenize input text
data = torch.tensor(encode(text), dtype=torch.long)

# create training and validation splits
n = int(train_val_split * len(data))
train_data, val_data = data[:n], data[n:]


# create batches
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# create loss function
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


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # compute the raw attention scores
        wei = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # apply the attention to the values
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out
    

class MultiHeadAttention(nn.Module):
    """ multi-head self-attention """

    def __init__(self, n_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C*n_heads)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ simple feedforward network """

    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            # multiply by 4 per "Attention is All You Need" paper
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # projection layer going back into the original pathway
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ transformer block: communication followed by computation """

    def __init__(self, n_embd, n_heads):
        super().__init__()

        self.sa = MultiHeadAttention(n_heads, n_embd // n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # with skip connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class LayerNorm1d:

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, device=device))
        self.beta = nn.Parameter(torch.zeros(dim, device=device))

    def __call__(self, x):
        # calculate the forward pass
        # dimension 0 -> 1 for LayerNorm
        # dimension 1 -> 0 for BatchNorm
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta # scale and shift

        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]


# create bigram language model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=attention_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) C:n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # view(-1) will also work

            # expecting loss values greater than -ln(1/vocab_size) = -ln(1/65) = 4.17
            # minimum loss value corresponds to perfectly randomized input data, which is never the case
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B,block_size)
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            # print(f"logits before softmax: {logits}")
            probs = F.softmax(logits, dim=-1) # (B,C)
            # sample from the distribution
            # print(f"probabilities after softmax: {probs}")
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)

        return idx
    
# instantiate the model
model = BigramLanguageModel()
m = model.to(device)

# instantiate the optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# training loop
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    _, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=max_new_tokens)[0].tolist()))