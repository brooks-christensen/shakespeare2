import torch
from torch import nn
from torch.nn import functional as F
from shakespeare_generator.config.config import CONFIG
from shakespeare_generator.data.data import set_up_data

N_EMBD = CONFIG.vocabulary.n_embd
BLOCK_SIZE = CONFIG.hyperparameters.block_size
DROPOUT = CONFIG.hyperparameters.dropout
DEVICE = CONFIG.device
ATTENTION_HEADS = CONFIG.hyperparameters.attention_heads
N_LAYERS = CONFIG.hyperparameters.n_layers

(
    TEXT,
    VOCAB_SIZE,
    ITOS,
    DATA
) = set_up_data()


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

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
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C*n_heads)
        out = self.dropout(self.proj(out))
        return out
    

class FeedForward(nn.Module):
    """ simple feedforward network """

    def __init__(self, N_EMBD):
        super().__init__()

        self.net = nn.Sequential(
            # multiply by 4 per "Attention is All You Need" paper
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            # projection layer going back into the original pathway
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    """ transformer block: communication followed by computation """

    def __init__(self, N_EMBD, n_heads):
        super().__init__()

        self.sa = MultiHeadAttention(n_heads, N_EMBD // n_heads)
        self.ffwd = FeedForward(N_EMBD)
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        # with skip connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class LayerNorm1d:

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, device=DEVICE))
        self.beta = nn.Parameter(torch.zeros(dim, device=DEVICE))

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
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_heads=ATTENTION_HEADS) for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(N_EMBD) # final layer norm
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C) C:N_EMBD
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
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
            idx_cond = idx[:, -BLOCK_SIZE:] # (B,block_size)
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