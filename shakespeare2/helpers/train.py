from pathlib import Path
import torch
from shakespeare2.config.config import CONFIG
from shakespeare2.models.bigram_language_model import BigramLanguageModel
from shakespeare2.data.data import set_up_data
from shakespeare2.utils.utils import seed_everything

# set random seed
seed_everything(1337)

TRAIN_VAL_SPLIT = CONFIG.hyperparameters.train_val_split
BLOCK_SIZE = CONFIG.hyperparameters.block_size
BATCH_SIZE = CONFIG.hyperparameters.batch_size
DEVICE = CONFIG.device
EVAL_ITERS = CONFIG.hyperparameters.eval_iters
LEARNING_RATE = float(CONFIG.hyperparameters.learning_rate)
MAX_ITERS = CONFIG.hyperparameters.max_iters
EVAL_INTERVAL = CONFIG.hyperparameters.eval_interval
MODEL_BASE_NAME = CONFIG.paths.model_base_name
MODEL_PATH = CONFIG.paths.model_path
MODEL_EXTENSION = CONFIG.paths.model_extension

(
    TEXT,
    VOCAB_SIZE,
    ITOS,
    DATA
) = set_up_data()


def train():

    # create training and validation splits
    n = int(TRAIN_VAL_SPLIT * len(DATA))
    train_data, val_data = DATA[:n], DATA[n:]

    # create batches
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y


    # loss function
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
        
    # instantiate the model
    model = BigramLanguageModel()
    m = model.to(DEVICE)

    # reporting
    print(f"number of parameters: {sum(p.numel() for p in m.parameters())}")

    # instantiate the optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    # training loop
    for iter in range(MAX_ITERS):

        # every once in a while evaluate the loss on train and val sets
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # # generate from the model
    # context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    # print(decode(m.generate(context, MAX_NEW_TOKENS=MAX_NEW_TOKENS)[0].tolist()))

    # store the model
    version = 0

    if MODEL_PATH.startswith('.'):
        MODEL_PATH = Path().cwd() / '/'.join((MODEL_PATH).split('/')[1:])

    file_path = Path(MODEL_PATH) / f'{MODEL_BASE_NAME}_v{version}.{MODEL_EXTENSION}'

    while file_path.exists():
        version += 1
        file_path = Path(MODEL_PATH) / f'{MODEL_BASE_NAME}_v{version}.{MODEL_EXTENSION}'

    torch.save(model.state_dict(), str(file_path))