from pathlib import Path
import torch
from shakespeare_generator.config.config import CONFIG
from shakespeare_generator.models.bigram_language_model import BigramLanguageModel
from shakespeare_generator.data.data import set_up_data
from shakespeare_generator.utils.utils import seed_everything
from loguru import logger

# set random seed
logger.info('Seeding randomization....')
seed_everything(1337)

# set up constants from config file
logger.info('Defining constants from config file....')
TRAIN_VAL_SPLIT = CONFIG.hyperparameters.train_val_split
BLOCK_SIZE = CONFIG.hyperparameters.block_size
BATCH_SIZE = CONFIG.hyperparameters.batch_size
DEVICE = CONFIG.device
EVAL_ITERS = CONFIG.hyperparameters.eval_iters
LEARNING_RATE = float(CONFIG.hyperparameters.learning_rate)
MAX_ITERS = CONFIG.hyperparameters.max_iters
EVAL_INTERVAL = CONFIG.hyperparameters.eval_interval
MODEL_BASE_NAME = CONFIG.paths.model_base_name
MODEL_EXTENSION = CONFIG.paths.model_extension

# set up constants from the data
logger.info('Setting up constants from the data....')
(
    TEXT,
    VOCAB_SIZE,
    ITOS,
    DATA
) = set_up_data()


def train():

    # create training and validation splits
    logger.info('Creating training and validation splits....')
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
    logger.info('Instantiating the model....')
    model = BigramLanguageModel()
    m = model.to(DEVICE)

    # reporting
    logger.info(f"number of parameters: {sum(p.numel() for p in m.parameters())}")

    # instantiate the optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

    # training loop
    for iter in range(MAX_ITERS):

        # every once in a while evaluate the loss on train and val sets
        if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
            losses = estimate_loss()
            logger.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        _, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # store the model (with versioning)
    version = 0
    model_path = Path(CONFIG.paths.model_path).resolve()
    file_path = Path(model_path) / f'{MODEL_BASE_NAME}_v{version}.{MODEL_EXTENSION}'
    while file_path.exists():
        version += 1
        file_path = Path(model_path) / f'{MODEL_BASE_NAME}_v{version}.{MODEL_EXTENSION}'

    logger.info(f'Saving model to {file_path}....')
    torch.save(m.state_dict(), str(file_path))