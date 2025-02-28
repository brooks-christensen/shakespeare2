import torch
from typing import List
from shakespeare2.config.config import CONFIG
from shakespeare2.data.data import set_up_data
from shakespeare2.models.bigram_language_model import BigramLanguageModel
from pathlib import Path

MODEL_PATH = CONFIG.paths.model_path
MODEL_BASE_NAME = CONFIG.paths.model_base_name
MODEL_VERSION = CONFIG.generation.model_version
MODEL_EXTENSION = CONFIG.paths.model_extension
DEVICE = CONFIG.device
MAX_NEW_TOKENS = CONFIG.generation.max_new_tokens


(
    TEXT,
    VOCAB_SIZE,
    ITOS,
    DATA
) = set_up_data()


# decoder: take a list of integers, output a string
def decode(li: List[int]) -> str:
    return ''.join([ITOS[i] for i in li])

def generate():
    # Load the model state dict
    model = BigramLanguageModel()
    load_path = Path(MODEL_PATH) / F'{MODEL_BASE_NAME}_v{MODEL_VERSION}.{MODEL_EXTENSION}'
    model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()))