import torch
from typing import List
from shakespeare_generator.config.config import CONFIG
from shakespeare_generator.models.bigram_language_model import BigramLanguageModel
from pathlib import Path
import json
from loguru import logger

# define constants from configuration file
logger.info('Defining constants from the configuration file....')
MODEL_PATH = CONFIG.paths.model_path
MODEL_BASE_NAME = CONFIG.paths.model_base_name
MODEL_VERSION = CONFIG.generation.model_version
MODEL_EXTENSION = CONFIG.paths.model_extension
DEVICE = CONFIG.device
MAX_NEW_TOKENS = CONFIG.generation.max_new_tokens
ITOS_PATH = Path(CONFIG.paths.itos_path).resolve()

# decoder: take a list of integers, output a string
def decode(li: List[int]) -> str:
    # import map of integers to characters for decoder
    logger.info(f'Importing map of integers to characters for decoder from: {ITOS_PATH}')
    with open(ITOS_PATH, 'r', encoding='utf-8') as f:
        itos_list = json.load(f)
    ITOS = {int(k): v for k, v in itos_list}
    logger.debug(f'itos length: {len(ITOS)}')
    return ''.join([ITOS[i] for i in li])

def generate():
    # Load the model state dict
    logger.info('Loading empty model....')
    model = BigramLanguageModel()
    load_path = Path(MODEL_PATH).resolve() / F'{MODEL_BASE_NAME}_v{MODEL_VERSION}.{MODEL_EXTENSION}'
    logger.info(f'Loading state dictionary from {load_path}....')
    model.load_state_dict(torch.load(load_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    logger.info(decode(model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()))