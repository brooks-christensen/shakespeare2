import subprocess
from torch import tensor, long
from typing import List
from pathlib import Path
import json
from loguru import logger

from shakespeare_generator.config.config import CONFIG


def download_and_read_data(url: str, download_path: Path) -> str:
    if not download_path.exists():
        logger.info(f'Training data file does not exist, downloading data file to {download_path}')
        try:
            subprocess.run(['wget', url], check=True)
            logger.info('File downloaded successfully')
        except subprocess.CalledProcessError as e:
            logger.info('Error downloading file:', e)
    with open(download_path, 'r', encoding='utf-8') as f:
        return f.read()


def create_vocab(text: str):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    # Save itos as UTF-8 characters in a JSON file
    itos_path = Path(CONFIG.paths.itos_path).resolve()
    logger.info(f'Writing map of integers to characters for decoding to {itos_path}')
    with open(itos_path, 'w', encoding='utf-8') as f:
        json.dump(list(itos.items()), f)
    # create encoder function
    def encode(s: str) -> List[int]:
        return [stoi[ch] for ch in s]
    data = tensor(encode(text), dtype=long)
    return text, vocab_size, itos, data


# Cache the result so subsequent calls return the same data
_DATA_CACHE = None

def set_up_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        # You can pull these values from your config module
        URL = CONFIG.paths.data_url
        logger.info(f'Data download URL: {URL}')
        DOWNLOAD_PATH = Path(CONFIG.paths.data_path).resolve()
        logger.info(f'Data download path: {DOWNLOAD_PATH}')
        text = download_and_read_data(URL, DOWNLOAD_PATH)
        _DATA_CACHE = create_vocab(text)
        logger.debug(f'text length: {len(_DATA_CACHE[0])}')
        logger.info(f'vocab_size: {_DATA_CACHE[1]}')
        logger.debug(f'itos length: {len(_DATA_CACHE[2])}')
        logger.debug(f'data type: {type(_DATA_CACHE[3])}')
    return _DATA_CACHE
