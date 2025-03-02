import subprocess
from torch import tensor, long
from typing import List
from pathlib import Path
import json

from shakespeare2.config.config import CONFIG

def download_and_read_data(url: str, download_path: Path) -> str:
    if not download_path.exists():
        print(f'Training data file does not exist, downloading data file to {download_path}')
        try:
            subprocess.run(['wget', url], check=True)
            print('File downloaded successfully')
        except subprocess.CalledProcessError as e:
            print('Error downloading file:', e)
    with open(download_path, 'r', encoding='utf-8') as f:
        return f.read()

def create_vocab(text: str):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    # Save itos as UTF-8 characters in a JSON file
    with open(Path(CONFIG.paths.itos_path).resolve(), 'w', encoding='utf-8') as f:
        json.dump(list(itos.items()), f)
    # create encoder function
    def encode(s: str) -> List[int]:
        return [stoi[ch] for ch in s]
    data = tensor(encode(text), dtype=long)
    return text, vocab_size, itos, data

def get_output_path(input_path):
    # relative path
    if input_path.startswith('.'):
        output_path = Path().cwd() / '/'.join((input_path).split('/')[1:])
    # absolute path
    else:
        output_path = Path(input_path)
    return output_path

# Cache the result so subsequent calls return the same data
_DATA_CACHE = None

def set_up_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        # You can pull these values from your config module
        URL = CONFIG.paths.data_url
        DOWNLOAD_PATH = get_output_path(CONFIG.paths.data_path)
        text = download_and_read_data(URL, DOWNLOAD_PATH)
        _DATA_CACHE = create_vocab(text)
    return _DATA_CACHE
