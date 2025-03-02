# config.py
from pathlib import Path
from dataclasses import dataclass
import yaml
from torch.cuda import is_available
import torch

@dataclass
class PathsConfig:
    data_url: str
    config_path: str
    data_path: str
    model_path: str
    model_base_name: str
    model_extension: str
    itos_path: str

@dataclass
class HyperparametersConfig:
    batch_size: int
    block_size: int
    max_iters: int
    eval_interval: int
    learning_rate: float
    eval_iters: int
    train_val_split: float
    attention_heads: int
    n_layers: int
    dropout: float

@dataclass
class VocabularyConfig:
    n_embd: int

@dataclass
class GenerationConfig:
    max_new_tokens: int
    model_version: int

@dataclass
class Config:
    paths: PathsConfig
    hyperparameters: HyperparametersConfig
    vocabulary: VocabularyConfig
    generation: GenerationConfig
    device: torch.device = torch.device('cuda' if is_available() else 'cpu')

def load_config(path=(Path().cwd() / 'config' / 'config.yaml')) -> Config:
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    return Config(
        paths=PathsConfig(**cfg_dict['paths']),
        hyperparameters=HyperparametersConfig(**cfg_dict['hyperparameters']),
        vocabulary=VocabularyConfig(**cfg_dict['vocabulary']),
        generation=GenerationConfig(**cfg_dict['generation']),
    )

CONFIG = load_config()
