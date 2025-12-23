from omegaconf import DictConfig
import random
from .files import convert, compare
from .datasets import sample, generate, normalize, split


def main(config:DictConfig) -> None:

    seed = config.get('seed', None)
    if seed is not None:
        random.seed(seed)

    match config.tool:
        case 'compare':
            compare(**config)
        case 'convert':
            convert(**config)
        case 'sample':
            sample(**config)
        case 'generate':
            generate(**config)
        case 'normalize':
            normalize(**config)
        case 'split':
            split(**config)
