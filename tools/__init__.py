from omegaconf import DictConfig
from .files import convert, compare
from .datasets import sample, generate


def main(config:DictConfig) -> None:

    match config.tool:
        case 'compare':
            compare(**config)
        case 'convert':
            convert(**config)
        case 'sample':
            sample(**config)
        case 'generate':
            generate(**config)
