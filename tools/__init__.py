from omegaconf import DictConfig
from .files import convert
from .datasets import sample, generate


def main(config:DictConfig) -> None:

    match config.tool:
        case 'convert':
            convert(
                input_path=config.input_path,
                output_path=config.output_path,
                suffix=config.suffix,
                params=config.params,
            )
        case 'sample':
            sample(config)
        case 'generate':
            generate(config)
