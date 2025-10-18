from omegaconf import DictConfig
from .files import convert


def main(config:DictConfig) -> None:
    match config.tool:
        case 'convert':
            convert(
                input_path=config.input_path,
                output_path=config.output_path,
                suffix=config.suffix,
                params=config.params,
            )
    return None
