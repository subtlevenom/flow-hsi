from omegaconf import DictConfig
from tools.utils import text
from .cave_hsi import sampler, generator


def sample(
    type: str,
    input_path: str,
    output_path: str,
    split: dict,
    params: DictConfig,
    **kwargs,
) -> None:

    match type:
        case 'cave-hsi':
            sampler.sample(input_path, output_path, split, params)


def generate(
    type: str,
    input_path: str,
    params: DictConfig = None,
    **kwargs,
) -> None:

    match type:
        case 'cave-hsi':
            generator.generate(input_path, params)
