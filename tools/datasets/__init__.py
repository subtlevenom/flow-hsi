from omegaconf import DictConfig
from tools.utils import text
from . import cave_hsi
from . import icvl


def adapt(
    type: str,
    input_path: str,
    output_path: str,
    params: DictConfig,
    **kwargs,
) -> None:

    match type:
        case 'cave-hsi':
            cave_hsi.adapter.adapt(input_path, output_path, params)
        case 'icvl':
            icvl.adapter.adapt(input_path, output_path, params)

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
            cave_hsi.sampler.sample(input_path, output_path, split, params)
        case 'icvl':
            icvl.sampler.sample(input_path, output_path, split, params)


def generate(
    type: str,
    input_path: str,
    params: DictConfig = None,
    optics: DictConfig = None,
    **kwargs,
) -> None:

    match type:
        case 'cave-hsi':
            cave_hsi.generator.generate(input_path, params, optics)
        case 'icvl':
            icvl.generator.generate(input_path, params, optics)
