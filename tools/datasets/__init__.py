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


def split(
    type: str,
    source_path: str,
    target_path: str,
    output_path: str,
    split: dict,
    **kwargs,
) -> None:

    match type:
        case 'cave-hsi':
            cave_hsi.splitter.split(source_path, target_path, output_path, split)
        case 'icvl':
            icvl.splitter.split(source_path, target_path, output_path, split)


def normalize(
    type: str,
    gt_path: str,
    src_path: str,
    output_path: str,
    **kwargs,
) -> None:

    match type:
        case 'cave-hsi':
            cave_hsi.normalizer.normalize(gt_path, src_path, output_path)
        case 'icvl':
            icvl.normalizer.normalize(gt_path, src_path, output_path)


def generate(
    type: str,
    input_path: str,
    output_path: str,
    split: dict = None,
    params: DictConfig = None,
    **kwargs,
) -> None:

    match type:
        case 'cave-hsi':
            cave_hsi.generator.generate(input_path, output_path)
        case 'icvl':
            icvl.generator.generate(split, params)
