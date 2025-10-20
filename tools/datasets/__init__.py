from omegaconf import DictConfig
from tools.utils import text
from .cave_hsi import sampler, generator


def sample(config: DictConfig) -> None:
    match (config.get('type', None)):
        case 'cave-hsi': 
            sampler.sample(config)


def generate(config: DictConfig) -> None:
    match (config.get('type', None)):
        case 'cave-hsi': 
            generator.generate(config)