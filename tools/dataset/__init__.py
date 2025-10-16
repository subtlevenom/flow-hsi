from omegaconf import DictConfig
from tools.utils import text
from .sr import image_scale
from .hs import cave_hsi


def sample(config: DictConfig) -> None:
    match (config.get('type', 1)):
        case 'sr-scale': 
            image_scale.sample(config)
        case 'cave-hsi': 
            cave_hsi.sample(config)

