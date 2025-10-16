from omegaconf import DictConfig
from tools.utils import text
from .hsi import cave_hsi


def sample(config: DictConfig) -> None:
    match (config.get('type', 1)):
        case 'cave-hsi': 
            cave_hsi.sample(config)

