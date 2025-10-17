from omegaconf import DictConfig
from tools.utils import text
from .cave_hsi import sample


def sample(config: DictConfig) -> None:
    match (config.get('type', 1)):
        case 'cave-hsi': 
            sample.sample(config)

