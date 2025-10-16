from omegaconf import DictConfig
from tools.utils import text
from tools.dataset import sample


def main(config: DictConfig) -> None:
    text.print(config)
    # delegate to the tools
    sample(config)
