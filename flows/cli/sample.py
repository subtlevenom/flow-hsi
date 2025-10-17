from omegaconf import DictConfig
from tools.utils import text
from tools.datasets import sample


def main(config: DictConfig) -> None:
    text.print(config)
    # delegate to the tools
    sample(config)
