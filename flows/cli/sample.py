from omegaconf import DictConfig, OmegaConf
from flows.tools.utils import text
from flows.tools.datasets import sample


def main(config: DictConfig) -> None:
    text.print_config(config)
    # delegate to the tools
    sample(config)
