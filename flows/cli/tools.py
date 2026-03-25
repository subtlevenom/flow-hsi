from omegaconf import DictConfig, OmegaConf
from flows.tools.utils import text
from flows import tools


def main(config: DictConfig) -> None:
    text.print_config(config)
    # delegate to the tools
    tools.main(config)
