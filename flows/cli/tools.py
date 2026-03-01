from omegaconf import DictConfig, OmegaConf
from tools.utils import text
import tools


def main(config: DictConfig) -> None:
    text.print_config(config)
    # delegate to the tools
    tools.main(config)
