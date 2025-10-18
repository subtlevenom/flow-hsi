from omegaconf import DictConfig
from tools.utils import text
import tools


def main(config: DictConfig) -> None:
    text.print(config)
    # delegate to the tools
    tools.main(config)
