import sys
from typing import Optional
import hydra
from omegaconf import DictConfig
from . import main as flows_main


@hydra.main(
    version_base="1.3.2", config_path="../.configs/config", config_name="config"
)
def main(cfg: DictConfig) -> Optional[float]:
    flows_main(cfg)


if __name__ == "__main__":
    # fixes hydra changing cwd
    # fixes hydra logs destination
    sys.argv.append("hydra.job.chdir=False")
    sys.argv.append("hydra.run.dir=.experiments/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}")
    main()
