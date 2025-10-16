import argparse
import sys
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from flows.core import logger
from flows import cli


@hydra.main(version_base='1.1.0', config_path='config', config_name='config')
def main(cfg: DictConfig) -> Optional[float]:
    entry = cli.register_task(cfg.get('task', None))
    return entry(cfg)


if __name__ == '__main__':
    # fixes hydra changing cwd
    sys.argv.append('hydra.job.chdir=False')
    sys.argv.append('hydra.run.dir=.experiments/logs/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
