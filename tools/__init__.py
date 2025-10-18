from omegaconf import DictConfig
from tools.utils import text
from tools.files.io import Reader, Writer


def main(config: DictConfig) -> None:
    npy = Reader().read('/home/korepanov/work/flow-hsi/.data/cave-hsi/val/source/watercolors_35.npy')
    npy = Writer().write('/home/korepanov/work/flow-hsi/.experiments/temp/water.mat', npy)
    return None

