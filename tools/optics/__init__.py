from pathlib import Path

from omegaconf import DictConfig
from .bayer import Bayer
from .lens import Lens
# from .cdf_band import CDF
# from .cdf_const import CDF
from .cdf_var import CDF


def create_cdf(config:DictConfig):
    filter_path = Path(__path__[0]).joinpath('cmv_400_graph')
    bayer = Bayer(filter_path)
    lens = Lens(
        focal_length=0.2,
        radius=0.03,
        refractive_index=1.62,
        height=4.38,
    )
    return CDF(
        lens=lens,
        bayer=bayer,
        configs=config.selected_configs,
        dx_image=10.0e-6,
        dx_lens=10.0e-6,
        dx_camera=10.0e-6,
    )
