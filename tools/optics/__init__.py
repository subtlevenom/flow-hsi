from pathlib import Path
from .bayer import Bayer
from .lens import Lens
from .cdf import CDF


def create_cdf():
    filter_path = Path(__path__[0]).joinpath('cmv_400_graph')
    return CDF(lens=Lens(), bayer=Bayer(filter_path))
