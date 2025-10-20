from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class Bayer():

    def __init__(self, filter_path: List[str]):
        """filter_path - path to filter dir"""

        self.bands = self._load_filter(filter_path)

    def _load_filter(self, path: str) -> np.ndarray:

        path: Path = Path(path)
        if not path.exists():
            print(f'No filter dir found at {path}.')
            raise ValueError(f'No filter dir found at {path}.')

        try:
            bands = []
            for f in sorted(path.glob('**/*.txt')):
                band = self._read_spectral_band(f)
                bands.append(band)

            return bands

        except Exception as e:
            print(f'Failed reading filter at {path} with exception {e}')
            raise RuntimeError(f'Failed reading filter at {path} with exception {e}')

    def _read_spectral_band(self, path: Path):
        df = pd.read_csv(path, sep='\t', header=None)
        return df.to_numpy(dtype=np.float32)

    def _resample_band(self, band, new_length, kind='linear'):
        old_indices = np.linspace(0, 1, num=band.shape[0])
        new_indices = np.linspace(0, 1, num=new_length)
        interpolator = interp1d(old_indices, band, kind=kind, axis=0)
        return interpolator(new_indices)

    def _prepend_zeros(self, band, zeros=400):
        zeros = np.zeros(zeros, dtype=band.dtype)
        return np.concatenate([zeros, band], axis=0)

    def get_filters(self, resample=600, zeros=400):
        filters = []
        for band in self.bands:
            filter = band[:,1]
            if resample > 0:
                filter = self._resample_band(filter, new_length=resample)
            if zeros > 0:
                filter = self._prepend_zeros(filter, zeros=zeros)
            filters.append(filter)
        return filters

