import math

import cv2
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from scipy.interpolate import interp1d
from tqdm import tqdm
from scipy.io import loadmat

from .bayer import Bayer
from .lens import Lens

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_grayscale(img, Nlocal):
    img = cv2.resize(img, (Nlocal, Nlocal), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img.astype('float32') / 255).to(DEVICE)
    return x, img * 255

def load_matfile(path, key):
    cube = loadmat(path)[key]
    return cube.transpose(2, 0, 1)


class CDF:
    START_WAVELENGTH = 400
    END_WAVELENGTH = 701
    STEP = 10
    IMAGE_SIZE = 512
    IMAGE_PADDING = 512
    SHIFT_Y = -1
    SHIFT_X = -1

    def __init__(
        self,
        lens: Lens,
        bayer: Bayer,
        configs: list,
        dx_image: float,
        dx_lens: float,
        dx_camera: float,
    ):
        self.dx_image = dx_image
        self.dx_lens = dx_lens
        self.dx_camera = dx_camera
        self.lens = lens
        self.bayer = bayer
        self.configs = configs or []

        filters = self.bayer.get_filters(resample=600, zeros=400)
        self.spectral_filters = np.stack(filters, axis=0).astype(np.float64) / 50.0

        self.freq_grid_cache = {}
        self.transfer_cache = {}
        self.lens_phase_cache = {}

    def __call__(self, image: np.ndarray, padding: int = 0):
        self.IMAGE_PADDING = padding
        dataset_wave = {}

        available_wavelengths = np.arange(self.START_WAVELENGTH, self.END_WAVELENGTH, self.STEP)
        wavelength_idx = self.START_WAVELENGTH
        for fname in load_matfile(image, 'hsi'):
            i_in, img_start = load_grayscale(fname, self.IMAGE_SIZE)
            padded = F.pad(i_in, (padding, padding, padding, padding), mode='constant', value=0)
            amp_in = torch.sqrt(padded)
            dataset_wave[wavelength_idx] = (amp_in, img_start)
            wavelength_idx += self.STEP

        if not dataset_wave:
            raise ValueError('Input cube does not contain valid spectral slices.')

        results = {}
        for cfg in tqdm(self.configs):
            h_lens = cfg.get('h')
            order = cfg.get('M', 0)
            lambda_for_lens = self.lens.get_lambda(height=h_lens, order=order)
            if not lambda_for_lens:
                continue
            self._simulate_config(
                dataset_wave=dataset_wave,
                available_wavelengths=available_wavelengths,
                lambda_for_lens=lambda_for_lens,
                h_lens=h_lens,
                results=results,
            )

        if not results:
            raise ValueError('No wavelengths were produced during simulation.')

        return self._postprocess(results)

    def _get_frequency_grid(self, size: int, dx: float):
        key = (size, round(dx, 12))
        grid = self.freq_grid_cache.get(key)
        if grid is None:
            freq = torch.fft.fftfreq(size, dx).to(DEVICE)
            grid = torch.meshgrid(freq, freq, indexing='xy')
            self.freq_grid_cache[key] = grid
        return grid

    def _fresnel_propagation(self, field: torch.Tensor, wavelength: float, z: float,
                             dx: float):
        size = field.shape[-1]
        key = (size, round(dx, 12), round(wavelength, 12), round(z, 6))
        transfer = self.transfer_cache.get(key)
        if transfer is None:
            fx, fy = self._get_frequency_grid(size, dx)
            transfer = torch.exp(-1j * math.pi * wavelength * z *
                                 (fx**2 + fy**2))
            self.transfer_cache[key] = transfer
        spectrum = torch.fft.fft2(field)
        propagated = torch.fft.ifft2(spectrum * transfer)
        return propagated

    def _generate_harmonic_lens_phase(self, wavelength: float, size: int,
                                      dx: float, height: float):
        key = (size, round(dx, 12), round(wavelength, 12), round(height or 0.0, 6))
        phase = self.lens_phase_cache.get(key)
        if phase is None:
            coords = (torch.arange(size, device=DEVICE) - size // 2).float() * dx
            X, Y = torch.meshgrid(coords, coords, indexing='xy')
            r2 = X**2 + Y**2
            phi_parabolic = -math.pi / (wavelength * self.lens.focal_length) * r2
            phi_max = 2 * math.pi * (self.lens.refractive_index - 1) * (height or self.lens.height) / wavelength
            phase = torch.remainder(phi_parabolic, phi_max)
            self.lens_phase_cache[key] = phase
        return phase

    def _simulate_config(
        self,
        dataset_wave: dict,
        available_wavelengths: list,
        lambda_for_lens: list,
        h_lens: float,
        results: dict,
    ):
        if not dataset_wave:
            return

        amp_example = next(iter(dataset_wave.values()))[0]
        n_local = amp_example.shape[-1]
        phase_in = torch.zeros((n_local, n_local), dtype=torch.float32, device=DEVICE)
        center = n_local // 2
        half = self.IMAGE_SIZE // 2
        crop_slice = slice(center - half, center + half)

        sum_image_B = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.float64)
        sum_image_G = np.zeros_like(sum_image_B)
        sum_image_R = np.zeros_like(sum_image_B)

        for wavelength_nm in available_wavelengths:
            if wavelength_nm not in dataset_wave:
                continue
            amp_in, img_start = dataset_wave[wavelength_nm]
            wavelength_m = wavelength_nm * 1e-9

            field = amp_in * torch.exp(1j * phase_in)
            field = self._fresnel_propagation(field.to(torch.complex64),
                                              wavelength_m, self.lens.z1,
                                              self.dx_image)
            
            lens_phase = self._generate_harmonic_lens_phase(
                wavelength=wavelength_m,
                size=n_local,
                dx=self.dx_lens,
                height=h_lens,
            )

            field = field * torch.exp(1j * lens_phase)
            field = self._fresnel_propagation(field.to(torch.complex64),
                                              wavelength_m, self.lens.z2,
                                              self.dx_camera)

            intensity = (torch.abs(field)**2).detach().cpu().numpy()
            intensity_cropped = intensity[crop_slice, crop_slice]

            denom = np.ptp(intensity_cropped)
            if denom:
                intensity_cropped = ((intensity_cropped - intensity_cropped.min()) /
                                     denom) * 255.0
            else:
                intensity_cropped = np.zeros_like(intensity_cropped)

            mean1 = intensity_cropped.mean()
            mean2 = img_start.mean()

            scale = (mean2 / mean1) if mean1 else 0.0
            intensity_cropped = np.clip(intensity_cropped * scale, 0, 255).astype(np.uint8)

            idx = int(wavelength_nm)
            sum_image_B += intensity_cropped * self.spectral_filters[0, idx]
            sum_image_G += intensity_cropped * self.spectral_filters[1, idx]
            sum_image_R += intensity_cropped * self.spectral_filters[2, idx]

        for wavelength in lambda_for_lens:
            results[wavelength] = self._select_channel(
                wavelength,
                sum_image_B,
                sum_image_G,
                sum_image_R,
            )

    @staticmethod
    def _select_channel(wavelength, sum_B, sum_G, sum_R):
        if wavelength < 500:
            return sum_B
        if wavelength > 600:
            return sum_R
        return sum_G

    def _postprocess(self, results: dict):
        ordered = dict(sorted(results.items()))
        if not ordered:
            raise ValueError('No results to postprocess.')

        wavelengths = np.array(list(ordered.keys()), dtype=np.float64)
        images = np.stack([ordered[w] for w in wavelengths], axis=1)

        target_wavelengths = np.arange(self.START_WAVELENGTH,
                                       self.END_WAVELENGTH,
                                       self.STEP,
                                       dtype=np.float64)

        if len(wavelengths) > 1:
            interpolator = interp1d(
                wavelengths,
                images,
                axis=1,
                kind='linear',
                fill_value='extrapolate',
            )
            cube = interpolator(target_wavelengths)
        else:
            cube = np.repeat(images, len(target_wavelengths), axis=1)

        global_min = cube.min()
        global_max = cube.max()
        if global_max > global_min:
            cube = (cube - global_min) / (global_max - global_min)
        else:
            cube = np.zeros_like(cube)

        cube = (cube * 65535.0).clip(0, 65535).astype(np.uint16)
        cube = np.rot90(cube.transpose(1,0,2), 2, axes=(1,2)).transpose(1,2,0)
        return cube


 