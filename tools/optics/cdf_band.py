import torch
import torch.fft
import cv2
import numpy as np
import os
import math
import albumentations as A
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.io import savemat
from .lens import Lens
from .bayer import Bayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BANDS = np.arange(400, 701, 10)  # lens focusing wavelength


class CDF:

    def __init__(
        self,
        lens: Lens,
        bayer: Bayer,
        configs: list,
        dx_image: float,
        dx_lens: float,
        dx_camera: float,
    ):
        """"
        image_size = 1024 Input image size (both width and hight)
        dx_in = 3.5e-6  # image pixes step
        dx_lenz = 3.5e-6  # lens step
        dx_kam = 3.5e-6  # camera pixel step
        M = 6  number of harmonics
        """
        self.dx_image = dx_image
        self.dx_lens = dx_lens
        self.dx_camera = dx_camera

        self.lens = lens
        self.bayer = bayer
        self.configs = configs

    def __call__(self, image: np.ndarray, padding: int = 0):

        # padding/crop
        if padding > 0:
            center_crop = A.CenterCrop(height=image.shape[0],
                                       width=image.shape[1])
            pad = A.Pad(padding=padding,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0)
            image = pad(image=image)['image']
        else:
            center_crop = A.Sequential()

        image_size = image.shape[0]
        image_shape = (image_size, image_size)
        image_channels = image.shape[-1]

        spectral_filters = self.bayer.get_filters(resample=600, zeros=400)
        hyperspec = {}
        
        for cfg in self.configs:
            lens_height = cfg["h"]
            lens_order = cfg["M"]

            lambda_for_lens = self.lens.get_lambda(height=lens_height,
                                                   order=lens_order)
            phase_in = torch.zeros(image_size,
                                   dtype=torch.float32,
                                   device=DEVICE)

            sum_image_B = np.zeros(image_shape, dtype=np.float64)
            sum_image_G = np.zeros(image_shape, dtype=np.float64)
            sum_image_R = np.zeros(image_shape, dtype=np.float64)

            for channel_index, channel_lambda in enumerate(BANDS):
                channel_image = image[:, :, channel_index]
                channel_tensor = torch.from_numpy(channel_image).to(
                    device=DEVICE)

                field = channel_tensor * torch.exp(1j * phase_in)
                field = self.fresnel_propagation(
                    field.type(torch.complex64),
                    channel_lambda * 1e-9,
                    self.lens.z1,
                    self.dx_image,
                )

                lens_phase = self.lens.harmonic_lens_phase(
                    channel_lambda * 1e-9,
                    image_size,
                    self.dx_lens,
                    lens_height,
                ).to(DEVICE)
                field = field * torch.exp(1j * lens_phase)

                field = self.fresnel_propagation(
                    field.type(torch.complex64),
                    channel_lambda * 1e-9,
                    self.lens.z2,
                    self.dx_camera,
                )

                intensity = (torch.abs(field)**2).cpu().numpy()

                sum_image_B += intensity * spectral_filters[0][channel_lambda]
                sum_image_G += intensity * spectral_filters[1][channel_lambda]
                sum_image_R += intensity * spectral_filters[2][channel_lambda]

            def select_channel(wavelength):
                if wavelength < 500: return sum_image_B
                if wavelength > 600: return sum_image_R
                return sum_image_G

            for wavelength in lambda_for_lens:
                hyperspec[wavelength] = select_channel(wavelength)

        sorted_keys = list(sorted(hyperspec.keys()))
        hyperspec = [hyperspec[key] for key in sorted_keys]

        hyperspec = np.stack(hyperspec, axis=-1)

        original_wavelengths = np.array(sorted_keys, dtype=np.float64)
        new_wavelengths = np.arange(400, 701, 10, dtype=np.float64)

        interpolator = interp1d(original_wavelengths,
                                hyperspec,
                                axis=-1,
                                kind='linear',
                                bounds_error=False,
                                fill_value="extrapolate")
        hyperspec = interpolator(new_wavelengths)
        
        hyperspec = center_crop(image=hyperspec[2:, 3:])['image']
        return hyperspec[::-1, ::-1]

    def fresnel_propagation(self, U_in, wavelength, z, dx):
        ny, nx = U_in.shape
        k = 2 * math.pi / wavelength
        fx = torch.fft.fftfreq(nx, d=dx).to(DEVICE)
        fy = torch.fft.fftfreq(ny, d=dx).to(DEVICE)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        sqrt_arg = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
        H = torch.exp(1j * k * z * torch.sqrt(sqrt_arg.type(torch.complex64)))
        U_f = torch.fft.fft2(U_in)
        U_out = torch.fft.ifft2(U_f * H)
        return U_out
