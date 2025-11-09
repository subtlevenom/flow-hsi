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

# TODO: pass into parameters. How???
BANDS = np.arange(400, 701, 10)  # lens focusing wavelength
MAX_BLUE = BANDS[10]
MAX_GREEN = BANDS[20]
MAX_RED = BANDS[-1] + 1


class CDF:

    def __init__(
        self,
        lens: Lens,
        bayer: Bayer,
        dx_image: float = 3.5e-6,
        dx_lens: float = 3.5e-6,
        dx_camera: float = 3.5e-6,
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

        # mass_amp_in = ["400.png", "410.png", "420.png", "430.png", "440.png", "450.png", "460.png", "470.png", "480.png", "490.png", "500.png", "510.png", "520.png", "530.png", "540.png", "550.png", "560.png", "570.png", "580.png", "590.png", "600.png", "610.png", "620.png", "630.png", "640.png", "650.png", "660.png", "670.png", "680.png", "690.png", "700.png"]

    def __call__(self, image: np.ndarray, padding: int = 0, harmonics=6):

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

        image_channels = image.shape[-1]
        image_size = image.shape[0]
        image_shape = (image_size, image_size)

        spectral_filters = self.bayer.get_filters(resample=600, zeros=400)
        spectral_filters = np.repeat(
            np.array(spectral_filters),
            [image_channels // 3, image_channels // 3, image_channels - 2 * image_channels // 3],
            axis=0,
        )

        phase_in = torch.zeros(image_size, dtype=torch.float32, device=DEVICE)

        # Aperture mask
        aperture_mask_lens = self.lens.aperture_mask(image_size, self.dx_lens)

        hyperspec = []

        for lambda_index, lens_lambda in enumerate(BANDS):
            sum_img = np.zeros(image_shape, dtype=np.float64)

            lens_phase = self.lens.lens_phase(
                lens_lambda * 1e-9,
                image_size,
                self.dx_lens,
            )

            for channel_index, channel_lambda in enumerate(BANDS):
                channel_image = image[:,:,channel_index]
                channel_tensor = torch.from_numpy(channel_image).to(device=DEVICE)

                field = channel_tensor * torch.exp(1j * phase_in)
                field = self.fresnel_propagation(
                    field.type(torch.complex64),
                    channel_lambda * 1e-9,
                    self.lens.z1,
                    self.dx_image,
                )
                field = field * aperture_mask_lens.type(field.dtype)

                phi_lambda = torch.exp(1j * lens_phase)
                field = field * phi_lambda
                field = self.fresnel_propagation(
                    field.type(torch.complex64),
                    channel_lambda * 1e-9,
                    self.lens.z2,
                    self.dx_lens,
                )

                intensity = torch.abs(field).cpu().numpy()

                spectral_filter = spectral_filters[lambda_index]
                sum_img += intensity * spectral_filter[int(channel_lambda)]

            hyperspec.append(sum_img)

        hyperspec = np.stack(hyperspec, dtype=np.float64).transpose(1, 2, 0)
        hyperspec = center_crop(image=hyperspec[:-1,:-1])['image']

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
