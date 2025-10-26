import torch
import torch.fft
import cv2
import numpy as np
import os
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.io import savemat
from .lens import Lens
from .bayer import Bayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# TODO: pass into parameters. How???
BANDS = np.arange(400, 710, 10)
BLUE_MAX = 0.5
GREEN_MAX = 0.6
RED_MAX = 0.7


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

    def __call__(self, image: np.ndarray, harmonics=6):

        lambda_for_lens = np.arange(0.4, 0.71, 0.01)
        spectral_filters = self.bayer.get_filters(resample=600, zeros=400)

        image_channels = image.shape[-1]
        image_size = image.shape[0]
        image_shape = (image_size, image_size)

        sum_image_B = np.zeros(image_shape, dtype=np.float64)
        sum_image_G = np.zeros(image_shape, dtype=np.float64)
        sum_image_R = np.zeros(image_shape, dtype=np.float64)

        phase_in = torch.zeros(image_size, dtype=torch.float32, device=device)

        # Aperture mask
        aperture_mask_lens = self.lens.aperture_mask(image_size, self.dx_lens)

        hypercube = []

        for iterator_lenz in range(0, len(lambda_for_lens), 1):
            sum_img = np.zeros(image_shape, dtype=np.float64)
            lens_phase = self.lens.lens_phase(
                lambda_for_lens[iterator_lenz] * 1e-6,
                image_size,
                self.dx_lens,
            )
            for channel_index in range(image.shape[-1]):
                amp_in = self.scale_channel_tensor(image[:, :, channel_index],
                                                   image_size)

                temp_lambd = int(BANDS[channel_index]) * 1e-9

                field = amp_in * torch.exp(1j * phase_in)
                field = self.fresnel_propagation(field.type(torch.complex64),
                                                 temp_lambd, self.lens.z1,
                                                 self.dx_image)
                field = self.resample_field(field,
                                            dx_src=self.dx_image,
                                            dx_dst=self.dx_lens)

                field = field * aperture_mask_lens.type(field.dtype)

                phi_lambda = lens_phase * (lambda_for_lens[iterator_lenz] *
                                           1e-6 / temp_lambd)
                phi_lambda = torch.exp(1j * phi_lambda)

                field = field * phi_lambda
                field = self.fresnel_propagation(field.type(torch.complex64),
                                                 temp_lambd, self.lens.z2,
                                                 self.dx_lens)
                field = self.resample_field(field,
                                            dx_src=self.dx_lens,
                                            dx_dst=self.dx_camera)

                intensity = torch.abs(field).cpu().numpy()

                temp_lambd_cpu = int(temp_lambd * 1e9)

                if lambda_for_lens[iterator_lenz] < BLUE_MAX: spectral_filter = spectral_filters[0]
                elif lambda_for_lens[iterator_lenz] < GREEN_MAX: spectral_filter = spectral_filters[1]
                else: spectral_filter = spectral_filters[2]
                
                sum_img += intensity * spectral_filter[temp_lambd_cpu]

            hypercube.append(sum_img)

        hypercube = np.stack(hypercube, dtype=np.float64).transpose(1, 2, 0)
        return hypercube[::-1,::-1]

    def scale_channel_tensor(self, img: np.ndarray,
                             image_size: int) -> torch.Tensor:
        img = cv2.resize(img, (image_size, image_size),
                         interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype('float32') / 255.0).to(device)

    def generate_lens_phase(self, lambda_design, dx_len):
        coords = (torch.arange(self.image_size, device=device) -
                  self.image_size // 2).float()
        x = coords * dx_len
        y = coords * dx_len
        X, Y = torch.meshgrid(x, y, indexing="xy")
        r2 = X**2 + Y**2
        phi_design = -math.pi / (lambda_design * self.lens.focal_length) * r2
        return phi_design

    def fresnel_propagation(self, U_in, wavelength, z, dx):
        ny, nx = U_in.shape
        k = 2 * math.pi / wavelength
        fx = torch.fft.fftfreq(nx, d=dx).to(device)
        fy = torch.fft.fftfreq(ny, d=dx).to(device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        sqrt_arg = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
        H = torch.exp(1j * k * z * torch.sqrt(sqrt_arg.type(torch.complex64)))
        U_f = torch.fft.fft2(U_in)
        U_out = torch.fft.ifft2(U_f * H)
        return U_out

    def resample_field(self, U_complex, dx_src, dx_dst):
        Nn = U_complex.shape[0]
        U_real = torch.real(U_complex).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        U_imag = torch.imag(U_complex).unsqueeze(0).unsqueeze(0)
        U_ch = torch.cat([U_real, U_imag], dim=1)  # (1,2,H,W)
        coords = (torch.arange(Nn, device=device) - Nn // 2).float()
        x_dst = coords * dx_dst
        y_dst = coords * dx_dst
        Xd, Yd = torch.meshgrid(
            x_dst, y_dst, indexing='xy')  # physical coords of target grid
        L_src = Nn * dx_src
        Xn = Xd / (L_src / 2.0)
        Yn = Yd / (L_src / 2.0)
        grid = torch.stack([Xn, Yn], dim=-1)  # shape (N,N,2)
        grid = grid.unsqueeze(0)  # (1,N,N,2)
        sampled = F.grid_sample(U_ch,
                                grid,
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=True)
        real_s = sampled[0, 0]
        imag_s = sampled[0, 1]
        return (real_s + 1j * imag_s)
