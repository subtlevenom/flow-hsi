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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Lens:

    def __init__(
        self,
        focal_length,
        radius,
        refractive_index,
        height,
    ):
        self.focal_length = focal_length
        self.radius = radius
        self.refractive_index = refractive_index
        self.height = height

        self.z1 = focal_length * 2  # distance to lens
        self.z2 = focal_length * 2  # distance to camera

    def get_lambda(self, height:float, order: int = 6):
        lambda_for_lens = []
        for m in range(1, order + 1):
            refraction = (self.refractive_index - 1) * height / order
            lambd = round(refraction * 1e3, 0)
            if lambd > 400 and lambd < 800:
                lambda_for_lens.append(lambd)
        lambda_for_lens.reverse()
        return lambda_for_lens

    def aperture_mesh(self, image_size: int, dx: float) -> torch.Tensor:
        """aperture mesh"""

        coords = (torch.arange(image_size, device=device) -
                  image_size // 2).float()
        x = coords * dx
        y = coords * dx
        X, Y = torch.meshgrid(x, y, indexing="xy")

        return X**2 + Y**2

    def aperture_mask(self, image_size: int, dx: float) -> torch.Tensor:
        """round aperture mask"""

        aperture_mesh = self.aperture_mesh(image_size, dx)
        aperture_mask = (aperture_mesh <= self.radius**2)
        return aperture_mask  #.to(device)

    def lens_phase(self, lambda_design, image_size: int, dx: float) -> torch.Tensor:
        """lens phase"""

        aperture_mesh = self.aperture_mesh(image_size, dx)
        phi_design = -math.pi / (lambda_design * self.focal_length) * aperture_mesh
        return phi_design

    def harmonic_lens_phase(self, wavelength, image_size: int, dx: float, height) -> torch.Tensor:
        """harmonic lens phase"""

        phi_parabolic = self.lens_phase(wavelength, image_size, dx)
        phi_max = 2 * math.pi * (self.refractive_index - 1) * height / wavelength
        phi_harmonic = torch.remainder(phi_parabolic, phi_max)
        return phi_harmonic
