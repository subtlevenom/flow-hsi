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
            focal_length=0.05,
            radius=0.05,
            refractive_index=1.62,
            height=4.38  # высота гармонической линзы
    ):
        self.focal_length = focal_length
        self.radius = radius
        self.refractive_index = refractive_index
        self.height = height

        self.z1 = focal_length * 2  # distance to lens
        self.z2 = focal_length * 2  # distance to camera
        self.refractive_power = (self.refractive_index - 1) * self.height

    def get_lambda(self, harmonics: int = 6):
        lens_lambda = []
        for m in range(1, harmonics + 1):
            if int((self.refractive_power / m) * 1e3) < 800:
                lens_lambda.append(self.refractive_power / m)
        return lens_lambda

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

    def lens_phase(self, lambda_design, image_size: int,
                   dx: float) -> torch.Tensor:
        """lens phase"""

        aperture_mesh = self.aperture_mesh(image_size, dx)
        phi_design = -math.pi / (lambda_design *
                                 self.focal_length) * aperture_mesh
        return phi_design
