import torch
from torch import nn


class NTIREColorimetry(nn.Module):
    """Relative reflectance, 400:10:700 nm -> XYZ -> CIELAB under D65.

    CIE 1931 2-degree observer and D65 numeric samples, also tabulated at:
    https://github.com/colour-science/colour/tree/develop/colour/colorimetry/datasets
    Trapezoidal integration and the matching band-limited perfect-diffuser white
    avoid assigning a chromatic tint to a flat spectrum.
    """

    def __init__(self):
        super().__init__()
        observer = torch.tensor([
            [0.01431, 0.000396, 0.06785001],
            [0.04351, 0.00121, 0.2074],
            [0.13438, 0.004, 0.6456],
            [0.2839, 0.0116, 1.3856],
            [0.34828, 0.023, 1.74706],
            [0.3362, 0.038, 1.77211],
            [0.2908, 0.06, 1.6692],
            [0.19536, 0.09098, 1.28764],
            [0.09564, 0.13902, 0.8129501],
            [0.03201, 0.20802, 0.46518],
            [0.0049, 0.323, 0.272],
            [0.0093, 0.503, 0.1582],
            [0.06327, 0.71, 0.07824999],
            [0.1655, 0.862, 0.04216],
            [0.2904, 0.954, 0.0203],
            [0.4334499, 0.9949501, 0.008749999],
            [0.5945, 0.995, 0.0039],
            [0.7621, 0.952, 0.0021],
            [0.9163, 0.87, 0.001650001],
            [1.0263, 0.757, 0.0011],
            [1.0622, 0.631, 0.0008],
            [1.0026, 0.503, 0.00034],
            [0.8544499, 0.381, 0.00019],
            [0.6424, 0.265, 4.999999e-5],
            [0.4479, 0.175, 2e-5],
            [0.2835, 0.107, 0],
            [0.1649, 0.061, 0],
            [0.0874, 0.032, 0],
            [0.04677, 0.017, 0],
            [0.0227, 0.00821, 0],
            [0.01135916, 0.004102, 0],
        ], dtype=torch.float32)
        illuminant = torch.tensor([
            82.7549, 91.486, 93.4318, 86.6823, 104.865, 117.008,
            117.812, 114.861, 115.923, 108.811, 109.354, 107.802,
            104.79, 107.689, 104.405, 104.046, 100, 96.3342,
            95.788, 88.6856, 90.0062, 89.5991, 87.6987, 83.2886,
            83.6992, 80.0268, 80.2146, 82.2778, 78.2842, 69.7213,
            71.6091,
        ], dtype=torch.float32)
        integration = torch.full((31,), 10.0)
        integration[[0, -1]] *= 0.5
        response = observer.T * (illuminant * integration)
        response /= response[1].sum()
        self.register_buffer("response", response)
        self.register_buffer("white", response.sum(dim=1).reshape(1, 3, 1, 1))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim != 4 or spectrum.shape[1] != 31:
            raise ValueError("CIE/D65 colorimetry requires 31 ordered bands at 400:10:700 nm.")
        with torch.autocast(device_type=spectrum.device.type, enabled=False):
            xyz = torch.einsum("rc,bchw->brhw", self.response.float(), spectrum.float())
            relative = xyz / self.white.float()
            delta = 6.0 / 29.0
            # Clamp only the inactive cube-root branch: negative predictions
            # retain a finite, nonzero gradient through the linear Lab segment.
            f = torch.where(
                relative > delta ** 3,
                relative.clamp_min(delta ** 3).pow(1.0 / 3.0),
                relative / (3 * delta ** 2) + 4.0 / 29.0,
            )
            x, y, z = f.unbind(dim=1)
            return torch.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), dim=1)
