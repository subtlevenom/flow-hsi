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


class CDF:

    def __init__(self,
        image_size = 1024,
        dx_in = 3.5e-6,
        dx_lenz = 3.5e-6,
        dx_kam = 3.5e-6,
        focal_length = 0.05,
        r_lenz = 0.003,
        n_prel = 1.62,
        h = 4.38,
        M = 6,
        ):
        """"
        image_size = 1024 Input image size (both width and hight)
        dx_in = 3.5e-6  # шаг пикселя входного распределения
        dx_lenz = 3.5e-6  # шаг линзы
        dx_kam = 3.5e-6  # шаг пикселя камеры
        focal_length = 0.05 Lens' focal length in meters

        r_lenz = 0.003 *no comments*

        z1 = focal_length * 2 distance to lens
        z2 = focal_length * 2 distance to camera

        n_prel = 1.62  Resist's refractive index
        h = 4.38  Harmonic lens height
        M = 6  number of harmonics

        """
        self.N = image_size
        self.dx_in = dx_in,
        self.dx_lenz = dx_lenz,
        self.dx_kam = dx_kam,
        self.focal_length = focal_length,
        self.r_lenz = r_lenz,

        self.focal_length = focal_length
        self.z1 = focal_length * 2,
        self.z2 = focal_length * 2,

        # mass_amp_in = ["400.png", "410.png", "420.png", "430.png", "440.png", "450.png", "460.png", "470.png", "480.png", "490.png", "500.png", "510.png", "520.png", "530.png", "540.png", "550.png", "560.png", "570.png", "580.png", "590.png", "600.png", "610.png", "620.png", "630.png", "640.png", "650.png", "660.png", "670.png", "680.png", "690.png", "700.png"]

        self.lambda_for_lens = self.calc_for_lenz(n_prel, M, h)
        self.spectral_filters = self.get_filter_bier(
            self.lambda_for_lens,
            'for_git/cmv_400_graph/',
            show_graph_on=True,
        )

    def __call__(self, image:np.ndarray, spectral_filters, lambda_for_lens, ):
        
        self.main_circl(
            mass_amp_in, # input hs image

            self.spectral_filters,
            self.dx_in,
            self.dx_lenz,
            self.dx_kam,
            self.r_lenz,
            self.focal_length,
            self.lambda_for_lens,
            self.z1,
            self.z2,
            self.N,
            True,
            512,
            True,
        )

    @staticmethod
    def load_grayscale(path, Nlocal):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Can't load image: {path}")
        img = cv2.resize(img, (Nlocal, Nlocal), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img.astype('float32') / 255.0).to(device)

    @staticmethod
    def generate_lens_phase(f, lambda_design, dx_len, Nlocal):
        coords = (torch.arange(Nlocal, device=device) - Nlocal // 2).float()
        x = coords * dx_len
        y = coords * dx_len
        X, Y = torch.meshgrid(x, y, indexing="xy")
        r2 = X**2 + Y**2
        phi_design = -math.pi / (lambda_design * f) * r2
        return phi_design

    @staticmethod
    def fresnel_propagation(U_in, wavelength, z, dx):
        ny, nx = U_in.shape
        k = 2 * math.pi / wavelength
        fx = torch.fft.fftfreq(nx, d=dx).to(device)
        fy = torch.fft.fftfreq(ny, d=dx).to(device)
        FX, FY = torch.meshgrid(fx, fy, indexing='xy')
        sqrt_arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
        H = torch.exp(1j * k * z * torch.sqrt(sqrt_arg.type(torch.complex64)))
        U_f = torch.fft.fft2(U_in)
        U_out = torch.fft.ifft2(U_f * H)
        return U_out

    @staticmethod
    def resample_field(U_complex, dx_src, dx_dst):
        Nn = U_complex.shape[0]
        U_real = torch.real(U_complex).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        U_imag = torch.imag(U_complex).unsqueeze(0).unsqueeze(0)
        U_ch = torch.cat([U_real, U_imag], dim=1)  # (1,2,H,W)
        coords = (torch.arange(Nn, device=device) - Nn // 2).float()
        x_dst = coords * dx_dst
        y_dst = coords * dx_dst
        Xd, Yd = torch.meshgrid(x_dst, y_dst, indexing='xy')  # physical coords of target grid
        L_src = Nn * dx_src
        Xn = Xd / (L_src / 2.0)
        Yn = Yd / (L_src / 2.0)
        grid = torch.stack([Xn, Yn], dim=-1)  # shape (N,N,2)
        grid = grid.unsqueeze(0)  # (1,N,N,2)
        sampled = F.grid_sample(U_ch, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        real_s = sampled[0,0]
        imag_s = sampled[0,1]
        return (real_s + 1j * imag_s)




    def main_circl(mass_amp_in, spectral_filters, dx_in, dx_lenz, dx_kam, r_lenz, focal_length, lambda_for_lens, z1, z2, Nlocal, show_img_on=False, n_otobr=512, save_img_on=False):
        sum_image_B=np.zeros((Nlocal, Nlocal), dtype=np.float64)
        sum_image_G=np.zeros((Nlocal, Nlocal), dtype=np.float64)
        sum_image_R=np.zeros((Nlocal, Nlocal), dtype=np.float64)

        phase_in = torch.zeros((N, N), dtype=torch.float32, device=device)

        # Маска КРУГЛОЙ апертуры
        coords = (torch.arange(Nlocal, device=device) - Nlocal // 2).float()
        x_l = coords * dx_lenz
        y_l = coords * dx_lenz
        XL, YL = torch.meshgrid(x_l, y_l, indexing="xy")
        aperture_mask_lenz = ((XL**2 + YL**2) <= r_lenz**2).to(device)

        hypercube = np.zeros((len(mass_amp_in), Nlocal, Nlocal), dtype=np.float64)
        index_channel = 0
        for temp_name_frame in mass_amp_in:
            for iterator_lenz in range(0, len(lambda_for_lens), 1):
                lens_phase = generate_lens_phase(focal_length, lambda_for_lens[iterator_lenz] * 1e-6, dx_lenz, Nlocal=Nlocal)
                amp_in = load_grayscale('/Users/mac/Desktop/CAVE_pronin/Test/pompoms/'+temp_name_frame, N)

                temp_lambd=int(temp_name_frame.split('.')[0])*1e-9

                field = amp_in * torch.exp(1j * phase_in)
                field = fresnel_propagation(field.type(torch.complex64), temp_lambd, z1, dx_in)
                field = resample_field(field, dx_src=dx_in, dx_dst=dx_lenz)

                field = field * aperture_mask_lenz.type(field.dtype)

                phi_lambda = lens_phase * (lambda_for_lens[iterator_lenz] * 1e-6 / temp_lambd)
                phi_lambda = torch.exp(1j * phi_lambda)
                field = field * phi_lambda

                field = fresnel_propagation(field, temp_lambd, z2, dx_lenz)
                field = resample_field(field, dx_src=dx_lenz, dx_dst=dx_kam)

                intensity = torch.abs(field).cpu().numpy()

                temp_lambd_cpu=int(temp_lambd * 1e9)

                if iterator_lenz==2:
                    sum_image_B += intensity * spectral_filters[0][temp_lambd_cpu]
                elif iterator_lenz == 1:
                    sum_image_G += intensity * spectral_filters[1][temp_lambd_cpu]
                elif iterator_lenz == 0:
                    sum_image_R += intensity * spectral_filters[2][temp_lambd_cpu]

            spectr_image = sum_image_B + sum_image_G + sum_image_R

            if show_img_on:
                cv2.imshow("Spectr", cv2.resize(cv2.normalize(spectr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (n_otobr,n_otobr)))
                cv2.waitKey(1)

            sum_image_B=np.zeros((Nlocal, Nlocal), dtype=np.float64)
            sum_image_G=np.zeros((Nlocal, Nlocal), dtype=np.float64)
            sum_image_R=np.zeros((Nlocal, Nlocal), dtype=np.float64)

            hypercube[index_channel] = spectr_image
            index_channel += 1

        hypercube = hypercube.transpose(1, 2, 0)
        savemat('hypercube.mat', {'hsi': hypercube})

        if show_img_on:
            cv2.destroyAllWindows()

    def __call__(self, image:np.ndarray, **kwargs):

        N = 1024  # Размер изображения
        dx_in = 3.5e-6  # шаг пикселя входного распределения
        dx_lenz = 3.5e-6  # шаг линзы
        dx_kam = 3.5e-6  # шаг пикселя камеры

        focal_length = 0.05  # фокусное расстояние линзы (м)
        r_lenz = 0.003

        z1 = focal_length * 2  # расстояние до линзы
        z2 = focal_length * 2  # расстояние до камеры

        n_prel = 1.62  # показатель преломления резиста
        h = 4.38  # высота гармонической линзы
        M = 6  # число гармоник

        mass_amp_in = ["400.png", "410.png", "420.png", "430.png", "440.png", "450.png", "460.png", "470.png", "480.png", "490.png", "500.png", "510.png", "520.png", "530.png", "540.png", "550.png", "560.png", "570.png", "580.png", "590.png", "600.png", "610.png", "620.png", "630.png", "640.png", "650.png", "660.png", "670.png", "680.png", "690.png", "700.png"]

        lambda_for_lens = self.calc_for_lenz(n_prel, self.M, self.h)
        spectral_filters = self.get_filter_bier(lambda_for_lens, 'for_git/cmv_400_graph/', show_graph_on=True)
        main_circl(mass_amp_in, spectral_filters, dx_in, dx_lenz, dx_kam, r_lenz, focal_length, lambda_for_lens, z1, z2, N, True, 512, True)
