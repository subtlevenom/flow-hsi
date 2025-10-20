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


def load_grayscale(path, Nlocal):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Can't load image: {path}")
    img = cv2.resize(img, (Nlocal, Nlocal), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype('float32') / 255.0).to(device)


def calc_for_lenz(n_prel, M, h):
    lambda_for_lens = []
    for m in range(1, M + 1):
        if int(((n_prel - 1) * h / m) * 1e3) < 800:
            lambda_for_lens.append((n_prel - 1) * h / m)
    print(f"lambda_for_lens: {lambda_for_lens}")
    return lambda_for_lens


def read_spectral_filters_from_txt(folder_path):
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    if not txt_files:
        print("Нет .txt файлов в папке.")
        return None, []

    filters = []
    wavelengths = None

    for filename in sorted(txt_files):
        filepath = os.path.join(folder_path, filename)
        x_vals = []
        y_vals = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue
                try:
                    x_str, y_str = line.strip().split()
                    x_vals.append(float(x_str))
                    y_vals.append(float(y_str))
                except ValueError:
                    print(f"Ошибка чтения строки в {filename}: {line.strip()}")
                    continue
        if wavelengths is None:
            wavelengths = np.array(x_vals, dtype=np.float32)
        filters.append(np.array(y_vals, dtype=np.float32))

    return wavelengths, filters


def get_filter_bier(center_wavelengths_for_graph_baier,
                    path_for_filt='./cmv_400_graph/',
                    show_graph_on=True):

    wavelengths, spectral_filters = read_spectral_filters_from_txt(
        path_for_filt)

    def interpolate_array(arr, new_length=1000, kind='linear'):

        old_indices = np.linspace(0, 1, num=len(arr))
        new_indices = np.linspace(0, 1, num=new_length)
        interpolator = interp1d(old_indices, arr, kind=kind)
        return interpolator(new_indices)

    spectral_filters[0] = interpolate_array(spectral_filters[0], 600)
    spectral_filters[1] = interpolate_array(spectral_filters[1], 600)
    spectral_filters[2] = interpolate_array(spectral_filters[2], 600)

    def prepend_zeros(arr, num_zeros=400):

        zeros = np.zeros(num_zeros, dtype=arr.dtype)
        return np.concatenate((zeros, arr))

    spectral_filters[0] = prepend_zeros(spectral_filters[0], 400)
    spectral_filters[1] = prepend_zeros(spectral_filters[1], 400)
    spectral_filters[2] = prepend_zeros(spectral_filters[2], 400)

    return spectral_filters


def generate_lens_phase(f, lambda_design, dx_len, Nlocal):
    coords = (torch.arange(Nlocal, device=device) - Nlocal // 2).float()
    x = coords * dx_len
    y = coords * dx_len
    X, Y = torch.meshgrid(x, y, indexing="xy")
    r2 = X**2 + Y**2
    phi_design = -math.pi / (lambda_design * f) * r2
    return phi_design


def fresnel_propagation(U_in, wavelength, z, dx):
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


def resample_field(U_complex, dx_src, dx_dst):
    Nn = U_complex.shape[0]
    U_real = torch.real(U_complex).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    U_imag = torch.imag(U_complex).unsqueeze(0).unsqueeze(0)
    U_ch = torch.cat([U_real, U_imag], dim=1)  # (1,2,H,W)
    coords = (torch.arange(Nn, device=device) - Nn // 2).float()
    x_dst = coords * dx_dst
    y_dst = coords * dx_dst
    Xd, Yd = torch.meshgrid(x_dst, y_dst,
                            indexing='xy')  # physical coords of target grid
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


def main_circl(mass_amp_in,
               spectral_filters,
               dx_in,
               dx_lenz,
               dx_kam,
               r_lenz,
               focal_length,
               lambda_for_lens,
               z1,
               z2,
               Nlocal,
               show_img_on=False,
               n_otobr=512,
               save_img_on=False):
    sum_image_B = np.zeros((Nlocal, Nlocal), dtype=np.float64)
    sum_image_G = np.zeros((Nlocal, Nlocal), dtype=np.float64)
    sum_image_R = np.zeros((Nlocal, Nlocal), dtype=np.float64)

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
            lens_phase = generate_lens_phase(focal_length,
                                             lambda_for_lens[iterator_lenz] *
                                             1e-6,
                                             dx_lenz,
                                             Nlocal=Nlocal)
            amp_in = load_grayscale(
                '/Users/mac/Desktop/CAVE_pronin/Test/pompoms/' +
                temp_name_frame, N)

            temp_lambd = int(temp_name_frame.split('.')[0]) * 1e-9

            field = amp_in * torch.exp(1j * phase_in)
            field = fresnel_propagation(field.type(torch.complex64),
                                        temp_lambd, z1, dx_in)
            field = resample_field(field, dx_src=dx_in, dx_dst=dx_lenz)

            field = field * aperture_mask_lenz.type(field.dtype)

            phi_lambda = lens_phase * (lambda_for_lens[iterator_lenz] * 1e-6 /
                                       temp_lambd)
            phi_lambda = torch.exp(1j * phi_lambda)
            field = field * phi_lambda

            field = fresnel_propagation(field, temp_lambd, z2, dx_lenz)
            field = resample_field(field, dx_src=dx_lenz, dx_dst=dx_kam)

            intensity = torch.abs(field).cpu().numpy()

            temp_lambd_cpu = int(temp_lambd * 1e9)

            if iterator_lenz == 2:
                sum_image_B += intensity * spectral_filters[0][temp_lambd_cpu]
            elif iterator_lenz == 1:
                sum_image_G += intensity * spectral_filters[1][temp_lambd_cpu]
            elif iterator_lenz == 0:
                sum_image_R += intensity * spectral_filters[2][temp_lambd_cpu]

        spectr_image = sum_image_B + sum_image_G + sum_image_R

        if show_img_on:
            cv2.imshow(
                "Spectr",
                cv2.resize(
                    cv2.normalize(spectr_image, None, 0, 255,
                                  cv2.NORM_MINMAX).astype(np.uint8),
                    (n_otobr, n_otobr)))
            cv2.waitKey(1)

        sum_image_B = np.zeros((Nlocal, Nlocal), dtype=np.float64)
        sum_image_G = np.zeros((Nlocal, Nlocal), dtype=np.float64)
        sum_image_R = np.zeros((Nlocal, Nlocal), dtype=np.float64)

        hypercube[index_channel] = spectr_image
        index_channel += 1

    hypercube = hypercube.transpose(1, 2, 0)
    savemat('hypercube.mat', {'hsi': hypercube})

    if show_img_on:
        cv2.destroyAllWindows()


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
        return aperture_mask #.to(device)

    def lens_phase(self, lambda_design, image_size: int,
                   dx: float) -> torch.Tensor:
        """lens phase"""

        aperture_mesh = self.aperture_mesh(image_size, dx)
        phi_design = -math.pi / (lambda_design *
                                 self.focal_length) * aperture_mesh
        return phi_design
