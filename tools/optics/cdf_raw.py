import torch
import torch.fft
import cv2
import numpy as np
import os
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torch.nn.functional as F
import json


results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_grayscale(path, Nlocal):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Can't load image: {path}")
    img = cv2.resize(img, (Nlocal, Nlocal), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype('float32') / 255.0).to(device)
def load_grayscale_amplitude(path, Nlocal):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Can't load image: {path}")
    img = cv2.resize(img, (Nlocal, Nlocal), interpolation=cv2.INTER_AREA)
    img_float = img.astype('float32') / 255.0
    amp = np.sqrt(img_float)
    return torch.from_numpy(amp).to(device)




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

def get_filter_bier(center_wavelengths_for_graph_baier, path_for_filt='./cmv_400_graph/', show_graph_on=True):

    wavelengths, spectral_filters = read_spectral_filters_from_txt(path_for_filt)

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

    spectral_filters[0] = prepend_zeros(spectral_filters[0], 400)/50
    spectral_filters[1] = prepend_zeros(spectral_filters[1], 400)/50
    spectral_filters[2] = prepend_zeros(spectral_filters[2], 400)/50


    # if show_graph_on:

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.linspace(0, 1000, len(spectral_filters[0])), spectral_filters[0], label='0 lens', color='blue')
    ax.plot(np.linspace(0, 1000, len(spectral_filters[1])), spectral_filters[1], label='1 lens', color='green')
    ax.plot(np.linspace(0, 1000, len(spectral_filters[2])), spectral_filters[2], label='2 lens', color='red')
    ax.set_xlim(0, 1000)
    ax.set_title("Спектральная чувствительность и дифракционная эффективность")
    ax.set_xlabel("Длина волны (нм)")
    ax.set_ylabel("Интенсивность / Эффективность")
    ax.grid(True)
    ax.legend()

    # if return_ax:
    return spectral_filters, ax
    # return spectral_filters

# def generate_lens_phase(f, lambda_design, dx_len, Nlocal):
#     coords = (torch.arange(Nlocal, device=device) - Nlocal // 2).float()
#     x = coords * dx_len
#     y = coords * dx_len
#     X, Y = torch.meshgrid(x, y, indexing="xy")
#     r2 = X**2 + Y**2
#     phi_design = -math.pi / (lambda_design * f) * r2
#     return phi_design

def generate_harmonic_lens_phase(f, wavelength, dx_len, Nlocal, H, n):
    coords = (torch.arange(Nlocal) - Nlocal // 2).float()
    x = coords * dx_len
    y = coords * dx_len
    X, Y = torch.meshgrid(x, y, indexing="xy")
    r2 = X ** 2 + Y ** 2
    phi_parabolic = -math.pi / (wavelength * f) * r2
    phi_max = 2 * math.pi * (n - 1) * H / wavelength
    phi_harmonic = torch.remainder(phi_parabolic, phi_max)
    return phi_harmonic


# def fresnel_propagation(U_in, wavelength, z, dx):
#     ny, nx = U_in.shape
#     k = 2 * math.pi / wavelength
#     fx = torch.fft.fftfreq(nx, d=dx).to(device)
#     fy = torch.fft.fftfreq(ny, d=dx).to(device)
#     FX, FY = torch.meshgrid(fx, fy, indexing='xy')
#     sqrt_arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
#     H = torch.exp(1j * k * z * torch.sqrt(sqrt_arg.type(torch.complex64)))
#     U_f = torch.fft.fft2(U_in)
#     U_out = torch.fft.ifft2(U_f * H)
#     return U_out
def fresnel_propagation(u0, wavelength, z, dx):
    N = u0.shape[0]
    k = 2 * math.pi / wavelength
    fx = torch.fft.fftfreq(N, dx).to(device)
    fy = torch.fft.fftfreq(N, dx).to(device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    H = torch.exp(-1j * math.pi * wavelength * z * (FX ** 2 + FY ** 2))
    U1 = torch.fft.fft2(u0)
    U2 = U1 * H
    u_z = torch.fft.ifft2(U2)
    return u_z



def main_circl(
    path_to_save,
    dataset_wave,
    mass_amp_in,
    h_lenz,
    n_lenz,
    spectral_filters,
    dx_in,
    dx_lenz,
    dx_kam,
    r_lenz,
    focal_length,
    lambda_for_lens,
    z1,
    z2,
    N,
    N_padding,
    Nlocal,
    show_img_on=False,
    n_otobr=512,
    save_img_on=False,
):

    sum_image_B=np.zeros((N, N), dtype=np.float64)
    sum_image_G=np.zeros((N, N), dtype=np.float64)
    sum_image_R=np.zeros((N, N), dtype=np.float64)

    phase_in = torch.zeros((Nlocal, Nlocal), dtype=torch.float32, device=device)

    for temp_name_frame in mass_amp_in:

        amp_in = dataset_wave[temp_name_frame].clone()

        temp_lambd=int(temp_name_frame.split('.')[0])*1e-9

        field = amp_in * torch.exp(1j * phase_in)
        field = fresnel_propagation(field.type(torch.complex64), temp_lambd, z1, dx_in)

        lens_phase = generate_harmonic_lens_phase(
            focal_length,
            temp_lambd,
            dx_lenz,
            Nlocal,
            h_lenz,
            n_lenz,
        ).to(device)
        field = field * torch.exp(1j * lens_phase)

        field = fresnel_propagation(field.type(torch.complex64), temp_lambd, z2, dx_kam)

        intensity = (torch.abs(field)**2).cpu().numpy()

        # -----------------------Padding---------------------------------
        crop_size = N
        c = Nlocal // 2
        half = crop_size // 2
        crop_slice = slice(c - half, c + half)
        intensity_cropped = intensity[crop_slice, crop_slice]
        # -----------------------Padding---------------------------------

        temp_lambd_cpu=int(temp_lambd * 1e9)

        sum_image_B += intensity_cropped * spectral_filters[0][temp_lambd_cpu]
        sum_image_G += intensity_cropped * spectral_filters[1][temp_lambd_cpu]
        sum_image_R += intensity_cropped * spectral_filters[2][temp_lambd_cpu]

    results.append((path_to_save+f"/{lambda_for_lens[0]} B.png", sum_image_B))
    if len(lambda_for_lens) > 1:
        results.append((path_to_save+f"/{lambda_for_lens[1]} G.png", sum_image_G))
    else:
        results.append((path_to_save + f"/{round(h_lenz, 2)} G_over.png", sum_image_G))

    if len(lambda_for_lens) > 2:
        results.append((path_to_save+f"/{lambda_for_lens[2]} R.png", sum_image_R))
    else:
        results.append((path_to_save+f"/{round(h_lenz, 2)} R_over.png", sum_image_R))

    print('END\nEND')
    if show_img_on:
        cv2.destroyAllWindows()

import json

if __name__ == '__main__':

    # ---------- Настройки оптической системы ----------
    N = 1024  # Размер изображения
    dx_in = 10.0e-6
    dx_lenz = 10.0e-6
    dx_kam = 10.0e-6

    focal_length = 0.2  # фокусное расстояние
    r_lenz = 0.03       # радиус линзы

    z1 = focal_length * 2  # расстояние от источника до линзы
    z2 = focal_length * 2  # расстояние от линзы до камеры
    n_prel=1.62
    N_padding = 512

    # ---------- Загрузка JSON ----------
    with open("selected_cover.json", "r") as file:
        data = json.load(file)

    selected_configs = data["selected_configs"]

    # ---------- Подготовка входных данных ----------
    mass_amp_in = [f"{i}.png" for i in range(400, 701, 10)]
    spectral_filters, ax = get_filter_bier('./cmv_400_graph/', show_graph_on=True)

    dataset_wave = {}
    for fname in mass_amp_in:
        # i_in = load_grayscale('Archive/450.png', N)
        i_in = load_grayscale('Archive/' + fname, N)
        # i_in = load_grayscale_amplitude('Archive/' + fname, N)
        # amp_in = torch.sqrt(i_in)
        amp_in = F.pad(i_in, (N_padding, N_padding, N_padding, N_padding), mode='constant', value=0)
        dataset_wave[fname] = amp_in
    Nlocal = dataset_wave[mass_amp_in[0]].shape[0]

    for cfg in selected_configs:
        h = cfg["h"]
        M = cfg["M"]

        lambda_for_lens = []
        for m in range(1, M + 1):
            lamb_temp=round(((n_prel - 1) * h / m) * 1e3, 0)
            if lamb_temp < 800 and lamb_temp>400:  # Пока что нас интересует только видимый диапазон
                lambda_for_lens.append(lamb_temp)
        lambda_for_lens.reverse()

        print(f"h = {h:.6f} µm,  M = {M}")

        main_circl(
            '17_',
            dataset_wave,
            mass_amp_in,
            h, n_prel,
            spectral_filters,
            dx_in, dx_lenz, dx_kam,
            r_lenz, focal_length,
            lambda_for_lens,  # длины волн в нм
            z1, z2, N, N_padding, Nlocal,
            show_img_on=False,
            n_otobr=1024,
            save_img_on=True
        )

    print("\n=== Глобальная нормализация и сохранение ===")

    all_values = np.concatenate([img.flatten() for _, img in results])
    global_min, global_max = all_values.min(), all_values.max()

    for fname, img in results:
        norm = ((img - global_min) / (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(fname, norm)
        print(fname)

    print("✅ Все изображения сохранены с общей нормализацией.")
