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

def get_difraction_eff(lamb_6, num_harmonics_vid, showing_on, ax):
    m = lamb_6[1]+1
    # k_values = np.array([6, 5, 4])

    if num_harmonics_vid == 1:
        k_values = np.array([4])
    elif num_harmonics_vid == 2:
        k_values = np.array([5, 4])
    elif num_harmonics_vid == 3:
        k_values = np.array([6, 5, 4])
    else:
        raise ValueError("num_harmonics должен быть 1, 2 или 3")

    # Диапазон длин волн для расчета
    lamd = np.linspace(400e-9, 800e-9, 400)

    def sinc_phys(x):
        return np.where(x == 0, 1.0, np.sin(x) / x)

    # Вычисление дифракционной эффективности для каждой гармоники k
    eta_array = np.zeros((len(k_values), len(lamd)))
    for i, k in enumerate(k_values):
        eta_array[i, :] = sinc_phys(m * lamb_6[0]*1e-9 / lamd - k) ** 2

    zeros = np.zeros((len(k_values), 400))
    eta_array = np.concatenate((zeros, eta_array), axis=1)


    if showing_on:
        lamd_for_show = np.linspace(0, 800, eta_array.shape[1])
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['blue', 'green', 'red']
        for i, k in enumerate(k_values):
            color = colors[i % len(colors)]
            ax.plot(lamd_for_show, eta_array[i],
                    linestyle='--',
                    color=color,
                    label=f'η k={k}')

        ax.legend()
        ax.grid(True)
    return eta_array

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
    sqrt_arg = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
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


def main_circl(
    path_to_save,
    dataset_wave,
    mass_amp_in,
    dif_filt,
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

    # amp_in = load_grayscale(path_to_arc+str(mass_amp_in[0]), N)
    # -----------------------Padding---------------------------------
    # amp_in = F.pad(amp_in, (N_padding, N_padding, N_padding, N_padding), mode='constant', value=0)
    # Nlocal = amp_in.shape[0]
    # -----------------------Padding---------------------------------

    phase_in = torch.zeros((Nlocal, Nlocal), dtype=torch.float32, device=device)

    # Маска КРУГЛОЙ апертуры
    coords = (torch.arange(Nlocal, device=device) - Nlocal // 2).float()
    x_l = coords * dx_lenz
    y_l = coords * dx_lenz
    XL, YL = torch.meshgrid(x_l, y_l, indexing="xy")
    aperture_mask_lenz = ((XL**2 + YL**2) <= r_lenz**2).to(device)

    for iterator_lenz in range(0, len(lambda_for_lens), 1):
        print(f'lenz number {iterator_lenz} and wave nm: {lambda_for_lens[iterator_lenz]} ')

        lens_phase = generate_lens_phase(focal_length, lambda_for_lens[iterator_lenz] * 1e-9, dx_lenz, Nlocal=Nlocal)


        for temp_name_frame in mass_amp_in:
            # print(temp_name_frame)

            amp_in = dataset_wave[temp_name_frame].clone()
            # i_in = load_grayscale(path_to_arc+temp_name_frame, N)
            # amp_in = torch.sqrt(i_in)

            # -----------------------Padding---------------------------------
            # amp_in = F.pad(amp_in, (N_padding, N_padding, N_padding, N_padding), mode='constant', value=0)
            # -----------------------Padding---------------------------------

            temp_lambd=int(temp_name_frame.split('.')[0])*1e-9

            field = amp_in * torch.exp(1j * phase_in)
            field = fresnel_propagation(field.type(torch.complex64), temp_lambd, z1, dx_in)
            field = resample_field(field, dx_src=dx_in, dx_dst=dx_lenz)

            field = field * aperture_mask_lenz.type(field.dtype)

            # phi_lambda = lens_phase * (lambda_for_lens[iterator_lenz] * 1e-9 / temp_lambd)
            # phi_lambda = torch.exp(1j * phi_lambda)

            field = field * torch.exp(1j * lens_phase)

            field = fresnel_propagation(field, temp_lambd, z2, dx_lenz)
            field = resample_field(field, dx_src=dx_lenz, dx_dst=dx_kam)

            intensity = (torch.abs(field)**2).cpu().numpy()

            # -----------------------Padding---------------------------------
            crop_size = N
            c = Nlocal // 2
            half = crop_size // 2
            crop_slice = slice(c - half, c + half)
            intensity_cropped = intensity[crop_slice, crop_slice]
            # -----------------------Padding---------------------------------

            temp_lambd_cpu=int(temp_lambd * 1e9)

            if iterator_lenz==0:
                sum_image_B += intensity_cropped * spectral_filters[0][temp_lambd_cpu] * dif_filt[0][temp_lambd_cpu]
            elif iterator_lenz == 1:
                sum_image_G += intensity_cropped * spectral_filters[1][temp_lambd_cpu] * dif_filt[1][temp_lambd_cpu]
            elif iterator_lenz == 2:
                sum_image_R += intensity_cropped * spectral_filters[2][temp_lambd_cpu] * dif_filt[2][temp_lambd_cpu]

            if show_img_on:
                cv2.imshow("All", cv2.resize(cv2.normalize(intensity_cropped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (n_otobr,n_otobr)))
                cv2.imshow("Blue", cv2.resize(cv2.normalize(sum_image_B, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (n_otobr,n_otobr)))
                cv2.imshow("Green", cv2.resize(cv2.normalize(sum_image_G, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (n_otobr,n_otobr)))
                cv2.imshow("Red1", cv2.resize(cv2.normalize(sum_image_R, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (n_otobr,n_otobr)))
                cv2.waitKey(1)

    results.append((path_to_save+f"/{lambda_for_lens[0]} B.png", sum_image_B))
    if len(lambda_for_lens) > 1:
        results.append((path_to_save+f"/{lambda_for_lens[1]} G.png", sum_image_G))
    if len(lambda_for_lens) > 2:
        results.append((path_to_save+f"/{lambda_for_lens[2]} R.png", sum_image_R))

    '''
    if save_img_on:
        # считаем общий min/max по всем трём каналам
        global_min = min(sum_image_B.min(), sum_image_G.min(), sum_image_R.min())
        global_max = max(sum_image_B.max(), sum_image_G.max(), sum_image_R.max())

        # нормализация всех трёх каналов в единую шкалу [0, 255]
        norm_B = ((sum_image_B - global_min) / (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)
        norm_G = ((sum_image_G - global_min) / (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)
        norm_R = ((sum_image_R - global_min) / (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)

        # сохраняем отдельно и в RGB
        cv2.imwrite(path_to_save+f"/{lambda_for_lens[0]} B.png", norm_B)
        if len(lambda_for_lens) > 1:
            cv2.imwrite(path_to_save+f"/{lambda_for_lens[1]} G.png", norm_G)
        if len(lambda_for_lens) > 2:
            cv2.imwrite(path_to_save+f"/{lambda_for_lens[2]} R.png", norm_R) 
    '''


    print('END\n\n\nEND')
    if show_img_on:
        cv2.destroyAllWindows()

import json

if __name__ == '__main__':

    # ---------- Настройки оптической системы ----------
    N = 1024  # Размер изображения
    dx_in = 3.5e-6
    dx_lenz = 3.5e-6
    dx_kam = 3.5e-6

    focal_length = 0.05  # фокусное расстояние
    r_lenz = 0.003       # радиус линзы

    z1 = focal_length * 2  # расстояние от источника до линзы
    z2 = focal_length * 2  # расстояние от линзы до камеры

    N_padding = 64

    # ---------- Загрузка JSON ----------
    with open("selected_cover.json", "r") as f:
        data = json.load(f)

    selected_configs = data["selected_configs"]

    # ---------- Подготовка входных данных ----------
    mass_amp_in = [f"{i}.png" for i in range(400, 701, 10)]
    spectral_filters, ax = get_filter_bier('./cmv_400_graph/', show_graph_on=True)

    dataset_wave = {}
    for fname in mass_amp_in:
        i_in = load_grayscale('Chisto_FREN/Archive/' + fname, N)
        amp_in = torch.sqrt(i_in)
        amp_in = F.pad(amp_in, (N_padding, N_padding, N_padding, N_padding), mode='constant', value=0)
        dataset_wave[fname] = amp_in
    Nlocal = dataset_wave[mass_amp_in[0]].shape[0]

    # ---------- Итерация по каждой конфигурации линзы ----------
    for cfg in selected_configs:
        h = cfg["h"]
        M = cfg["M"]
        covers = cfg["covers"]

        print(f"\n    Конфигурация линзы ")
        print(f"h = {h:.6f} µm,  M = {M}")

        covers_list = [(c["lambda"] * 1e3, c["m"]) for c in covers]
        print("Covers:", covers_list)

        lam_covers_nm = [c[0] for c in covers_list]

        print(lam_covers_nm)

        _, ax = get_filter_bier('./cmv_400_graph/', show_graph_on=False)
        dif_filt = get_difraction_eff(covers_list[0], len(covers_list), True, ax)

        ax.legend()
        plt.tight_layout()
        plt.show()

        main_circl(
            '17_',
            dataset_wave,
            mass_amp_in,
            dif_filt,
            spectral_filters,
            dx_in, dx_lenz, dx_kam,
            r_lenz, focal_length,
            lam_covers_nm,  # длины волн в нм
            z1, z2, N, N_padding, Nlocal,
            show_img_on=False,
            n_otobr=512,
            save_img_on=True
        )

    # ---------- Глобальная нормализация ----------
    print("\n=== Глобальная нормализация и сохранение ===")

    all_values = np.concatenate([img.flatten() for _, img in results])
    global_min, global_max = all_values.min(), all_values.max()

    for fname, img in results:
        norm = ((img - global_min) / (global_max - global_min) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(fname + ".png", norm)

    print("✅ Все изображения сохранены с общей нормализацией.")
