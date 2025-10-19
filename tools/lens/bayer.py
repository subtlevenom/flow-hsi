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


class Bayer:

    def __init__(
        self,
        refractive_index = 1.62,
        lens_height = 4.38,
        harmonics = 6,
    ):
        """
        n_prel = 1.62  Resist's refractive index
        h = 4.38  Harmonic lens height
        M = 6  number of harmonics
        """

        self.lambda_for_lens = self.calc_for_lenz(
            refractive_index,
            lens_height,
            harmonics,
        )
        self.spectral_filters = self.get_filter_bier(
            self.lambda_for_lens,
            'for_git/cmv_400_graph/',
            show_graph_on=True,
        )

    @staticmethod
    def calc_for_lenz(n_prel, h, M):
        lambda_for_lens = []
        for m in range(1, M + 1):
            if int(((n_prel - 1) * h / m) * 1e3) < 800:
                lambda_for_lens.append((n_prel - 1) * h / m)
        print(f"lambda_for_lens: {lambda_for_lens}")
        return lambda_for_lens

    @staticmethod
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

    @staticmethod
    def get_filter_bier(
        center_wavelengths_for_graph_baier,
        path_for_filt='./cmv_400_graph/',
        show_graph_on=True,
    ):
        wavelengths, spectral_filters = Bayer.read_spectral_filters_from_txt(path_for_filt)

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


        if show_graph_on:
            center_wavelengths_for_graph_baier[0] = round(center_wavelengths_for_graph_baier[0] * 1e3, 2)
            center_wavelengths_for_graph_baier[1] = round(center_wavelengths_for_graph_baier[1] * 1e3, 2)
            center_wavelengths_for_graph_baier[2] = round(center_wavelengths_for_graph_baier[2] * 1e3, 2)

            plt.plot(np.linspace(0, 1000, len(spectral_filters[0])), spectral_filters[0], label=f'{center_wavelengths_for_graph_baier[2]} nm', color='blue')
            plt.plot(np.linspace(0, 1000, len(spectral_filters[1])), spectral_filters[1], label=f'{center_wavelengths_for_graph_baier[1]} nm', color='green')
            plt.plot(np.linspace(0, 1000, len(spectral_filters[2])), spectral_filters[2], label=f'{center_wavelengths_for_graph_baier[0]} nm', color='red')
            plt.xlim(0, 1000)

            for x in lambda_for_lens:
                x = int(x)
                plt.axvline(x=x, color='gray', linestyle='--', linewidth=1)
                plt.text(x + 2, 0.01, f'{x} nm', rotation=90, verticalalignment='bottom', fontsize=8)

            plt.title("График чувствительности фильтра")
            plt.xlabel("Длина волны (nm)")
            plt.ylabel("Яркость")
            plt.legend()
            plt.grid(True)
            plt.show()

        return spectral_filters
