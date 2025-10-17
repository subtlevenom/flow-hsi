import numpy as np
from scipy.io import loadmat, savemat

for name in [
    "balloons",
    # "beads",
    # "cd",
    # "chart_and_stuffed_toy",
    # "clay",
    # "cloth",
    # "egyptian_statue",
    # "face",
    # "fake_and_real_beers",
    # "fake_and_real_food",
    # "fake_and_real_lemon_slices",
    # "fake_and_real_lemons",
    # "fake_and_real_strawberries",
    # "fake_and_real_sushi",
    # "fake_and_real_tomatoes",
    # "feathers",
    # "flowers",
    # "glass_tiles",
    # "hairs",
    #"corner_dot",
    ]:

    PATH_TO_WL = "for_git/wl_cave.txt"
    PATH_TO_FILTERS = "for_git/cmv_400_clip"
    PATH_TO_DIR = f"/Users/mac/Desktop/CAVE_pronin/Train/{name}"
    CAVE_WL = np.loadtxt(PATH_TO_WL)

    def read_filter(path: str):
        return np.loadtxt(path, dtype=np.float64)

    RED_FILTER = read_filter(f"{PATH_TO_FILTERS}/graph_R.txt")
    GREEN_FILTER = read_filter(f"{PATH_TO_FILTERS}/graph_G.txt")
    BLUE_FILTER = read_filter(f"{PATH_TO_FILTERS}/graph_B.txt")

    def interpolate_coef(wl_l, wl_h, coef_l, coef_h, x):
        return coef_l + (coef_h - coef_l) * (x - wl_l) / (wl_h - wl_l)

    def load_mat(path: str):
        matfile = loadmat(path)
        red_channel = matfile["Red"]
        green_channel = matfile["Green"]
        blue_channel = matfile["Blue"]
        return np.array([red_channel, green_channel, blue_channel])


    def calculate_coeffs(nm: int):
        for i in range(1, len(RED_FILTER)):
            wl_l = RED_FILTER[i-1][0]
            wl_h = RED_FILTER[i][0]
            if wl_l <= nm <= wl_h:
                red_coef = interpolate_coef(wl_l, wl_h, RED_FILTER[i-1][1], RED_FILTER[i][1], nm)
                break

        for i in range(1, len(GREEN_FILTER)):
            wl_l = GREEN_FILTER[i-1][0]
            wl_h = GREEN_FILTER[i][0]
            if wl_l <= nm <= wl_h:
                green_coef = interpolate_coef(wl_l, wl_h, GREEN_FILTER[i-1][1], GREEN_FILTER[i][1], nm)
                break

        for i in range(1, len(BLUE_FILTER)):
            wl_l = BLUE_FILTER[i-1][0]
            wl_h = BLUE_FILTER[i][0]
            if wl_l <= nm <= wl_h:
                blue_coef = interpolate_coef(wl_l, wl_h, BLUE_FILTER[i-1][1], BLUE_FILTER[i][1], nm)
                break

        return red_coef, green_coef, blue_coef

    def process_channel(rgb_array, c):
        nm = CAVE_WL[c]
        red_c, green_c, blue_c = calculate_coeffs(nm)
        rgb_array[0] = rgb_array[0] * red_c
        rgb_array[1] = rgb_array[1] * green_c
        rgb_array[2] = rgb_array[2] * blue_c
        
        sum_array = rgb_array[0] + rgb_array[1] + rgb_array[2]
        sum_array = np.rot90(sum_array, 2)
        return sum_array

    def create_hsi(path_to_dir):
        new_hsi = np.zeros((512, 512, 31))
        for i in range(31):
            current_channel = load_mat(f"{path_to_dir}/{i}.mat")
            current_channel = process_channel(current_channel, i)
            new_hsi[...,i] = current_channel

        # normalize
        #new_hsi = (new_hsi - new_hsi.min()) / (new_hsi.max() - new_hsi.min())
        
        savemat(f"{path_to_dir}/preprocess_limits.mat",
        {"hsi": new_hsi}
        )
        return new_hsi

    create_hsi(PATH_TO_DIR)
