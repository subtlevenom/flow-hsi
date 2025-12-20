import numpy as np
import scipy.io as sio
import h5py
import math
from pathlib import Path
from multiprocessing import Pool
import os


def _psnr(img_a, img_b):
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(255.0 / math.sqrt(mse))
    return psnr


def norm_channel(ch_res, ch_gt_255):
    #print(ch_res.shape)
    #print(ch_gt_255.shape)

    x = ch_res.ravel()
    y = ch_gt_255.ravel()

    m, c = np.polyfit(x, y, 1)

    # нормализуем res по среднему значению gt
    ch_res_norm = ch_res * m + c

    # клипим res
    ch_res_255 = np.clip(ch_res_norm, 0, 255)

    return ch_res_255, m, c


def load_mat(mat_path, mat_key, print_keys=False):
    try:
        if print_keys:
            mat_contents = sio.loadmat(mat_path, spmatrix=False)
            print(sorted(mat_contents.keys()))
        
        else:
            np_mat = sio.loadmat(mat_path)[mat_key]

    except:
        
        if print_keys:
            with h5py.File(mat_path, 'r') as mat:
                print(list(mat.keys()))        

        with h5py.File(mat_path, 'r') as mat:
            np_mat = np.array(mat[mat_key])

    return np_mat


def compute_channel_norm(data_dir, gt_dir, gt_key, ch_num=4, moveaxis=None):

    data_path = Path(data_dir)
    gt_path = Path(gt_dir)

    print(f"data dir: {data_path.resolve()}\n")

    c_all = None
    m_all = None

    count = 0
    for hsi_path in data_path.rglob('*gyper.npy'):
        
        count += 1
        hsi_name = hsi_path.parts[-1]
        print(f"{ch_num = }, {hsi_name}")
        hsi_res = np.load(data_dir + "/" + hsi_name)

        gt_path = (gt_dir + "/" + hsi_name).replace("_gyper.npy", ".mat")
        #mat_contents = sio.loadmat(gt_path, spmatrix=False)
        #print(sorted(mat_contents.keys()))
        #0/0
        
        #!!!hsi_gt = sio.loadmat(gt_path)['hsi']
        
        #print(sio.loadmat(gt_path)['lbl'])
        #hsi_gt = sio.loadmat(gt_path)[gt_key]
        hsi_gt = load_mat(gt_path, gt_key)
        if moveaxis is not None:
            #print(f"------------- {hsi_gt.shape}")
            hsi_gt = np.moveaxis(hsi_gt, *moveaxis)
            #print(f"+++++++++++++ {hsi_gt.shape}")
        
        
        #0/0
        
        hsi_gt *= 255

        [ch_res_255, m, c] = norm_channel(hsi_res[:,:,ch_num], hsi_gt[:,:,ch_num])
        print(f"{m = }, {c = }")
        if m_all is None:
            c_all = np.array([c])
            m_all = np.array([m])

            x = hsi_res[:,:,ch_num]
            x_all = x.reshape(x.shape + (1,))
    
            y = hsi_gt[:,:,ch_num]
            y_all = y.reshape(y.shape + (1,))

        else:
            c_all = np.append(c_all, c)
            m_all = np.append(m_all, m)

            
            x = hsi_res[:,:,ch_num]
            try:
                x_all = np.append(x_all, x.reshape(x.shape + (1,)), axis=2)
            except:
                print(f"++++++++++    {hsi_name} {x.shape} {y.shape}")

            
            y = hsi_gt[:,:,ch_num]
            y_all = np.append(y_all, y.reshape(y.shape + (1,)), axis=2)

        if count >=200:
            break

    print(m_all)
    print(np.mean(m_all))
    print(np.mean(c_all))

    x = x_all.ravel()
    y = y_all.ravel()

    m, c = np.polyfit(x, y, 1)
    print(f"{m = }, {c = }")

    return m, c


def normalize_ds(data_dir, gt_dir, out_dir, gt_key, moveaxis=None):
    #hsi_name = "pompoms.npy"
    #out_dir = "C:/_ya/ftp/Hydra.DS/CAVE/norm_251209"
    #out_dir = '/data/nikonorov/Hydra.DS/CAVE/norm_251211/'
    #out_dir = '/data/nikonorov/Hydra.DS/CAVE/norm_251215/'
    
    coefs = np.load(f"{out_dir}/_coefs.npy")
    psnr_by_pic = []
    data_path = Path(data_dir)
    gt_path = Path(gt_dir)
    for hsi_path in data_path.rglob('*_gyper.npy'):
        
        hsi_name = hsi_path.parts[-1]

        hsi_res = np.load(data_dir + "/" + hsi_name)

        hsi_name = hsi_path.parts[-1].replace("_gyper.npy", ".npy")

        gt_path = (gt_dir + "/" + hsi_name).replace(".npy", ".mat")

        #hsi_gt = sio.loadmat(gt_path)[gt_key] * 255
        hsi_gt = load_mat(gt_path, gt_key) * 255
        if moveaxis is not None:
            hsi_gt = np.moveaxis(hsi_gt, *moveaxis)

        psnr_all = np.zeros((31))
        for ch_num in range(31):
            m = coefs[ch_num, 0]
            c = coefs[ch_num, 1]
            res_norm = hsi_res[:,:,ch_num] * m + c

            res_255 = np.clip(res_norm, 0, 255)

            psnr_all[ch_num] = _psnr(hsi_gt[:,:,ch_num], res_255)

            hsi_res[:,:,ch_num] = res_255


        psnr_mean = np.mean(psnr_all)
        print(f"{hsi_name}: {psnr_mean}")
        psnr_by_pic.append(psnr_mean)
        np.save(out_dir+"/"+hsi_name, hsi_res)


    psnr_total = np.mean(np.array(psnr_by_pic))
    print(psnr_total)


def test_ds(data_dir, gt_dir, gt_key='hsi', moveaxis=None):
    
    psnr_by_pic = []
    data_path = Path(data_dir)
    gt_path = Path(gt_dir)
    for hsi_path in data_path.rglob('*_gyper.npy'):
        
        hsi_name = hsi_path.parts[-1]

        hsi_res = np.load(data_dir + "/" + hsi_name)

        hsi_name = hsi_path.parts[-1].replace("_gyper.npy", ".npy")

        gt_path = (gt_dir + "/" + hsi_name).replace(".npy", ".mat")
        # hsi_gt = sio.loadmat(gt_path)[gt_key] * 255
        hsi_gt = load_mat(gt_path, gt_key) * 255
        if moveaxis is not None:
            hsi_gt = np.moveaxis(hsi_gt, *moveaxis)

        psnr_all = np.zeros((31))
        for ch_num in range(31):
            psnr_all[ch_num] = _psnr(hsi_gt[:,:,ch_num], hsi_res[:,:,ch_num])


        psnr_mean = np.mean(psnr_all)
        print(f"{hsi_name}: {psnr_mean}")
        psnr_by_pic.append(psnr_mean)

    psnr_total = np.mean(np.array(psnr_by_pic))
    print(psnr_total)


def compute_channel_norm_parallel(args):
    data_dir, gt_dir, gt_key, ch_num, moveaxis = args
    [m, c] = compute_channel_norm(data_dir, gt_dir, gt_key, ch_num, moveaxis)
    return ch_num, m, c


def main():
    data_dir = 'C:/tmp/Hydra.CAVE/_reshaped'
    gt_dir = '/data/nikonorov/Hydra.DS/CAVE/CAVE.gt'
    gt_key = 'hsi'
    moveaxis = None

    
    data_dir = '/data/nikonorov/Hydra.DS/CAVE/cave_251211'
    data_dir = '/data/nikonorov/Hydra.DS/CAVE/cave_251215'

    data_dir = '/data/nikonorov/Hydra.DS/CAVE/cave_251216'
    out_dir = '/data/nikonorov/Hydra.DS/CAVE/norm_251216'

    # dataset9
    
    data_dir = '/data/nikonorov/Hydra.DS/CAVE/cave_251216_dataset9/NO_norm'
    out_dir = '/data/nikonorov/Hydra.DS/CAVE/norm_251216_ds9'

    # 5e-06_1512_0.00055_0.02_crop1.00
    data_dir = '/data/nikonorov/Hydra.DS/CAVE/5e-06_1512_0.00055_0.02_crop1.00/NO_norm'
    out_dir = '/data/nikonorov/Hydra.DS/CAVE/norm_251216_5e-06_1512'

    # 5e-06_1512_0.00055_0.02_crop1.00
    data_dir = '/data/nikonorov/Hydra.DS/CAVE_v8/NO_norm'
    out_dir = '/data/nikonorov/Hydra.DS/CAVE/norm_cave_v8'

    
    # CZ_hsdb
    data_dir = '/data/nikonorov/Hydra.DS/V8/CZ_hsdb/5e-06_1512_0.0006_0.02_crop1.00/NO_norm'
    out_dir = '/data/nikonorov/Hydra.DS/V8/CZ_hsdb/5e-06_1512_0.0006_0.02_crop1.00/channel_norm'
    gt_dir = '/data/nikonorov/Hydra.DS/V8/CZ_hsdb/gt'
    gt_key = 'ref'

    # ICVL
    data_dir = '/data/nikonorov/Hydra.DS/V8/ICVL/NO_norm'
    out_dir = '/data/nikonorov/Hydra.DS/V8/ICVL/channel_norm'
    gt_dir = '/data/nikonorov/Hydra.DS/V8/ICVL/gt/mat/mat'
    gt_key = 'rad'
    moveaxis = (0,2)


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)



    # считаем и записываем коэфты нормировки по всему датасету, по одному каналу
    #recalc_coefs = False
    recalc_coefs = True
    if recalc_coefs:
        
        coefs = np.zeros((31, 2))
        args_list = [(data_dir, gt_dir, gt_key, ch_num, moveaxis) for ch_num in range(31)]

        with Pool() as pool:
            results = pool.map(compute_channel_norm_parallel, args_list)
    
        # Заполняем матрицу coefs результатами
        for ch_num, m, c in results:
            coefs[ch_num, 0] = m
            coefs[ch_num, 1] = c

        #for ch_num in range(31):
        #    [m, c] = compute_channel_norm(data_dir, gt_dir, ch_num)
        #    coefs[ch_num, 0] = m
        #    coefs[ch_num, 1] = c

        np.save(f"{out_dir}/_coefs.npy", coefs)
    
    # нормализуем датасет с применением сохраненых коэфтов
    normalize_ds(data_dir, gt_dir, out_dir, gt_key, moveaxis)

    return


main()
